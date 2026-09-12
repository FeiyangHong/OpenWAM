"""Audit real checkpoint generation in eager, compiled and cached modes."""

import argparse
import gc
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openwam.dataloader.registry import build_dataset  # noqa: E402
from openwam.deploy.denoise_schedule import make_schedule  # noqa: E402
from openwam.deploy.model_loader import load_from_checkpoint_dir  # noqa: E402
from openwam.deploy.optimizations import DiTVelocityCache  # noqa: E402
from openwam.train.utils.ckpt_model_loader import copy_ckpt_artifacts  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Eager actions.npy from the original baseline for zero-gate parity")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--roundtrip", action="store_true", help="Save and reload real weights, then require exact eager generation parity")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(args.checkpoint / "config.yaml")
    cfg.dataloader.dataset_dir = str(args.dataset.resolve())
    cfg.dataloader.normalization_stats_path = str((args.checkpoint / "normalization_stats.npy").resolve())
    cfg.dataloader.color_jitter.enabled = False
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    sample = build_dataset(cfg.dataloader, split="train")[0]
    _, model = load_from_checkpoint_dir(str(args.checkpoint), device="cuda:0")
    schedule = make_schedule("sync", video_scheduler=model.video_scheduler,
                             action_scheduler=model.action_scheduler, num_steps=10,
                             shift=model.action_backbone.shift_action, shift_video=model.video_backbone.shift_video)
    kwargs = dict(schedule=schedule, prompt=sample["prompt"], first_frame_image=sample["first_frame_image"],
                  num_frames=len(sample["video"]), action_num_frames=cfg.dataloader.num_frames,
                  height=cfg.dataloader.height, width=cfg.dataloader.width, seed=42,
                  num_inference_steps=10, decode_video=False,
                  proprio=sample["proprio"].to(device=model.device, dtype=model.dtype),
                  _proprio_sample_mask=sample["proprio_mask"].any().reshape(1).to(model.device))
    report = {"goal_id": "r1/robotwin-smoke", "checkpoint": str(args.checkpoint.resolve()),
              "seed": 42, "prompt": sample["prompt"], "episode_path": sample["episode_path"], "checks": {}}

    def generate(name, cache=None):
        if cache is not None:
            cache.reset()
        torch.cuda.synchronize()
        start = time.monotonic()
        actions = model.generate(**kwargs, dit_cache=cache)["actions"]
        torch.cuda.synchronize()
        if actions.shape != (32, 20) or not np.isfinite(actions).all():
            raise RuntimeError(f"Invalid generated RoboTwin actions: {actions.shape}")
        report["checks"][name] = {"seconds": time.monotonic() - start, "shape": list(actions.shape), "finite": True}
        if cache is not None:
            report["checks"][name]["cache_stats"] = cache.stats
        np.save(args.output / f"{name}.npy", actions)
        return actions

    eager = generate("eager")
    repeated = generate("eager_repeat")
    np.testing.assert_array_equal(eager, repeated)
    if args.reference:
        reference = np.load(args.reference)
        np.testing.assert_array_equal(eager, reference)
        report["reference_max_absolute_error"] = float(np.abs(eager - reference).max())
    if args.compile:
        compile_cfg = OmegaConf.load(ROOT / "configs/deploy.yaml").optimization.compile
        model.apply_compile_optimizations(compile_cfg)
        compiled = generate("compiled")
        if model._compiled_mot_run_joint_loop is None:
            raise RuntimeError("Compiled loop fell back to eager")
        # BF16 fusion can change roundoff; report measured errors, not bitwise parity.
        report["compiled_max_absolute_error"] = float(np.abs(compiled - eager).max())
        report["compiled_mean_absolute_error"] = float(np.abs(compiled - eager).mean())
        np.testing.assert_array_equal(compiled, generate("compiled_repeat"))
        report["compile_stats"] = dict(torch._dynamo.utils.counters["stats"])
        cache = DiTVelocityCache(cosine_threshold=0.99, max_consecutive_skips=3)
        cached = generate("compiled_cached", cache)
        report["cached_max_absolute_error_vs_eager"] = float(np.abs(cached - eager).max())
    if args.roundtrip:
        saved = args.output / "roundtrip_checkpoint"
        saved.mkdir()
        copy_ckpt_artifacts(str(args.checkpoint), str(saved))
        for name in ("config.yaml", "normalization_stats.npy"):
            shutil.copy2(args.checkpoint / name, saved / name)
        start = time.monotonic()
        model.save_checkpoint(str(saved / "checkpoint_step_0.safetensors"))
        report["roundtrip_save_seconds"] = time.monotonic() - start
        del model
        gc.collect()
        torch.cuda.empty_cache()
        start = time.monotonic()
        _, model = load_from_checkpoint_dir(str(saved), device="cuda:0")
        report["roundtrip_load_seconds"] = time.monotonic() - start
        np.testing.assert_array_equal(eager, generate("roundtrip_eager"))
        report["roundtrip_max_absolute_error"] = 0.0
    report["pass"] = True
    (args.output / "generation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Generation audit passed: {args.output}")


if __name__ == "__main__":
    main()
