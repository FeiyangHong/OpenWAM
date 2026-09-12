"""Read one existing Wuji batch and audit raw/unified contracts without training."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openwam.dataloader.registry import build_dataset  # noqa: E402
from openwam.dataloader.utils.unify_action import parse_unify_spec, unmap_from_unify  # noqa: E402
from scripts.statewam.capture_manifest import sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(ROOT / "configs/dataloader/wuji_real_task.yaml")
    cfg.dataset_dir = str(args.dataset.resolve())
    raw_cfg = copy.deepcopy(cfg)
    raw_cfg.unify_action = False
    unified_dataset = build_dataset(cfg, split="train")
    raw_dataset = build_dataset(raw_cfg, split="train")
    unified, raw = unified_dataset[0], raw_dataset[0]
    if unified_dataset._fail_count or raw_dataset._fail_count:
        raise RuntimeError("Sample loading retried; cannot claim the requested first window")
    mapping = parse_unify_spec(cfg.unify_action_map)
    unused = np.ones(80, dtype=bool)
    unused[mapping] = False
    shapes = {}
    for key, time_steps in (("action", 32), ("proprio", 1)):
        source, target = raw[key].numpy(), unified[key].numpy()
        assert source.shape == (time_steps, 58) and target.shape == (time_steps, 80)
        assert np.isfinite(source).all() and np.isfinite(target).all()
        np.testing.assert_array_equal(unmap_from_unify(target, mapping), source)
        mask = unified[f"{key}_mask"].numpy()
        np.testing.assert_array_equal(mask[:, mapping], raw[f"{key}_mask"].numpy())
        assert not mask[:, unused].any() and not target[:, unused].any()
        assert mask.sum() == time_steps * 58
        shapes[key] = {"raw": list(source.shape), "unified": list(target.shape), "valid_elements": int(mask.sum())}
    assert len(unified["video"]) == 9 and unified["video_mask"].all()
    frame_std = []
    for frame, other in zip(unified["video"], raw["video"]):
        assert frame.size == (320, 384)
        np.testing.assert_array_equal(np.asarray(frame), np.asarray(other))
        frame_std.append(float(np.asarray(frame).std()))
    assert min(frame_std) > 1 and unified["prompt"] == raw["prompt"]
    unified["video"][0].save(args.output / "first_frame.png")
    row = unified_dataset._eps_df.iloc[0]
    episode = int(row["episode_index"])
    paths = [args.dataset / "meta" / name for name in (
        "info.json", "episodes.jsonl", "tasks.jsonl", "normalization_stats.npy", "adapter_manifest.json",
    )]
    paths.append(args.dataset / f"data/chunk-000/episode_{episode:06d}.parquet")
    for camera in unified_dataset._resolve_cameras(json.loads(paths[0].read_text())):
        paths.append(args.dataset / f"videos/chunk-000/{camera}/episode_{episode:06d}.mp4")
    info = json.loads(paths[0].read_text())
    report = {
        "goal_id": "r1/robotwin-smoke", "benchmark": "wuji_real_task_adapter_only", "pass": True,
        "dataset": str(args.dataset.resolve()), "window": 0, "episode": episode,
        "windows_in_train": len(unified_dataset), "fps_metadata": info.get("fps"),
        "config": OmegaConf.to_container(cfg, resolve=True), "shapes": shapes,
        "mapping": mapping.tolist(), "video_frames": 9, "video_frame_std": frame_std,
        "prompt": unified["prompt"], "normalization": "Existing quantile statistics reused, not refitted",
        "files": {str(p.resolve()): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in paths},
        "scope": "One real dataset window; no model training, robot control or held-out efficacy claim",
    }
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Real Wuji adapter batch passed: {args.output}")


if __name__ == "__main__":
    main()
