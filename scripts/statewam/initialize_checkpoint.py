"""Explicit, audited conversion of a dual-system checkpoint into StateWAM."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openwam.model import build_architecture, resolve_architecture_config  # noqa: E402
from openwam.train.utils.checkpointing import find_latest_weights  # noqa: E402
from openwam.train.utils.ckpt_model_loader import copy_ckpt_artifacts  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_skeleton(cfg, source_dir):
    resolved = resolve_architecture_config(cfg.model)
    params = dict(resolved.params)
    vb = OmegaConf.to_container(cfg.model.video_backbone, resolve=True)
    params["video_backbone"] = {**vb, "_source": vb, "_ckpt_dir": str(source_dir), "_materialize_weights": False}
    return build_architecture(resolved.registry_name, params, allow_experimental=True)


def load_original_parameters(architecture, state):
    expected = architecture.state_dict()
    added = {k for k in expected if k.startswith("interaction_state.")}
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    if missing != added or unexpected:
        raise ValueError(f"Conversion key mismatch: missing={sorted(missing - added)}, unexpected={sorted(unexpected)}")
    mismatches = [k for k in state if state[k].shape != expected[k].shape]
    if mismatches:
        raise ValueError(f"Conversion shape mismatch: {mismatches}")
    if any(not v.is_floating_point() and expected[k].dtype != v.dtype for k, v in state.items()):
        raise ValueError("Conversion cannot change non-floating buffer dtype")
    report = architecture.load_state_dict(state, strict=False, assign=True)
    if set(report.missing_keys) != added or report.unexpected_keys:
        raise RuntimeError("Unexpected load result after validated conversion")
    if any(p.is_meta for p in architecture.parameters()) or any(b.is_meta for b in architecture.buffers()):
        raise RuntimeError("Conversion left unmaterialized meta tensors")
    return sorted(added)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-mode", choices=["current", "future", "future_action", "off"], default="future_action")
    args = parser.parse_args()
    source = args.source.resolve()
    source_dir = source if source.is_dir() else source.parent
    weights = Path(find_latest_weights(str(source))) if source.is_dir() else source
    cfg = OmegaConf.load(source_dir / "config.yaml")
    if (cfg.model.architecture.framework, cfg.model.architecture.variant) != ("dual_system", "joint_self_attn"):
        raise ValueError("Conversion requires dual_system/joint_self_attn")
    if args.output.exists():
        raise FileExistsError(f"Output must be a new directory: {args.output}")
    torch.manual_seed(args.seed)
    state_cfg = OmegaConf.load(ROOT / "configs/model/statewam.yaml").architecture
    cfg.model.architecture = OmegaConf.merge(cfg.model.architecture, state_cfg)
    cfg.model.architecture.interaction_state.source = args.source_mode
    print("Building StateWAM skeleton", flush=True)
    architecture = build_skeleton(cfg, source_dir)
    state = load_file(str(weights), device="cpu")
    added = load_original_parameters(architecture, state)
    del state
    # Match the source compute/storage precision for new parameters only.
    source_dtype = next(architecture.action_backbone.parameters()).dtype
    architecture.interaction_state.to(dtype=source_dtype)
    for layer in architecture.interaction_state.layers.values():
        if layer.gate.item() != 0:
            raise RuntimeError("Conversion must preserve zero gates")
    args.output.mkdir(parents=True, exist_ok=False)
    copy_ckpt_artifacts(str(source_dir), str(args.output))
    stats = source_dir / "normalization_stats.npy"
    if stats.exists():
        shutil.copy2(stats, args.output / stats.name)
    OmegaConf.save(cfg, args.output / "config.yaml")
    destination = args.output / "checkpoint_step_0.safetensors"
    print(f"Saving {destination}", flush=True)
    architecture.save_checkpoint(str(destination))
    report = {
        "source": str(weights), "source_sha256": sha256(weights),
        "source_config_sha256": sha256(source_dir / "config.yaml"),
        "output": str(destination.resolve()), "output_sha256": sha256(destination),
        "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff"], cwd=ROOT)).hexdigest(),
        "seed": args.seed, "new_parameters": added, "storage_dtype": str(source_dtype),
        "unexplained_missing": [], "unexpected": [], "meta_tensors": 0,
    }
    # Include untracked implementation files in the provenance, too.
    report["statewam_code_sha256"] = {
        str(p.relative_to(ROOT)): sha256(p)
        for p in sorted((ROOT / "openwam/model/architectures/statewam").glob("*.py"))
    }
    (args.output / "initialization.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Converted {len(added)} new tensors; report: {args.output / 'initialization.json'}", flush=True)


if __name__ == "__main__":
    main()
