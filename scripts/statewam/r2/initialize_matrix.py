"""Create and strictly audit E0--E3 from one Foundation and one State initial value."""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from openwam.train.utils.ckpt_model_loader import copy_ckpt_artifacts
from scripts.statewam.initialize_checkpoint import build_skeleton, load_original_parameters
from scripts.statewam.r2.audit_sources import FOUNDATION, WEIGHT_HASH
from scripts.statewam.r2.common import GOAL_ID, ROOT, atomic_json, sha256
from scripts.statewam.r2.matrix import GROUPS, apply_parameter_policy, group_config
from scripts.statewam.r2.paired_data import tensor_hash


def load_exact_initialization(architecture, path):
    """Preserve checkpoint dtypes before any optimizer owns parameter slots."""
    state = load_file(str(path), device="cpu")
    architecture.load_state_dict(state, strict=True, assign=True)
    restored = architecture.state_dict()
    for key, source in state.items():
        actual = restored[key]
        if actual.is_meta or actual.dtype != source.dtype or actual.shape != source.shape or not torch.equal(actual, source):
            raise ValueError(f"Initialization reload changed tensor: {key}")
    return {"tensor_count": len(state), "all_values_shapes_dtypes_identical": True}


def compare_original(source, converted):
    with safe_open(source, framework="pt", device="cpu") as baseline, safe_open(converted, framework="pt", device="cpu") as state:
        original_keys = set(baseline.keys())
        new_keys = set(state.keys()) - original_keys
        if not new_keys or any(not key.startswith("interaction_state.") for key in new_keys):
            raise ValueError("Unexpected converted keys")
        if original_keys - set(state.keys()):
            raise ValueError("Missing original parameter keys")
        for key in sorted(original_keys):
            a, b = baseline.get_tensor(key), state.get_tensor(key)
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                raise ValueError(f"Original parameter changed: {key}")
        return {"original_tensor_count": len(original_keys), "all_original_values_identical": True,
                "new_state_hashes": {key: tensor_hash(state.get_tensor(key)) for key in sorted(new_keys)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=FOUNDATION)
    parser.add_argument("--initialization-seed", type=int, default=42)
    args = parser.parse_args()
    started = time.time()
    code_files = {str(path.relative_to(ROOT)): sha256(path)
                  for path in (Path(__file__), Path(__file__).with_name("matrix.py"))}
    weights = args.source / "checkpoint_step_154000.safetensors"
    if sha256(weights) != WEIGHT_HASH:
        raise ValueError("Not the pinned Foundation checkpoint")
    if args.output.exists():
        raise FileExistsError("Use a new initialization version; never overwrite a matrix")
    args.output.mkdir(parents=True)
    cfg = group_config(args.source, args.data, "E3")
    torch.manual_seed(args.initialization_seed)
    architecture = build_skeleton(cfg, args.source)
    load_original_parameters(architecture, load_file(str(weights), device="cpu"))
    architecture.interaction_state.to(dtype=next(architecture.action_backbone.parameters()).dtype)
    if any(layer.gate.item() != 0 for layer in architecture.interaction_state.layers.values()):
        raise ValueError("State gates are not zero")
    destination = args.output / "E3/checkpoint_step_0.safetensors"
    destination.parent.mkdir()
    print(f"Saving shared State initialization: {destination}", flush=True)
    partial = destination.with_suffix(".partial")
    architecture.save_checkpoint(str(partial))
    os.replace(partial, destination)
    del architecture
    gc.collect()
    comparison = compare_original(str(weights), str(destination))
    common_state_hash = sha256(destination)
    print("Original tensors unchanged; State value hashes recorded", flush=True)
    records = {}
    for group in GROUPS:
        directory = args.output / group
        directory.mkdir(exist_ok=True)
        path = directory / "checkpoint_step_0.safetensors"
        if group == "E0":
            path.symlink_to(weights.resolve())
        elif group != "E3":
            path.symlink_to(destination.resolve())
        cfg = group_config(args.source, args.data, group)
        copy_ckpt_artifacts(str(args.source), str(directory))
        shutil.copy2(args.data / "normalization_stats.npy", directory / "normalization_stats.npy")
        OmegaConf.save(cfg, directory / "config.yaml")
        restored = build_skeleton(cfg, directory)
        reload_identity = load_exact_initialization(restored, path)
        counts = apply_parameter_policy(restored)
        if any(p.is_meta for p in restored.parameters()):
            raise ValueError("Strict reloading left meta tensors")
        if group != "E0":
            actual = {f"interaction_state.{name}": tensor_hash(tensor) for name, tensor in restored.interaction_state.state_dict().items()}
            if actual != comparison["new_state_hashes"]:
                raise ValueError(f"State initialization differs for {group}")
        records[group] = {"checkpoint": str(path.resolve()), "config_sha256": sha256(directory / "config.yaml"),
                          "checkpoint_sha256": WEIGHT_HASH if group == "E0" else common_state_hash,
                          "strict_reload": True, "reload_identity": reload_identity, "parameter_counts": counts,
                          "statistics_sha256": sha256(directory / "normalization_stats.npy")}
        del restored
        gc.collect()
        print(f"Strict reload {group}: {counts}", flush=True)
    record = {"goal_id": GOAL_ID, "protocol_id": "r2-initialization-v1", "initialization_seed": args.initialization_seed,
              "source_sha256": WEIGHT_HASH, "source_config_sha256": sha256(args.source / "config.yaml"),
              "groups": records, **comparison, "started_at": started, "ended_at": time.time(),
              "limits": "Parameter and strict-load identity only; forward/backward active capacity and zero-gate parity pending",
              "code_files": code_files, "code_hash_timing": "before_initialization"}
    atomic_json(args.output / "initialization.json", record)
    atomic_json(args.output / "COMPLETE.json", {"initialization_sha256": sha256(args.output / "initialization.json")})
    print(json.dumps({group: row["parameter_counts"] for group, row in records.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
