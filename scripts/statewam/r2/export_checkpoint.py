"""Export a verified local ZeRO-2 checkpoint without changing its tensor values."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from openwam.train.utils.ckpt_model_loader import copy_ckpt_artifacts
from scripts.statewam.r2.checkpoints import validate_checkpoint
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256, stable_hash
from scripts.statewam.r2.cpu_job import alive


def deployment_tensors(state, reference):
    module = state["module"]
    if (not module or state["mp_world_size"] != 1 or state["dp_world_size"] != 4
            or any(not name.startswith("architecture.") for name in module)):
        raise ValueError("Require the complete four-rank R2 architecture state")
    tensors = {name.removeprefix("architecture."): value for name, value in module.items()}
    with safe_open(str(reference), framework="pt", device="cpu") as original:
        if set(tensors) != set(original.keys()):
            raise ValueError("Deployment tensor inventory differs from initialization")
        for name, value in tensors.items():
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16
                    or list(value.shape) != original.get_slice(name).get_shape()
                    or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid deployment tensor: {name}")
    return {name: value.contiguous() for name, value in tensors.items()}


def export(checkpoint, matrix, output):
    started = time.time()
    checkpoint, matrix, output = (Path(path).resolve() for path in (checkpoint, matrix, output))
    training = checkpoint.parent.parent
    run_state = json.loads((training.parent / "state.json").read_text())
    if (run_state["state"] != "completed" or run_state["exit_code"] != 0
            or alive(run_state.get("supervisor")) or alive(run_state.get("child"))
            or any(alive(row) for row in run_state.get("processes", []))):
        raise ValueError("Export requires a completed, released source run")
    identity = json.loads((training / "identity.json").read_text())
    position = validate_checkpoint(checkpoint, identity=identity)
    if (identity["goal_id"] != GOAL_ID or identity["group"] not in ("E0", "E1", "E2", "E3")
            or position["optimizer_update"] != identity["updates"]):
        raise ValueError("Require the registered final update, not an intermediate checkpoint")
    initialization = json.loads((matrix / "initialization.json").read_text())
    if (sha256(matrix / "initialization.json") != identity["matrix_sha256"]
            or json.loads((matrix / "COMPLETE.json").read_text())["initialization_sha256"] != identity["matrix_sha256"]):
        raise ValueError("Export matrix differs from training initialization")
    reference = matrix / identity["group"] / "checkpoint_step_0.safetensors"
    group = initialization["groups"][identity["group"]]
    if sha256(reference) != group["checkpoint_sha256"]:
        raise ValueError("Initialization weights changed")
    cfg = OmegaConf.load(training / "config.yaml")
    if (stable_hash(OmegaConf.to_container(cfg, resolve=True)) != identity["config_sha256"]
            or cfg.r2.group != identity["group"]
            or sha256(cfg.dataloader.normalization_stats_path) != identity["statistics_sha256"]):
        raise ValueError("Export configuration/statistics differ from trained identity")
    sources = list((checkpoint / "accelerate").rglob("*_model_states.pt"))
    if len(sources) != 1:
        raise ValueError("Expected one non-tensor-parallel model state")
    # Only deserialize the local checkpoint after checking every published shard.
    state = torch.load(sources[0], map_location="cpu", mmap=True, weights_only=False)
    tensors = deployment_tensors(state, reference)
    output.mkdir(parents=True, exist_ok=False)
    name = f"checkpoint_step_{position['optimizer_update']}.safetensors"
    temporary = output / f"{name}.partial"
    save_file(tensors, str(temporary))
    with safe_open(str(temporary), framework="pt", device="cpu") as restored:
        if set(restored.keys()) != set(tensors):
            raise ValueError("Exported tensor inventory differs")
        for key, value in tensors.items():
            loaded = restored.get_tensor(key)
            if loaded.dtype != value.dtype or not torch.equal(loaded, value):
                raise ValueError(f"Export changed tensor values: {key}")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.rename(temporary, output / name)
    shutil.copy2(training / "config.yaml", output / "config.yaml")
    shutil.copy2(cfg.dataloader.normalization_stats_path, output / "normalization_stats.npy")
    copy_ckpt_artifacts(str(training), str(output))
    if sha256(output / "normalization_stats.npy") != identity["statistics_sha256"]:
        raise ValueError("Copied deployment statistics changed")
    files = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    report = {"goal_id": GOAL_ID, "protocol_id": "r2-deployment-export-v1", "passed": True,
        "identity": identity, "position": position, "source_checkpoint": str(checkpoint),
        "source_complete_sha256": sha256(checkpoint / "COMPLETE.json"), "files": files,
        "weight_name": name, "tensor_count": len(tensors), "all_tensor_values_exact": True,
        "started_at": started, "ended_at": time.time(),
        "limits": "Inference weights only, not a resumable optimizer checkpoint; engineering exports are not formal results"}
    atomic_json(output / "export.json", report)
    atomic_json(output / "COMPLETE.json", {"goal_id": GOAL_ID, "export_sha256": sha256(output / "export.json")})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = export(args.checkpoint, args.matrix, args.output)
    print(json.dumps({key: report[key] for key in ("passed", "weight_name", "tensor_count", "all_tensor_values_exact")}))


if __name__ == "__main__":
    main()
