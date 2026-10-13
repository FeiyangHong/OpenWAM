"""Atomic, verified full-state checkpoints for the R2 Accelerate/ZeRO-2 loop."""

import json
import os
from pathlib import Path

import torch.distributed as dist

from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256, stable_hash


def main_operation(accelerator, operation):
    """Publish rank-zero filesystem errors to all ranks before the next collective."""
    result = [None]
    if accelerator.is_main_process:
        try:
            result[0] = {"value": operation()}
        except Exception as error:
            result[0] = {"error": f"{type(error).__name__}: {error}"}
    if accelerator.num_processes > 1:
        dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise RuntimeError(result[0]["error"])
    return result[0]["value"]


def inventory(directory):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Checkpoint cannot contain symlinks: {path}")
        if path.is_file() and path.name != "COMPLETE.json":
            result[str(path.relative_to(directory))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return result


def validate_position(position, identity):
    update = position["optimizer_update"]
    if type(update) is not int or not 0 <= update <= identity["updates"]:
        raise ValueError("Invalid optimizer update")
    if position["next_occurrence"] != update * identity["global_batch"]:
        raise ValueError("Checkpoint is not an optimizer-update exposure boundary")


def validate_checkpoint(directory, *, identity):
    """Read only local trusted checkpoints, before any pickle-backed state load."""
    directory = Path(directory)
    if directory.name.endswith(".partial") or directory.is_symlink():
        raise ValueError("Incomplete or redirected checkpoint")
    complete = directory / "COMPLETE.json"
    if not complete.is_file():
        raise ValueError("Missing complete checkpoint marker")
    marker = json.loads(complete.read_text())
    if marker["identity_sha256"] != stable_hash(identity):
        raise ValueError("Checkpoint protocol/data/initialization identity mismatch")
    if marker["files"] != inventory(directory):
        raise ValueError("Checkpoint file inventory/hash mismatch")
    metadata = json.loads((directory / "position.json").read_text())
    if metadata["identity"] != identity or metadata["goal_id"] != GOAL_ID:
        raise ValueError("Checkpoint metadata identity mismatch")
    validate_position(metadata["position"], identity)
    for rank in range(identity["world_size"]):
        row = json.loads((directory / f"rank_{rank}_ready.json").read_text())
        if row != {"rank": rank, "position": metadata["position"], "identity_sha256": stable_hash(identity)}:
            raise ValueError("Inconsistent rank checkpoint position")
        if f"accelerate/random_states_{rank}.pkl" not in marker["files"]:
            raise ValueError("Missing per-rank RNG state")
    if identity.get("zero_stage") == 2:
        shards = list((directory / "accelerate").rglob("*_optim_states.pt"))
        if len(shards) != identity["world_size"]:
            raise ValueError("Missing ZeRO-2 optimizer shards")
        if not list((directory / "accelerate").rglob("*_model_states.pt")):
            raise ValueError("Missing ZeRO model state")
    return metadata["position"]


def save_checkpoint(accelerator, root, *, identity, position):
    validate_position(position, identity)
    if not accelerator.sync_gradients:
        raise ValueError("Cannot checkpoint during gradient accumulation")
    root = Path(root)
    target = root / f"update_{position['optimizer_update']:08d}"
    partial = target.with_name(target.name + ".partial")

    def create():
        root.mkdir(parents=True, exist_ok=True)
        if target.exists() or partial.exists():
            raise FileExistsError("Checkpoint attempt already exists; preserve it and use a new run")
        partial.mkdir()
        atomic_json(partial / "position.json", {"goal_id": GOAL_ID, "identity": identity, "position": position})

    main_operation(accelerator, create)
    accelerator.save_state(str(partial / "accelerate"))
    atomic_json(partial / f"rank_{accelerator.process_index}_ready.json", {
        "rank": accelerator.process_index, "position": position, "identity_sha256": stable_hash(identity),
    })
    accelerator.wait_for_everyone()

    def publish():
        files = inventory(partial)
        # Flush every shard before publishing a directory that claims completeness.
        for name in files:
            with (partial / name).open("rb") as stream:
                os.fsync(stream.fileno())
        atomic_json(partial / "COMPLETE.json", {"identity_sha256": stable_hash(identity), "files": files})
        os.rename(partial, target)
        validate_checkpoint(target, identity=identity)
        atomic_json(root / "latest-complete.json", {"checkpoint": target.name,
                                                    "complete_sha256": sha256(target / "COMPLETE.json")})
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return str(target)

    return Path(main_operation(accelerator, publish))


def load_checkpoint(accelerator, directory, *, identity):
    position = main_operation(accelerator, lambda: validate_checkpoint(directory, identity=identity))
    accelerator.load_state(str(Path(directory) / "accelerate"))
    accelerator.wait_for_everyone()
    return position
