"""Audit real four-rank reference/prefix/resume runs and their full saved states.

Only locally produced, trusted checkpoints are loaded, after SHA256 validation.
No tensor tolerance: this gate requires identical dtype, shape and bytes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.checkpoints import validate_checkpoint
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256
from scripts.statewam.r2.cpu_job import alive


def require_exact(left, right, path="root"):
    """Compare semantic checkpoint contents, independently of pickle encoding."""
    if type(left) is not type(right):
        raise ValueError(f"{path}: type differs")
    if torch.is_tensor(left):
        if left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError(f"{path}: tensor dtype/shape differs")
        a = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        b = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        for offset in range(0, a.numel(), 16 * 1024 * 1024):
            if not torch.equal(a[offset:offset + 16 * 1024 * 1024], b[offset:offset + 16 * 1024 * 1024]):
                raise ValueError(f"{path}: tensor bytes differ at block {offset}")
    elif isinstance(left, np.ndarray):
        if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():
            raise ValueError(f"{path}: array differs")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError(f"{path}: keys differ")
        for key in left:
            require_exact(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise ValueError(f"{path}: length differs")
        for index, (a, b) in enumerate(zip(left, right)):
            require_exact(a, b, f"{path}[{index}]")
    elif type(left).__module__ == "deepspeed.runtime.fp16.loss_scaler":
        require_exact(vars(left), vars(right), f"{path}.attributes")
    elif (type(left).__module__, type(left).__name__) == (
            "deepspeed.utils.tensor_fragment", "fragment_address"):
        if vars(left).keys() != {"numel", "start"} or vars(right).keys() != {"numel", "start"}:
            raise ValueError(f"{path}: unexpected fragment address fields")
        require_exact(vars(left), vars(right), f"{path}.attributes")
    elif isinstance(left, (str, bytes, int, float, bool, set, type(None), np.generic)):
        if left != right:
            raise ValueError(f"{path}: value differs")
    else:
        raise TypeError(f"{path}: unsupported checkpoint value {type(left)}")


def audit(reference, prefix, resumed):
    runs = [Path(p).resolve() for p in (reference, prefix, resumed)]
    identities, checkpoints, summaries = [], [], []
    for run in runs:
        state = json.loads((run / "state.json").read_text())
        if (state["state"] != "completed" or alive(state.get("child")) or alive(state.get("supervisor"))
                or any(alive(p) for p in state.get("processes", []))):
            raise ValueError(f"Run not completed and released: {run}")
        training = run / "training"
        identity = json.loads((training / "identity.json").read_text())
        ranks = [json.loads((training / f"summary_rank{r}.json").read_text()) for r in range(4)]
        checkpoint = Path(ranks[0]["checkpoint"]).resolve()
        if not checkpoint.is_relative_to(training / "checkpoints"):
            raise ValueError("Checkpoint not owned by expected training run")
        position = validate_checkpoint(checkpoint, identity=identity)
        pointer = json.loads((training / "checkpoints/latest-complete.json").read_text())
        if pointer != {"checkpoint": checkpoint.name, "complete_sha256": sha256(checkpoint / "COMPLETE.json")}:
            raise ValueError("Latest checkpoint pointer mismatch")
        for row in ranks:
            if row["position"] != position or row["identity"] != identity or row["stopped"]:
                raise ValueError("Rank summary/identity mismatch or stopped run")
        identities.append(identity)
        checkpoints.append(checkpoint)
        summaries.append(ranks)
    full, first, rest = identities
    if not full == first == rest or full["world_size"] != 4:
        raise ValueError("Resume protocol identities differ")
    end = {"optimizer_update": full["updates"], "next_occurrence": full["updates"] * full["global_batch"]}
    if summaries[0][0]["position"] != end or summaries[2][0]["position"] != end:
        raise ValueError("Full update budget not completed")
    if not 0 < summaries[1][0]["position"]["optimizer_update"] < full["updates"]:
        raise ValueError("Prefix must end inside the update budget")
    manifests = [json.loads((r / "training/source_snapshot/manifest.json").read_text()) for r in runs]
    if not manifests[0] == manifests[1] == manifests[2]:
        raise ValueError("Executable source snapshots differ")
    rank_evidence = []
    fields = ("position", "samples", "noise_hashes", "metrics", "gradients")
    for rank in range(4):
        paths = [r / f"training/steps_rank{rank}.jsonl" for r in runs]
        rows = [[json.loads(line) for line in p.read_text().splitlines()] for p in paths]
        expected_microsteps = full["updates"] * full["accumulation"]
        if len(rows[0]) != expected_microsteps or len(rows[1]) + len(rows[2]) != expected_microsteps:
            raise ValueError("Missing or repeated microsteps")
        select = lambda data: [{key: row[key] for key in fields} for row in data]
        require_exact(select(rows[0]), select(rows[1] + rows[2]), f"rank{rank}.steps")
        rank_evidence.append({"rank": rank, "microsteps": expected_microsteps,
                              "sample_noise_loss_gradient_exact": True,
                              "steps_sha256": [sha256(p) for p in paths]})
    files = sorted(p.relative_to(checkpoints[0]) for p in checkpoints[0].rglob("*")
                   if p.is_file() and p.suffix in {".pt", ".pkl"})
    other_files = sorted(p.relative_to(checkpoints[2]) for p in checkpoints[2].rglob("*")
                         if p.is_file() and p.suffix in {".pt", ".pkl"})
    if files != other_files:
        raise ValueError("Serialized state file sets differ")
    states = []
    for relative in files:
        print(f"Comparing {relative}", flush=True)
        # mmap avoids simultaneously copying two whole model checkpoints into RAM.
        a = torch.load(checkpoints[0] / relative, map_location="cpu", weights_only=False, mmap=True)
        b = torch.load(checkpoints[2] / relative, map_location="cpu", weights_only=False, mmap=True)
        require_exact(a, b, str(relative))
        states.append({"file": str(relative), "contents_exact": True})
        del a, b
    return {"goal_id": GOAL_ID, "passed": True, "identity": full,
            "runs": [str(r) for r in runs], "ranks": rank_evidence, "saved_states": states,
            "checkpoint_complete_sha256": [sha256(p / "COMPLETE.json") for p in checkpoints],
            "tolerance": "Exact scalar values and tensor/array dtype, shape and bytes",
            "scheduler": "No LR scheduler in this engineering configuration (constant LR)",
            "scaler": "No active mixed-precision loss scaling in bf16; serialized optimizer metadata compared",
            "limits": "Real E3 orderly save/restart at an update boundary; not forced write interruption, client interrupt, other groups or other batch configurations."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "prefix", "resumed", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous audit evidence")
    try:
        result = audit(args.reference, args.prefix, args.resumed)
    except Exception as error:
        atomic_json(args.output, {"goal_id": GOAL_ID, "passed": False,
                                  "error": f"{type(error).__name__}: {error}"})
        raise
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
