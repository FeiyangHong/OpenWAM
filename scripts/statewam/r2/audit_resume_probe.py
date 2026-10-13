"""Verify the independent four-rank reference/prefix/resume engineering evidence."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.checkpoints import validate_checkpoint
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256
from scripts.statewam.r2.cpu_job import alive


def audit(reference, prefix, resumed):
    directories = [Path(path) for path in (reference, prefix, resumed)]
    results = []
    for path in directories:
        state = json.loads((path / "state.json").read_text())
        if state["state"] != "completed" or alive(state.get("child")) or alive(state.get("supervisor")):
            raise ValueError(f"Run not completed and released: {path}")
    for rank in range(4):
        rows = [json.loads((path / f"probe/rank_{rank}.json").read_text()) for path in directories]
        full, first, rest = rows
        if not full["identity"] == first["identity"] == rest["identity"]:
            raise ValueError("Probe identities differ")
        for row in rows:
            if validate_checkpoint(row["checkpoint"], identity=row["identity"]) != row["position"]:
                raise ValueError("Checkpoint/result position mismatch")
        if full["rows"] != first["rows"] + rest["rows"]:
            raise ValueError(f"Rank {rank}: samples, loss or random streams differ")
        for key in ("position", "model", "optimizer", "scheduler"):
            if full[key] != rest[key]:
                raise ValueError(f"Rank {rank}: {key} differs after resume")
        if full["position"] != {"optimizer_update": 4, "next_occurrence": 64}:
            raise ValueError("Probe budget not completed")
        results.append({"rank": rank, "sample_loss_rng_exact": True, "model_optimizer_scheduler_exact": True,
                        "result_sha256": [sha256(path / f"probe/rank_{rank}.json") for path in directories]})
    return {"goal_id": GOAL_ID, "protocol_id": "r2-zero2-probe-v1", "passed": True,
            "runs": [str(path) for path in directories], "ranks": results,
            "limits": "Small synthetic model, orderly save and independent restart. Not real-model recovery, forced write interruption, or client interrupt testing."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "prefix", "resumed", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous audit evidence")
    record = audit(args.reference, args.prefix, args.resumed)
    atomic_json(args.output, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
