"""Require the real-model prefix to match its reference before resuming."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.audit_model_resume import require_exact
from scripts.statewam.r2.checkpoints import validate_checkpoint
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256
from scripts.statewam.r2.cpu_job import alive


def audit(reference, prefix):
    runs = [Path(value).resolve() for value in (reference, prefix)]
    identities, snapshots = [], []
    for run in runs:
        state = json.loads((run / "state.json").read_text())
        if (state["state"] != "completed" or state["exit_code"] != 0 or alive(state.get("supervisor"))
                or alive(state.get("child")) or any(alive(row) for row in state.get("processes", []))):
            raise ValueError("Require completed, released runs")
        identities.append(json.loads((run / "training/identity.json").read_text()))
        snapshot = run / "training/source_snapshot"
        manifest = json.loads((snapshot / "manifest.json").read_text())
        if not manifest:
            raise ValueError("Empty source snapshot")
        for name, digest in manifest.items():
            source = (snapshot / name).resolve()
            if not source.is_relative_to(snapshot) or sha256(source) != digest:
                raise ValueError("Source snapshot content differs from its manifest")
        snapshots.append(manifest)
    identity = identities[0]
    if identities[0] != identities[1] or snapshots[0] != snapshots[1] or identity["world_size"] != 4:
        raise ValueError("Prefix configuration/source differs from reference")
    prefix_summary = json.loads((runs[1] / "training/summary_rank0.json").read_text())
    end = prefix_summary["position"]["optimizer_update"]
    if not 0 < end < identity["updates"]:
        raise ValueError("Prefix must end strictly inside the reference budget")
    checkpoint = Path(prefix_summary["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(runs[1] / "training/checkpoints"):
        raise ValueError("Prefix checkpoint is not owned by this run")
    position = validate_checkpoint(checkpoint, identity=identity)
    pointer = json.loads((checkpoint.parent / "latest-complete.json").read_text())
    if pointer != {"checkpoint": checkpoint.name, "complete_sha256": sha256(checkpoint / "COMPLETE.json")}:
        raise ValueError("Prefix latest checkpoint pointer mismatch")
    if position != {"optimizer_update": end, "next_occurrence": end * identity["global_batch"]}:
        raise ValueError("Prefix checkpoint position differs from summary")
    fields = ("position", "samples", "noise_hashes", "metrics", "gradients")
    ranks = []
    for rank in range(4):
        paths = [run / f"training/steps_rank{rank}.jsonl" for run in runs]
        rows = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
        if len(rows[1]) != end * identity["accumulation"] or len(rows[0]) != identity["updates"] * identity["accumulation"]:
            raise ValueError("Incomplete reference/prefix trace")
        for run, expected in zip(runs, (
                {"optimizer_update": identity["updates"], "next_occurrence": identity["updates"] * identity["global_batch"]},
                position)):
            summary = json.loads((run / f"training/summary_rank{rank}.json").read_text())
            if summary["position"] != expected or summary["stopped"] or summary["identity"] != identity:
                raise ValueError("Rank summary position/identity mismatch or stopped run")
        select = lambda values: [{key: row[key] for key in fields} for row in values]
        require_exact(select(rows[0][:len(rows[1])]), select(rows[1]), f"rank{rank}.prefix")
        ranks.append({"rank": rank, "prefix_microsteps": len(rows[1]), "exact": True,
                      "steps_sha256": [sha256(path) for path in paths]})
    return {"goal_id": GOAL_ID, "passed": True, "identity": identity, "ranks": ranks,
            "runs": [str(run) for run in runs], "prefix_checkpoint": str(checkpoint),
            "prefix_complete_sha256": sha256(checkpoint / "COMPLETE.json"),
            "limits": "Pre-resume gate: exact sample/noise/loss/gradient prefix and complete saved state; final resumed state audit still required"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "prefix", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous audit")
    try:
        result = audit(args.reference, args.prefix)
    except Exception as error:
        atomic_json(args.output, {"goal_id": GOAL_ID, "passed": False,
                                 "error": f"{type(error).__name__}: {error}"})
        raise
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
