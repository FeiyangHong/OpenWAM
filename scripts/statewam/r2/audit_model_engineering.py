"""Audit real E0--E3 two-update engineering runs without claiming convergence."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.checkpoints import validate_checkpoint
from scripts.statewam.r2.common import ARTIFACTS, GOAL_ID, atomic_json, sha256
from scripts.statewam.r2.cpu_job import alive


def audit(runs):
    all_rows, records = {}, {}
    for group, path in runs.items():
        state = json.loads((path / "state.json").read_text())
        if state["state"] != "completed" or state["exit_code"] != 0 or alive(state.get("child")):
            raise ValueError(f"Not completed: {group}")
        training = path / "training"
        identity = json.loads((training / "identity.json").read_text())
        if identity["group"] != group or identity["world_size"] != 4 or identity["updates"] != 2:
            raise ValueError("Not the registered two-update four-rank protocol")
        checkpoint = training / "checkpoints/update_00000002"
        position = validate_checkpoint(checkpoint, identity=identity)
        latest = json.loads((training / "checkpoints/latest-complete.json").read_text())
        if latest["checkpoint"] != checkpoint.name or latest["complete_sha256"] != sha256(checkpoint / "COMPLETE.json"):
            raise ValueError("Latest pointer integrity failed")
        all_rows[group] = []
        for rank in range(4):
            file = training / f"steps_rank{rank}.jsonl"
            rows = [json.loads(line) for line in file.read_text().splitlines()]
            if len(rows) != 2 or [r["position"]["optimizer_update"] for r in rows] != [1, 2]:
                raise ValueError("Missing model update evidence")
            summary = json.loads((training / f"summary_rank{rank}.json").read_text())
            if summary["identity"] != identity or summary["position"] != position:
                raise ValueError("Rank completion identity differs")
            for row in rows:
                if any(value[1] != 1 for value in row["gradients"].values()):
                    raise ValueError("Nonfinite gradients")
                if any(name.startswith("video_backbone.") for name in row["gradients"]):
                    raise ValueError("Frozen world parameters received gradients")
            if group != "E0":
                state0 = {k: v for k, v in rows[0]["gradients"].items() if k.startswith("interaction_state.")}
                state1 = {k: v for k, v in rows[1]["gradients"].items() if k.startswith("interaction_state.")}
                if len(state0) != 330 or len(state1) != 330 or not all(v[0] > 0 for v in state1.values()):
                    raise ValueError("State parameters not active after gate warm start")
                if {k for k,v in state0.items() if v[0] > 0} != {k for k in state0 if k.endswith(".gate")}:
                    raise ValueError("Unexpected zero-gate gradient routing")
            all_rows[group].append(rows)
        performance = json.loads((path / "performance_summary.json").read_text())
        marker = json.loads((checkpoint / "COMPLETE.json").read_text())
        records[group] = {"run": str(path), "position": position,
                          "identity_sha256": sha256(training / "identity.json"),
                          "checkpoint_complete_sha256": sha256(checkpoint / "COMPLETE.json"),
                          "checkpoint_logical_bytes": sum(r["bytes"] for r in marker["files"].values()),
                          "wall_seconds": performance["wall_seconds"],
                          "allocated_gpu_hours": performance["allocated_gpu_hours"],
                          "peak_allocated_by_rank": [max(r["peak_allocated_bytes"] for r in rows) for rows in all_rows[group]],
                          "save_seconds_by_rank": [json.loads((training / f"save_update2_rank{rank}.json").read_text())["seconds"] for rank in range(4)]}
        print(f"Validated {group} full checkpoint and four-rank steps", flush=True)
    for rank in range(4):
        baseline = all_rows["E0"][rank]
        for group in ("E1", "E2", "E3"):
            for a, b in zip(baseline, all_rows[group][rank]):
                if a["samples"] != b["samples"] or a["noise_hashes"] != b["noise_hashes"]:
                    raise ValueError("Samples/noise are not paired")
            if baseline[0]["metrics"] != all_rows[group][rank][0]["metrics"]:
                raise ValueError("Initial zero-gate loss is not exactly baseline")
    return {"goal_id": GOAL_ID, "protocol_id": "r2-model-engineering-v1", "engineering_passed": True,
            "groups": records, "all_samples_noise_paired": True, "first_loss_exact": True,
            "state_second_step_gradients_nonzero": True,
            "limits": ["Two updates, global batch 4, diagnostic seed only", "Not steady profiling or convergence",
                       "Not output/action parity or exact real-model resume", "Source-token/attention workload and capacity cost audit still pending",
                       "E0/E3 predate source snapshots; E1/E2 have snapshot and drift checks", "NVML process attribution remains unresolved"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new audit output")
    runs = {group: ARTIFACTS / f"runs/p2-model-{group.lower()}-001" for group in ("E0", "E1", "E2", "E3")}
    atomic_json(args.output, audit(runs))


if __name__ == "__main__":
    main()
