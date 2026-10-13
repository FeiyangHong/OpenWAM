"""Verify bounded profiling completion, paired exposures, checkpoints and timings."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.checkpoints import validate_checkpoint
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256
from scripts.statewam.r2.cpu_job import alive
from scripts.statewam.r2.exposure_sampler import ExposureBatchSampler
from scripts.statewam.r2.profile_metrics import summarize_steps
from scripts.statewam.r2.profile_training import load_plan


def validate_steps(rows, identity, rank):
    sampler = ExposureBatchSampler(updates=identity["updates"], micro_batch=identity["micro_batch"],
                                    accumulation=identity["accumulation"], world_size=4, rank=rank)
    if len(rows) != len(sampler):
        raise ValueError("Incomplete profiling microstep count")
    for index, (row, expected) in enumerate(zip(rows, sampler)):
        update = (index + 1) // identity["accumulation"]
        if row["position"] != {"optimizer_update": update, "next_occurrence": update * identity["global_batch"]}:
            raise ValueError("Profiling update/exposure cursor mismatch")
        if [sample["occurrence"] for sample in row["samples"]] != expected or len(row["noise_hashes"]) != len(expected):
            raise ValueError("Profiling sample/noise evidence has gaps")
        if not all(math.isfinite(value) for value in row["metrics"].values()):
            raise ValueError("Nonfinite profiling loss")
        for name, (norm, finite) in row["gradients"].items():
            if finite != 1 or not math.isfinite(norm) or norm < 0 or name.startswith("video_backbone."):
                raise ValueError("Invalid profiling parameter/gradient evidence")
        phases = row["phase_timings"]
        if phases["engine_backward"]["calls"] != 1 or phases["forward"]["calls"] != 1:
            raise ValueError("Missing forward/backward timings")
        if (index + 1) % identity["accumulation"] == 0:
            if phases.get("engine_step", {}).get("calls") != 1:
                raise ValueError("Missing optimizer-step timing at update boundary")
        elif "engine_step" in phases:
            raise ValueError("Unexpected optimizer step during accumulation")


def audit_run(path, plan, *, plan_hash, group):
    state = json.loads((path / "state.json").read_text())
    if state["state"] != "completed" or state["exit_code"] != 0 or any(
        alive(item) for item in [state.get("supervisor"), state.get("child"), *state.get("processes", [])]
    ):
        raise ValueError(f"Profiling is not completed and released: {path}")
    training = path / "training"
    identity = json.loads((training / "identity.json").read_text())
    if identity["purpose"] != "profiling" or identity["group"] != group or identity["workload_trace"]:
        raise ValueError("Not an untraced profiling run of the requested group")
    if identity["plan_identity"] != {"plan_sha256": plan_hash, "plan": plan}:
        raise ValueError("Profiling plan identity mismatch")
    for key in ("updates", "micro_batch", "accumulation", "global_batch", "workers", "seed", "world_size", "protocol_id", "matrix_sha256"):
        if identity[key] != plan[key]:
            raise ValueError(f"Profiling plan configuration differs: {key}")
    ranks, compact, counts = [], [], Counter()
    for rank in range(4):
        rows = [json.loads(line) for line in (training / f"steps_rank{rank}.jsonl").read_text().splitlines()]
        validate_steps(rows, identity, rank)
        summary = json.loads((training / f"summary_rank{rank}.json").read_text())
        if summary["identity"] != identity or summary["stopped"] or summary["position"] != rows[-1]["position"]:
            raise ValueError("Rank profiling completion mismatch")
        timing = json.loads((training / f"timing_rank{rank}.json").read_text())
        computed = summarize_steps(rows, accumulation=plan["accumulation"], micro_batch=plan["micro_batch"],
                                   world_size=4, warmup_updates=plan["warmup_updates"])
        if timing["steady_summary"] != computed or timing["identity"] != identity:
            raise ValueError("Profiling timings disagree with raw records")
        ranks.append(computed)
        compact.append([{key: row[key] for key in ("samples", "noise_hashes", "metrics")} for row in rows])
        counts.update(str(sample["task_index"]) for row in rows for sample in row["samples"])
        print(f"Validated {group} rank {rank}: {len(rows)} profiling microsteps", flush=True)
    checkpoints = []
    for update in sorted(set(range(plan["save_every"], plan["updates"] + 1, plan["save_every"])) | {plan["updates"]}):
        checkpoint = training / "checkpoints" / f"update_{update:08d}"
        position = validate_checkpoint(checkpoint, identity=identity)
        if position != {"optimizer_update": update, "next_occurrence": update * plan["global_batch"]}:
            raise ValueError("Profiling checkpoint cursor differs")
        marker = json.loads((checkpoint / "COMPLETE.json").read_text())
        checkpoints.append({"update": update, "path": str(checkpoint), "complete_sha256": sha256(checkpoint / "COMPLETE.json"),
                            "logical_bytes": sum(row["bytes"] for row in marker["files"].values()),
                            "save_seconds_by_rank": [json.loads((training / f"save_update{update}_rank{rank}.json").read_text())["seconds"] for rank in range(4)]})
        print(f"Validated {group} checkpoint {update} full inventory", flush=True)
    latest = json.loads((training / "checkpoints/latest-complete.json").read_text())
    if latest != {"checkpoint": f"update_{plan['updates']:08d}", "complete_sha256": checkpoints[-1]["complete_sha256"]}:
        raise ValueError("Profiling latest-complete mismatch")
    performance = json.loads((path / "performance_summary.json").read_text())
    windows = []
    for start in range(0, plan["updates"], 50):
        per_rank = [rank[start * plan["accumulation"]:min(start + 50, plan["updates"]) * plan["accumulation"]] for rank in compact]
        windows.append({"start_update": start + 1, "end_update": min(start + 50, plan["updates"]),
                        "mean_training_losses": {key: statistics.mean(row["metrics"][key] for rank in per_rank for row in rank)
                                                  for key in compact[0][0]["metrics"]}})
    return {"run": str(path), "identity": identity, "ranks": ranks, "checkpoints": checkpoints,
            "task_exposures": dict(counts), "logical_exposures": sum(counts.values()), "loss_windows": windows,
            "wall_seconds": performance["wall_seconds"], "allocated_gpu_hours": performance["allocated_gpu_hours"],
            "end_to_end_samples_per_second_including_load_save": sum(counts.values()) / performance["wall_seconds"],
            "steady_samples_per_second_slowest_rank": min(rank["global_samples_per_second_excluding_save"] for rank in ranks),
            "performance_sha256": sha256(path / "performance_summary.json")}, compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--e0", type=Path, required=True)
    parser.add_argument("--e3", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing profiling evidence")
    plan = load_plan(args.plan)
    groups, traces = {}, {}
    for group, path in (("E0", args.e0), ("E3", args.e3)):
        if path:
            groups[group], traces[group] = audit_run(path, plan, plan_hash=sha256(args.plan), group=group)
    if args.e3:
        for baseline, state in zip(traces["E0"], traces["E3"]):
            if any(left[key] != right[key] for left, right in zip(baseline, state) for key in ("samples", "noise_hashes")):
                raise ValueError("Profiling groups did not receive paired samples/noise")
            if baseline[0]["metrics"] != state[0]["metrics"]:
                raise ValueError("Zero-gate profiling first loss differs")
    atomic_json(args.output, {"goal_id": GOAL_ID, "plan_sha256": sha256(args.plan),
                             "completed_groups": list(groups), "paired_profiling_passed": bool(args.e3), "groups": groups,
                             "limits": ["Profiling models are excluded from formal experiments",
                                        "Training loss windows are not validation or convergence evidence",
                                        "Rank-local event spans do not sum to actual device FLOPs or kernel runtime",
                                        "Allocated GPU-hours include initialization/checkpoint waits; process attribution remains incomplete"]})


if __name__ == "__main__":
    main()
