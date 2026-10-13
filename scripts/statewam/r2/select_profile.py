"""Review the finite E3 tuning candidates and emit a paired profiling plan."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256
from scripts.statewam.r2.cpu_job import alive
from scripts.statewam.r2.profile_metrics import summarize_steps


def read_candidate(path):
    state = json.loads((path / "state.json").read_text())
    if state["state"] != "completed" or state["exit_code"] != 0 or any(
        alive(row) for row in [state.get("supervisor"), state.get("child"), *state.get("processes", [])]
    ):
        raise ValueError("Candidate is not completed and released")
    training = path / "training"
    identity = json.loads((training / "identity.json").read_text())
    if (identity["group"] != "E3" or identity["updates"] != 8 or identity["world_size"] != 4
            or identity["purpose"] != "engineering" or identity["workload_trace"]):
        raise ValueError("Not a registered timing candidate")
    ranks, occurrences = [], []
    for rank in range(4):
        timing = json.loads((training / f"timing_rank{rank}.json").read_text())
        summary = json.loads((training / f"summary_rank{rank}.json").read_text())
        rows = [json.loads(line) for line in (training / f"steps_rank{rank}.jsonl").read_text().splitlines()]
        if timing["identity"] != identity or summary["identity"] != identity or summary["stopped"]:
            raise ValueError("Candidate rank identity differs")
        if summary["position"] != {"optimizer_update": 8, "next_occurrence": 8 * identity["global_batch"]}:
            raise ValueError("Candidate did not reach its budget")
        recomputed = summarize_steps(rows, micro_batch=identity["micro_batch"], accumulation=identity["accumulation"],
                                     world_size=4, warmup_updates=2)
        if timing["steady_summary"] != recomputed or len(rows) != 8 * identity["accumulation"]:
            raise ValueError("Timing summary does not match raw rows")
        ranks.append(recomputed)
        occurrences.extend(sample["occurrence"] for row in rows for sample in row["samples"])
    if sorted(occurrences) != list(range(8 * identity["global_batch"])):
        raise ValueError("Candidate exposure has gaps or duplicates")
    performance = json.loads((path / "performance_summary.json").read_text())
    capacity = {row["uuid"]: row["memory_total_bytes"] for row in performance["gpu_release_snapshot"]}
    headroom = {uuid: capacity[uuid] - record["memory_used_bytes"]["max"] for uuid, record in performance["by_gpu"].items()}
    if set(headroom) != set(state["gpu_uuids"]):
        raise ValueError("Incomplete device peak memory evidence")
    return {"run": str(path), "identity": identity, "micro_batch": identity["micro_batch"],
            "workers": identity["workers"], "accumulation": identity["accumulation"],
            "samples_per_second": min(row["global_samples_per_second_excluding_save"] for row in ranks),
            "peak_headroom_bytes_by_gpu": headroom,
            "minimum_headroom_bytes": min(headroom.values()), "ranks": ranks,
            "wall_seconds": performance["wall_seconds"], "allocated_gpu_hours": performance["allocated_gpu_hours"],
            "performance_sha256": sha256(path / "performance_summary.json"),
            "timing_sha256_by_rank": [sha256(training / f"timing_rank{rank}.json") for rank in range(4)]}


def select(records):
    eligible = [row for row in records if row["minimum_headroom_bytes"] >= 10 * 2**30]
    if not eligible:
        raise ValueError("No candidate retains ten GiB peak headroom")
    best = max(row["samples_per_second"] for row in eligible)
    close = [row for row in eligible if row["samples_per_second"] >= best / 1.05]
    return min(close, key=lambda row: (row["micro_batch"], row["workers"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--capacity-report", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.plan.exists():
        raise FileExistsError("Preserve existing selection/plan")
    registration = json.loads(args.candidates.read_text())
    records = [read_candidate(path) for path in args.run]
    actual = {(row["micro_batch"], row["accumulation"], row["workers"]) for row in records}
    expected = {(row["micro_batch"], row["accumulation"], row["workers"]) for row in registration["candidates"]}
    if actual != expected or len(actual) != len(records):
        raise ValueError("This selection requires exactly the three registered successful candidates; failed/skipped candidates need an explicit amended audit")
    matrix_hash = sha256(args.matrix / "initialization.json")
    for row in records:
        if row["identity"]["matrix_sha256"] != matrix_hash:
            raise ValueError("Tuning initialization mismatch")
    selected = select(records)
    report = {"goal_id": GOAL_ID, "registration_sha256": sha256(args.candidates), "candidates": records,
              "selected_run": selected["run"], "limits": "Six post-warmup updates, noisy short-window diagnostic throughput; initialization/save excluded. Not an efficacy search or formal training budget."}
    atomic_json(args.output, report)
    plan = {"goal_id": GOAL_ID, "purpose": "profiling", "protocol_id": "r2-profile-v1",
            "groups": ["E0", "E3"], "updates": 200, "world_size": 4, "seed": 20260914,
            "micro_batch": selected["micro_batch"], "accumulation": selected["accumulation"],
            "workers": selected["workers"], "save_every": 100, "warmup_updates": 10,
            "global_batch": 4 * selected["micro_batch"] * selected["accumulation"],
            "matrix": str(args.matrix.resolve()), "matrix_sha256": matrix_hash,
            "capacity_report": str(args.capacity_report.resolve()), "capacity_report_sha256": sha256(args.capacity_report),
            "selection_report": str(args.output.resolve()), "selection_report_sha256": sha256(args.output)}
    atomic_json(args.plan, plan)
    print(json.dumps({"selected": selected["run"], "plan": str(args.plan)}))


if __name__ == "__main__":
    main()
