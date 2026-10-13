"""Goal-wide physical UUID interval unions, with explicit attribution gaps."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.common import ARTIFACTS, GOAL_ID, atomic_json, sha256
from scripts.statewam.r2.cpu_job import alive


def interval_union(intervals):
    """Integrate UUID sets, never counting co-located processes twice."""
    events = defaultdict(list)
    for row in intervals:
        start, end = row["start"], row["end"]
        if not all(math.isfinite(value) for value in (start, end)) or end < start:
            raise ValueError("Invalid resource interval")
        if start == end:
            continue
        for uuid in set(row["uuids"]):
            events[start].append((uuid, 1))
            events[end].append((uuid, -1))
    active = defaultdict(int)
    seconds_by_uuid = defaultdict(float)
    previous, peak, busy_seconds = None, 0, 0.0
    violations = []
    for stamp in sorted(events):
        devices = sorted(uuid for uuid, count in active.items() if count)
        if previous is not None and devices:
            duration = stamp - previous
            busy_seconds += duration
            peak = max(peak, len(devices))
            for uuid in devices:
                seconds_by_uuid[uuid] += duration
            if len(devices) > 4:
                violations.append({"start": previous, "end": stamp, "uuids": devices})
        for uuid, delta in events[stamp]:
            active[uuid] += delta
        previous = stamp
    return {"gpu_hours": sum(seconds_by_uuid.values()) / 3600,
            "by_uuid_gpu_hours": {uuid: seconds / 3600 for uuid, seconds in sorted(seconds_by_uuid.items())},
            "busy_wall_seconds": busy_seconds, "peak_devices": peak, "over_four_intervals": violations}


def scan_samples(path, *, start, end, leased):
    digest, samples, previous = hashlib.sha256(), 0, None
    observed, spill_intervals = [], []
    gaps, missing_attribution, spill_samples, peak_rss, sampling_seconds = [], 0, 0, 0, 0.0
    incomplete_tail = False
    first_stamp = last_stamp = None
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                incomplete_tail = True
                break
            digest.update(line)
            row = json.loads(line)
            stamp = row["time"]
            if not math.isfinite(stamp) or previous is not None and stamp <= previous["time"]:
                raise ValueError("Resource samples have invalid or unordered timestamps")
            samples += 1
            first_stamp = stamp if first_stamp is None else first_stamp
            last_stamp = stamp
            peak_rss = max(peak_rss, row["rss_bytes"])
            sampling_seconds += row["sampling_seconds"]
            missing_attribution += row.get("process_attribution_complete") is not True
            spills = {item["uuid"] for item in row.get("lease_associated_outside_processes", [])}
            spills |= set(row.get("used_uuids", [])) - set(leased)
            spill_samples += bool(spills)
            if previous is not None:
                gaps.append(stamp - previous["time"])
                begin, finish = max(start, previous["time"]), min(end, stamp)
                if finish > begin:
                    observed.append({"start": begin, "end": finish, "uuids": previous["known"]})
                    spill_intervals.append({"start": begin, "end": finish, "uuids": previous["spills"]})
            previous = {"time": stamp, "known": set(row.get("used_uuids", [])) | spills, "spills": spills}
    return {"sample_count": samples, "complete_lines_sha256": digest.hexdigest(),
            "incomplete_tail_excluded": incomplete_tail, "first_sample": first_stamp, "last_sample": last_stamp,
            "max_sample_gap_seconds": max(gaps, default=None),
            "unattributed_or_unverified_samples": missing_attribution, "spill_sample_count": spill_samples,
            "peak_process_tree_rss_bytes": peak_rss, "sampling_seconds": sampling_seconds}, observed, spill_intervals


def collect(root, *, cutoff=None):
    root = Path(root).resolve()
    cutoff = time.time() if cutoff is None else cutoff
    leases, observed, spills, runs, incomplete = [], [], [], [], []
    categories = defaultdict(list)
    cpu_runs = []
    for job_path in sorted(root.glob("*/job.json")):
        run = job_path.parent
        job = json.loads(job_path.read_text())
        if job["goal_id"] != GOAL_ID:
            continue
        state_path = run / "state.json"
        if not state_path.exists():
            incomplete.append({"run_id": run.name, "reason": "missing state; launch alone does not prove execution"})
            continue
        state = json.loads(state_path.read_text())
        start = state["started_at"]
        if start > cutoff:
            continue
        live = any(alive(row) for row in [state.get("supervisor"), state.get("child"), *state.get("processes", [])])
        terminal = state["state"] in ("completed", "stopped", "failed") and "ended_at" in state and not live
        if terminal:
            end = min(cutoff, state["ended_at"])
        elif live:
            end = cutoff
        else:
            end = min(cutoff, state.get("heartbeat", start))
        if not terminal:
            incomplete.append({"run_id": run.name, "reason": "live at cutoff" if live else "terminal time unverified"})
        uuids = job["gpu_uuids"]
        if not uuids:
            cpu_runs.append({"run_id": run.name, "state": state["state"], "live_now": live,
                             "wall_seconds_through_cutoff": end - start})
            continue
        if state.get("gpu_uuids") != uuids:
            raise ValueError(f"Job/state GPU identity differs: {run.name}")
        interval = {"start": start, "end": end, "uuids": uuids}
        leases.append(interval)
        categories[job["phase"]].append(interval)
        record = {"run_id": run.name, "phase": job["phase"], "state": state["state"], "live_now": live,
                  "terminal_verified": terminal, "job_sha256": sha256(job_path),
                  "state_snapshot": state, "interval": interval,
                  "allocated_gpu_hours": len(set(uuids)) * (end - start) / 3600}
        sample_path = run / "resource_samples.jsonl"
        if sample_path.exists():
            summary, known, extra = scan_samples(sample_path, start=start, end=end, leased=uuids)
            record["samples"] = summary
            observed.extend(known)
            spills.extend(extra)
        else:
            record["samples"] = None
            incomplete.append({"run_id": run.name, "reason": "resource samples missing"})
        performance_path = run / "performance_summary.json"
        if terminal and performance_path.exists():
            record["performance_sha256"] = sha256(performance_path)
            record["performance"] = json.loads(performance_path.read_text())
        elif terminal:
            incomplete.append({"run_id": run.name, "reason": "terminal performance summary missing"})
        runs.append(record)
    return {"goal_id": GOAL_ID, "as_of_unix_seconds": cutoff, "runs_root": str(root),
            "allocation_union": interval_union(leases),
            "allocation_plus_sampled_known_spills_union": interval_union(leases + spills),
            "sampled_attributed_process_union": interval_union(observed),
            "allocation_by_registered_phase": {phase: interval_union(rows) for phase, rows in sorted(categories.items())},
            "actual_total_gpu_hours": None, "project_energy_kwh": None,
            "runs": runs, "cpu_runs": cpu_runs, "incomplete_observations": incomplete,
            "historical_spill_samples": sum((row["samples"] or {}).get("spill_sample_count", 0) for row in runs),
            "limits": ["Allocation counts one physical UUID per interval, including load/save/service waits; not model compute time",
                "Known spill intervals use left-endpoint sample estimates; unseen historical spills and PID attribution gaps remain unknown",
                "GPU limit compliance is not inferred from leases alone; historical violations are retained",
                "Open jobs counted through cutoff only after PID/creation-time liveness verification; unknown terminal tails are not filled with zero",
                "Per-phase unions need not sum to the global union when devices overlap across phases",
                "No precise total GPU compute or energy claim; CPU RSS may double-count shared pages; no storage inventory in this report"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ARTIFACTS / "runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior resource accounting snapshots")
    atomic_json(args.output, collect(args.runs_root))


if __name__ == "__main__":
    main()
