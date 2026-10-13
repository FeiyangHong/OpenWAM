import json

import pytest

from scripts.statewam.r2 import resource_accounting as module
from scripts.statewam.r2.common import GOAL_ID, atomic_json


def test_union_deduplicates_shared_devices_and_counts_real_overlaps():
    rows = [{"start": 0, "end": 3600, "uuids": ["a", "b", "b"]},
            {"start": 1800, "end": 5400, "uuids": ["b", "c"]}]
    report = module.interval_union(rows)
    assert report["gpu_hours"] == 3.5
    assert report["peak_devices"] == 3
    assert report["busy_wall_seconds"] == 5400
    rows.append({"start": 2000, "end": 2200, "uuids": ["d", "e"]})
    assert module.interval_union(rows)["over_four_intervals"] == [
        {"start": 2000, "end": 2200, "uuids": ["a", "b", "c", "d", "e"]}]
    with pytest.raises(ValueError):
        module.interval_union([{"start": 1, "end": 0, "uuids": ["a"]}])


def test_spill_samples_and_unknown_attribution_are_preserved(tmp_path):
    path = tmp_path / "samples.jsonl"
    rows = [{"time": stamp, "rss_bytes": 10, "sampling_seconds": 0.1, "used_uuids": ["a"],
             "process_attribution_complete": False, "lease_associated_outside_processes": [{"uuid": "b"}]}
            for stamp in (1, 5, 10)]
    path.write_bytes(("".join(json.dumps(row) + "\n" for row in rows) + '{"time":').encode())
    summary, known, spills = module.scan_samples(path, start=0, end=8, leased=["a"])
    assert summary["incomplete_tail_excluded"] and summary["unattributed_or_unverified_samples"] == 3
    assert summary["spill_sample_count"] == 3
    assert module.interval_union(known)["gpu_hours"] == 14 / 3600
    assert module.interval_union(spills)["gpu_hours"] == 7 / 3600


def test_live_jobs_use_verified_cutoff_and_dead_unknown_tails_stay_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "alive", lambda row: bool(row and row.get("live")))
    for name, owner, state in (("live", {"live": True}, "running"), ("unknown", None, "running"),
                               ("done", None, "completed")):
        run = tmp_path / name
        atomic_json(run / "job.json", {"goal_id": GOAL_ID, "gpu_uuids": ["a"], "phase": "engineering"})
        atomic_json(run / "state.json", {"started_at": 0, "heartbeat": 20, "ended_at": 25,
                    "state": state, "supervisor": owner, "gpu_uuids": ["a"]})
    report = module.collect(tmp_path, cutoff=30)
    assert report["allocation_union"]["gpu_hours"] == 30 / 3600
    ends = {row["run_id"]: row["interval"]["end"] for row in report["runs"]}
    assert ends == {"live": 30, "unknown": 20, "done": 25}
    assert report["actual_total_gpu_hours"] is None
    assert any(row["reason"] == "terminal time unverified" for row in report["incomplete_observations"])
