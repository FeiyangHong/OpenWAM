import json
from pathlib import Path

import pytest

from benchmarks.robotwin import r2_scene_queue as queue
from scripts.statewam.r2.common import GOAL_ID, TASKS, atomic_json, sha256


def test_four_lanes_cover_exact_fixed_scene_matrix():
    entries = [entry for lane in range(4) for entry in queue.lane_entries(lane)]
    assert len(entries) == 40
    assert len({tuple(sorted(entry.items())) for entry in entries}) == 40
    assert {entry["task"] for entry in entries} == set(TASKS)
    assert sum(queue.shards.COUNTS[entry["split"]] for entry in entries) == 500
    assert all(len(queue.lane_entries(lane)) == 10 for lane in range(4))
    for value in (-1, 4, True):
        with pytest.raises(ValueError):
            queue.lane_entries(value)


def mock_shards(monkeypatch, execute):
    entries = queue.lane_entries(0)[:2]
    monkeypatch.setattr(queue, "lane_entries", lambda lane: entries)
    monkeypatch.setattr(queue.shards, "run_shard", execute)
    monkeypatch.setattr(queue, "verify_shard", lambda path, entry: {"entry": entry, "count": 5})
    return entries


def test_stop_resume_skips_complete_shards_and_preserves_remaining_work(tmp_path, monkeypatch):
    stop = tmp_path / "STOP"
    calls = []

    def execute(path, **kwargs):
        calls.append((path, kwargs["resume"]))
        path.mkdir(parents=True, exist_ok=True)
        if len(calls) == 2:
            stop.touch()
            return "stopped"
        atomic_json(path / "COMPLETE.json", {})
        return "completed"

    mock_shards(monkeypatch, execute)
    output = tmp_path / "lane"
    assert queue.run_lane(output, lane=0, stop_file=stop) == "stopped"
    assert not (output / "COMPLETE.json").exists()
    assert queue.run_lane(output, lane=0, stop_file=tmp_path / "new-stop", resume=True) == "completed"
    assert len(calls) == 3 and calls[-1][1] is True
    assert calls[1][0] == calls[2][0] and calls[0][0] != calls[2][0]
    assert len(json.loads((output / "manifest.json").read_text())["shards"]) == 2
    with pytest.raises(ValueError, match="already complete"):
        queue.run_lane(output, lane=0, stop_file=stop, resume=True)


def test_failure_stops_lane_without_automatic_retry(tmp_path, monkeypatch):
    calls = []

    def execute(path, **kwargs):
        calls.append(path)
        raise RuntimeError("reset mismatch")

    mock_shards(monkeypatch, execute)
    output = tmp_path / "lane"
    with pytest.raises(RuntimeError, match="reset mismatch"):
        queue.run_lane(output, lane=0, stop_file=tmp_path / "STOP")
    assert len(calls) == 1
    assert json.loads((output / "progress.json").read_text())["state"] == "failed"
    assert not (output / "COMPLETE.json").exists()


def test_resume_refuses_queue_identity_change(tmp_path):
    stop = tmp_path / "STOP"
    stop.touch()
    output = tmp_path / "lane"
    assert queue.run_lane(output, lane=0, stop_file=stop) == "stopped"
    with pytest.raises(ValueError, match="identity changed"):
        queue.run_lane(output, lane=1, stop_file=stop, resume=True)


def test_completed_shard_hash_is_verified_before_reuse(tmp_path):
    atomic_json(tmp_path / "manifest.json", {})
    atomic_json(tmp_path / "COMPLETE.json", {"manifest_sha256": "wrong"})
    with pytest.raises(ValueError, match="manifest hash"):
        queue.verify_shard(tmp_path, queue.lane_entries(0)[0])


@pytest.fixture
def complete_shard(tmp_path, monkeypatch):
    entry = queue.lane_entries(0)[0]
    monkeypatch.setitem(queue.shards.COUNTS, "development", 1)
    first = queue.shards.seed_start(entry["task"], entry["split"])
    attempt = tmp_path / "candidates" / str(first) / "attempt_001"
    probe = attempt / "probe"
    probe.mkdir(parents=True)
    atomic_json(attempt / "attempt.json", {"exit_code": 0})
    atomic_json(probe / "scene_probe.json", {"completed": True})
    row = {"seed": first, "probe": str(probe.resolve()), "expert_eligible": True}
    monkeypatch.setattr(queue.shards, "inspect_probe", lambda *args, **kwargs: dict(row))
    identity = {"goal_id": GOAL_ID, **entry, "first_seed": first, "target": 1,
                "max_candidates": 1000, "max_infrastructure_attempts_per_seed": 2,
                "sources": {"benchmarks/robotwin/r2_scene_manifest.py": sha256(queue.shards.__file__)}}
    accepted = {**row, "scene_id": f"{entry['split']}/{entry['task']}/{entry['condition']}/{first}"}
    manifest = {"identity": identity, "count": 1, "accepted": [accepted], "rejected": []}
    atomic_json(tmp_path / "identity.json", identity)

    def publish():
        atomic_json(tmp_path / "manifest.json", manifest)
        atomic_json(tmp_path / "COMPLETE.json", {"manifest_sha256": sha256(tmp_path / "manifest.json")})

    publish()
    return tmp_path, entry, manifest, publish


def test_verify_complete_shard_and_reject_changed_scene_evidence(complete_shard):
    directory, entry, manifest, publish = complete_shard
    assert queue.verify_shard(directory, entry)["count"] == 1
    manifest["accepted"][0]["scene_id"] = "another-scene"
    publish()
    with pytest.raises(ValueError, match="independently verified"):
        queue.verify_shard(directory, entry)


@pytest.mark.parametrize("mutation", ["gap", "duplicate", "count", "outside", "attempts", "source"])
def test_complete_marker_does_not_override_ledger_validation(complete_shard, mutation):
    directory, entry, manifest, publish = complete_shard
    row = manifest["accepted"][0]
    if mutation == "gap":
        row["seed"] += 1
    elif mutation == "duplicate":
        manifest["rejected"].append(dict(row))
    elif mutation == "count":
        manifest["count"] = 0
    elif mutation == "outside":
        row["probe"] = str(directory / "foreign-probe")
    elif mutation == "attempts":
        candidate = Path(row["probe"]).parent.parent
        (candidate / "attempt_002").mkdir()
        (candidate / "attempt_003").mkdir()
    elif mutation == "source":
        manifest["identity"]["sources"]["benchmarks/robotwin/r2_scene_manifest.py"] = "changed"
        atomic_json(directory / "identity.json", manifest["identity"])
    publish()
    with pytest.raises(ValueError):
        queue.verify_shard(directory, entry)


def test_live_queue_lock_refuses_duplicate_takeover(tmp_path):
    import fcntl

    stop = tmp_path / "STOP"
    stop.touch()
    output = tmp_path / "lane"
    assert queue.run_lane(output, lane=0, stop_file=stop) == "stopped"
    with (output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            queue.run_lane(output, lane=0, stop_file=stop, resume=True)


def test_queue_rechecks_provenance_before_completion(tmp_path, monkeypatch):
    def execute(path, **kwargs):
        path.mkdir(parents=True)
        atomic_json(path / "COMPLETE.json", {})
        return "completed"

    mock_shards(monkeypatch, execute)
    provenance = {"asset_report": str(tmp_path / "assets.json"), "environment_report": str(tmp_path / "env.json")}
    monkeypatch.setattr(queue, "verify_provenance", lambda *args: {**provenance, "changed": True})
    output = tmp_path / "lane"
    with pytest.raises(ValueError, match="provenance changed"):
        queue.run_lane(output, lane=0, stop_file=tmp_path / "STOP", provenance=provenance)
    assert not (output / "COMPLETE.json").exists()


def test_unpassed_provenance_cannot_start_queue(tmp_path):
    asset_report, environment_report = tmp_path / "assets.json", tmp_path / "environment.json"
    atomic_json(asset_report, {"goal_id": GOAL_ID, "passed": False})
    atomic_json(environment_report, {"goal_id": GOAL_ID, "passed": True})
    with pytest.raises(ValueError, match="passed R2"):
        queue.verify_provenance(asset_report, environment_report)
