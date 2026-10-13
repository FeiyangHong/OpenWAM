import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from benchmarks.robotwin import r2_pilot_evaluation as module
from benchmarks.robotwin.r2_episode_ledger import EpisodeLedger
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256
from tests.statewam.test_r2_evaluation_plan import schedule_fixture


def test_execution_refuses_modified_protocol_or_schedule(tmp_path):
    schedule = schedule_fixture()
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("fixture\n")
    schedule["protocol_sha256"] = sha256(protocol)
    path = tmp_path / "schedule.json"
    atomic_json(path, schedule)
    execution = tmp_path / "execution.json"
    atomic_json(execution, {"goal_id": GOAL_ID, "protocol": str(protocol), "protocol_sha256": sha256(protocol),
                            "schedule": str(path), "schedule_sha256": sha256(path)})
    assert module.read_execution(execution)[1] == schedule
    protocol.write_text("changed\n")
    with pytest.raises(ValueError, match="identity changed"):
        module.read_execution(execution)


def test_block_selection_requires_unique_registered_identity():
    schedule = schedule_fixture()
    assert module.select_block(schedule, "wave_000_lane_0")["split"] == "repeatability"
    with pytest.raises(ValueError, match="Unknown"):
        module.select_block(schedule, "wave_999_lane_0")
    schedule["waves"].append(schedule["waves"][0])
    with pytest.raises(ValueError, match="duplicate"):
        module.select_block(schedule, "wave_000_lane_0")


def test_completed_failure_skip_unknown_partial_and_retry_limit(tmp_path):
    episode = schedule_fixture()["waves"][0]["blocks"][0]["episodes"][0]
    assert not module.completed_episode(tmp_path, episode)
    with EpisodeLedger(tmp_path, episode["identity"]) as ledger:
        attempt = ledger.begin()
    with pytest.raises(ValueError, match="Unclassified"):
        module.completed_episode(tmp_path, episode)
    with EpisodeLedger(tmp_path, episode["identity"]) as ledger:
        (attempt / "evidence.json").write_text("fixture\n")
        ledger.finish(attempt, outcome="invalid", reason="external_interrupt", stats={}, artifacts={"evidence": "evidence.json"})
    assert not module.completed_episode(tmp_path, episode)
    with EpisodeLedger(tmp_path, episode["identity"]) as ledger:
        second = ledger.begin()
        (second / "video.mp4").write_bytes(b"test video identity only")
        (second / "rollout.json").write_text("{}\n")
        ledger.finish(second, outcome="failure", reason="action_limit",
                      stats={"policy_actions": 400, "physics_steps": 10, "wall_seconds": 1.0},
                      artifacts={"video": "video.mp4", "rollout": "rollout.json"})
    assert module.completed_episode(tmp_path, episode)
    assert module.completed_episode(tmp_path, episode)
    assert len(list(attempt.parent.glob("attempt_*"))) == 2


def test_wait_handles_real_child_exit_and_nonzero(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    module.wait_child(child, deadline=time.monotonic() + 10, stop_file=tmp_path / "STOP")
    child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(RuntimeError, match="exited 3"):
        module.wait_child(child, deadline=time.monotonic() + 10, stop_file=tmp_path / "STOP")


def test_wait_rejects_stop_and_cleanup_reaps_owned_child(tmp_path, monkeypatch):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    state = {"state": "failed", "children": {"test": {"owner": module.process_identity(child.pid)}}}
    stop = tmp_path / "STOP"
    atomic_json(stop, {"reason": "test"})
    monkeypatch.setitem(module.EXECUTION, "shutdown_seconds", 0.05)
    try:
        with pytest.raises(RuntimeError, match="stopped"):
            module.wait_child(child, deadline=time.monotonic() + 10, stop_file=stop)
    finally:
        module.finish_children({"test": child}, state, stop_file=stop)
    assert child.poll() is not None
    assert not module.alive(state["children"]["test"]["owner"])
    assert state["state"] == "failed"


@pytest.mark.parametrize("failing_episode", [False, True])
def test_block_reuses_real_service_process_and_reaps_children(tmp_path, monkeypatch, failing_episode):
    schedule = schedule_fixture()
    block = schedule["waves"][0]["blocks"][0]
    fixture = tmp_path / "fixture.json"
    ledger_root = tmp_path / "ledger"
    execution = {"ledger_root": str(ledger_root), "simulator_python": sys.executable}
    atomic_json(fixture, {"block": block, "ledger": str(ledger_root)})
    monkeypatch.setattr(module, "read_execution", lambda path: (execution, schedule))
    monkeypatch.setattr(module, "verify_cuda_devices", lambda: ["simulator", "policy"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("R2_STOP_FILE", str(tmp_path / "STOP"))
    actual_popen = subprocess.Popen
    spawned = []
    worker = '''
import json, sys, time
from pathlib import Path
from scripts.statewam.r2.common import atomic_json
from benchmarks.robotwin.r2_episode_ledger import EpisodeLedger
fixture, role, target, index, fail = sys.argv[1:]
data = json.loads(Path(fixture).read_text())
target = Path(target)
if role == "service":
    atomic_json(target / "service/ready.json", {"ready": True})
    deadline = time.monotonic() + 15
    while not (target / "SERVICE_STOP").exists():
        if time.monotonic() > deadline:
            raise SystemExit(4)
        time.sleep(0.02)
else:
    if fail == "True":
        raise SystemExit(2)
    episode = data["block"]["episodes"][int(index)]
    with EpisodeLedger(data["ledger"], episode["identity"]) as ledger:
        attempt = ledger.begin()
        (attempt / "video.mp4").write_bytes(b"fixture, not physical acceptance")
        (attempt / "rollout.json").write_text("{}")
        ledger.finish(attempt, outcome="failure", reason="action_limit",
                      stats={"policy_actions":400,"physics_steps":1,"wall_seconds":0.1},
                      artifacts={"video":"video.mp4","rollout":"rollout.json"})
'''
    def spawn(command, **kwargs):
        role = command[2]
        output = command[command.index("--output") + 1]
        index = command[command.index("--index") + 1] if "--index" in command else "0"
        child = actual_popen([sys.executable, "-c", worker, str(fixture), role, output, index, str(failing_episode)], **kwargs)
        spawned.append((role, child))
        return child
    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    output = tmp_path / "block"
    if failing_episode:
        with pytest.raises(RuntimeError, match="exited 2"):
            module.run_block(tmp_path / "execution.json", block["block_id"], output)
    else:
        module.run_block(tmp_path / "execution.json", block["block_id"], output)
    state = json.loads((output / "state.json").read_text())
    assert state["state"] == ("failed" if failing_episode else "completed")
    assert len([role for role, _ in spawned if role == "service"]) == 1
    assert len([role for role, _ in spawned if role == "episode"]) == (1 if failing_episode else 2)
    assert all(child.poll() is not None for _, child in spawned)
    assert all(not module.alive(row["owner"]) for row in state["children"].values())
    if not failing_episode:
        module.run_block(tmp_path / "execution.json", block["block_id"], tmp_path / "resumed_block")
        resumed = json.loads((tmp_path / "resumed_block/state.json").read_text())
        assert len(resumed["skipped"]) == 2 and not resumed["children"]


@pytest.mark.parametrize("failing_lane", [False, True])
def test_full_coordinator_real_children_wave_barrier_and_device_assignment(tmp_path, monkeypatch, failing_lane):
    schedule = schedule_fixture()
    schedule["waves"] = schedule["waves"][:2]
    plan = {"environment": {"path": str(tmp_path / "environment.json")},
            "assets": {"path": str(tmp_path / "assets.json")}}
    atomic_json(tmp_path / "environment.json", {"python": sys.executable})
    atomic_json(tmp_path / "job.json", {"gpu_indices": [0, 1, 2]})
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("coordinator fixture")
    monkeypatch.setattr(module, "verify_cuda_devices", lambda: ["simulator", "policy-a", "policy-b"])
    monkeypatch.setattr(module, "snapshot_sources", lambda path: {})
    monkeypatch.setattr(module, "assert_sources_unchanged", lambda sources: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    monkeypatch.setenv("R2_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("R2_STOP_FILE", str(tmp_path / "STOP"))
    actual_popen = subprocess.Popen
    spawned = []
    worker = '''
import json, os, sys, time
from pathlib import Path
from scripts.statewam.r2.common import atomic_json
role, target, fail = sys.argv[1:]
target = Path(target)
if role == "provenance":
    atomic_json(target, {"provenance": "same fixture before and after"})
else:
    target.mkdir()
    started = time.time()
    time.sleep(0.15 if target.name.endswith("lane_0") else 0.3)
    atomic_json(target / "observation.json", {
        "started": started, "ended": time.time(), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_uuids": os.environ["R2_GPU_UUIDS"], "pid": os.getpid()})
    if fail == "True" and target.name == "wave_000_lane_0":
        raise SystemExit(2)
'''
    def spawn(command, **kwargs):
        role = command[2]
        target = command[command.index("--output") + 1]
        child = actual_popen([sys.executable, "-c", worker, role, target, str(failing_lane)], **kwargs)
        spawned.append((role, child))
        return child
    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    output = tmp_path / "evaluation"
    if failing_lane:
        with pytest.raises(RuntimeError, match="Wave"):
            module.execute_schedule(plan, schedule, protocol, tmp_path / "ledger", output)
    else:
        module.execute_schedule(plan, schedule, protocol, tmp_path / "ledger", output)
    state = json.loads((output / "state.json").read_text())
    assert state["state"] == ("failed" if failing_lane else "completed")
    assert all(child.poll() is not None for _, child in spawned)
    assert all(not module.alive(row["owner"]) for row in state["children"].values())
    first = [json.loads((output / f"wave_000_lane_{lane}/observation.json").read_text()) for lane in (0, 1)]
    assert [row["cuda_visible_devices"] for row in first] == ["0,1", "0,2"]
    assert [row["gpu_uuids"] for row in first] == ["simulator,policy-a", "simulator,policy-b"]
    if failing_lane:
        assert not (output / "wave_001_lane_0").exists()
        assert (tmp_path / "STOP").exists()
    else:
        second = [json.loads((output / f"wave_001_lane_{lane}/observation.json").read_text()) for lane in (0, 1)]
        assert min(row["started"] for row in second) >= max(row["ended"] for row in first)
        assert state["completed_waves"] == [0, 1]
        assert len({row["pid"] for row in first + second}) == 4
        assert [role for role, _ in spawned].count("provenance") == 2
