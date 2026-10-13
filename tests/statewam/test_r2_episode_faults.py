import json
import os
import subprocess
import sys

import pytest

from benchmarks.robotwin import r2_episode_engineering as module
from scripts.statewam.r2.common import atomic_json, stable_hash
from tests.statewam.test_r2_episode_engineering import plan_path


@pytest.mark.parametrize("fault", [
    {"mode": "forced_interrupt", "after_actions": 0, "latest_action": 32},
    {"mode": "cooperative_stop", "after_actions": 8, "latest_action": 400},
    {"mode": "none", "after_actions": 8, "latest_action": 32},
    {"mode": "kill_all", "after_actions": 8, "latest_action": 32},
])
def test_fault_plan_rejects_unbounded_or_changed_triggers(plan_path, fault):
    plan = json.loads(plan_path.read_text())
    atomic_json(plan_path, {**plan, "fault": fault})
    with pytest.raises(ValueError, match="predeclared engineering fault"):
        module.load_plan(plan_path)


def prepare(plan_path, mode, pid):
    plan, _, identity = module.load_plan(plan_path)
    plan["fault"] = {"mode": mode, "after_actions": 8, "latest_action": 32}
    attempt = module.Path(plan["ledger_root"]) / stable_hash(identity) / "attempt_001"
    atomic_json(attempt / "begin.json", {"owner": module.process_identity(pid), "episode_key": stable_hash(identity)})
    atomic_json(attempt / "progress.json", {"actions": 8, "physics_steps": 10, "attempt": 1})
    output = plan_path.parent / "physical"
    output.mkdir()
    return plan, identity, attempt, output


def test_cooperative_fault_writes_stop_and_never_kills(plan_path, monkeypatch):
    plan, identity, attempt, output = prepare(plan_path, "cooperative_stop", os.getpid())
    stop = output / "STOP"
    monkeypatch.setenv("R2_STOP_FILE", str(stop))

    class Child:
        pid = os.getpid()

        def kill(self):
            raise AssertionError("Cooperative STOP must not signal a process")

    assert module.maybe_inject_fault(plan, identity, output, Child())
    assert json.loads(stop.read_text())["reason"] == "registered_engineering_stop"
    assert json.loads((output / "fault_action.json").read_text())["executed"]
    assert not module.maybe_inject_fault(plan, identity, output, Child())


def test_forced_fault_kills_only_owned_child_and_is_not_reinjected(plan_path, monkeypatch):
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"], stdin=subprocess.PIPE)
    try:
        plan, identity, attempt, output = prepare(plan_path, "forced_interrupt", child.pid)
        monkeypatch.setenv("R2_STOP_FILE", str(output / "STOP"))
        assert module.maybe_inject_fault(plan, identity, output, child)
        assert child.wait(timeout=10) == -9
        assert not (output / "STOP").exists()
        assert not module.maybe_inject_fault(plan, identity, output, child)
        assert (attempt / "fault_injection.json").exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        child.stdin.close()


@pytest.mark.parametrize("case", ["identity", "late", "completed", "attempt"])
def test_fault_rejects_wrong_target_or_completed_episode(plan_path, monkeypatch, case):
    plan, identity, attempt, output = prepare(plan_path, "forced_interrupt", os.getpid())
    monkeypatch.setenv("R2_STOP_FILE", str(output / "STOP"))
    if case == "identity":
        begin = json.loads((attempt / "begin.json").read_text())
        begin["owner"]["created_at"] += 1
        atomic_json(attempt / "begin.json", begin)
    elif case == "late":
        atomic_json(attempt / "progress.json", {"actions": 33, "attempt": 1})
    elif case == "completed":
        atomic_json(attempt / "result.json", {"outcome": "failure"})
    else:
        atomic_json(attempt / "progress.json", {"actions": 8, "attempt": 2})

    class Child:
        pid = os.getpid()

        def kill(self):
            raise AssertionError("Invalid target must not receive a signal")

    with pytest.raises((ValueError, RuntimeError)):
        module.maybe_inject_fault(plan, identity, output, Child())
    assert not (attempt / "fault_injection.json").exists()


def test_fault_waits_for_persisted_action_threshold(plan_path, monkeypatch):
    plan, identity, attempt, output = prepare(plan_path, "cooperative_stop", os.getpid())
    atomic_json(attempt / "progress.json", {"actions": 7, "attempt": 1})
    assert not module.maybe_inject_fault(plan, identity, output, None)
    assert not (attempt / "fault_injection.json").exists()
