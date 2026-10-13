import pytest
from types import SimpleNamespace

from scripts.statewam.r2.common import GOAL_ID
from scripts.statewam.r2 import fault_probe
from scripts.statewam.r2.fault_probe import check_target, open_writers


def spec():
    return {"goal_id": GOAL_ID, "run_id": "p2-fault-test", "phase": "engineering",
            "command": ["python", "scripts/statewam/r2/train.py", "--updates", "4"]}


def test_fault_target_must_be_dedicated_bounded_engineering_run():
    check_target(spec(), "p2-fault-test")
    for key, value in (("phase", "training"), ("goal_id", "other"), ("run_id", "another-fault"),
                       ("command", ["python", "scripts/statewam/r2/train.py", "--updates", "200"])):
        row = spec()
        row[key] = value
        with pytest.raises(ValueError):
            check_target(row, "p2-fault-test")


def test_dead_target_has_no_eligible_writer(tmp_path):
    assert open_writers({"child": None, "supervisor": None}, tmp_path) == []


def test_writer_must_be_owned_and_hold_writable_checkpoint_file(tmp_path, monkeypatch):
    partial = tmp_path / "update_00000004.partial"
    partial.mkdir()
    shard = partial / "model.pt"
    shard.write_bytes(b"checkpoint")
    opened = SimpleNamespace(path=str(shard), mode="w", fd=7)
    writer = SimpleNamespace(pid=3, cmdline=lambda: ["python", "scripts/statewam/r2/train.py"],
                             open_files=lambda: [opened])
    leader = SimpleNamespace(pid=2, ppid=lambda: 1, children=lambda recursive: [writer])
    supervisor = SimpleNamespace(pid=1)
    monkeypatch.setattr(fault_probe, "alive", lambda record: True)
    monkeypatch.setattr(fault_probe, "identity", lambda pid: {"pid": pid, "created_at": 123})
    monkeypatch.setattr(fault_probe.psutil, "Process", lambda pid: {1: supervisor, 2: leader}[pid])
    state = {"supervisor": {"pid": 1}, "child": {"pid": 2}}
    rows = open_writers(state, partial)
    assert len(rows) == 1 and rows[0]["process"]["pid"] == 3 and rows[0]["bytes"] == 10
    opened.mode = "r"
    assert open_writers(state, partial) == []
    opened.mode = "w"
    opened.path = str(tmp_path / "unrelated.pt")
    assert open_writers(state, partial) == []
    leader.ppid = lambda: 99
    with pytest.raises(ValueError, match="not owned"):
        open_writers(state, partial)
