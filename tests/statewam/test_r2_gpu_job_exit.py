import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.statewam.r2 import gpu_job as module
from scripts.statewam.r2.common import atomic_json
from scripts.statewam.r2.cpu_job import alive, identity


@pytest.mark.parametrize("live_descendant,exit_code", [(False, 0), (True, 0), (False, 1)])
def test_exit_rechecks_live_descendants_after_poll(tmp_path, monkeypatch, live_descendant, exit_code):
    atomic_json(tmp_path / "job.json", {"protocol_id": "test", "phase": "engineering", "session": "test",
        "gpu_uuids": ["uuid"], "gpu_indices": [0], "environment": {}, "command": ["unused"],
        "cwd": str(tmp_path), "stop_grace_seconds": 1, "sample_interval_seconds": 1})
    leader = {"pid": 123, "created_at": 1}
    worker = {"pid": 124, "created_at": 2}
    observations = iter([[leader], [worker] if live_descendant else [], [], []])
    monkeypatch.setattr(module, "descendants", lambda: next(observations))
    monkeypatch.setattr(module, "identity", lambda pid: leader)
    monkeypatch.setattr(module, "process_gpu_ids", lambda records: set())
    monkeypatch.setattr(module, "tree_sample", lambda pid: {"time": 1, "processes": [], "rss_bytes": 0})
    monkeypatch.setattr(module, "update_lease", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "lease_process_spill", lambda *args: [])
    monkeypatch.setattr(module, "performance_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr(module.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(prctl=lambda *args: 0))

    class Monitor:
        def snapshot(self):
            return [{"uuid": "uuid", "processes": []}]

        def close(self):
            pass

    class Child:
        pid = 123
        returncode = exit_code

        def poll(self):
            return self.returncode

    monkeypatch.setattr(module, "DeviceMonitor", Monitor)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: Child())
    assert module.run(tmp_path) == int(live_descendant or exit_code != 0)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["state"] == ("failed" if live_descendant or exit_code != 0 else "completed")
    assert state["leader_exit_observation"]["live_descendants"] == ([worker] if live_descendant else [])
    assert (tmp_path / "STOP").exists() == live_descendant
    if live_descendant:
        assert state["failure"] == "Workload leader exited with live descendants"


@pytest.mark.parametrize("submitted", [False, True])
def test_interrupted_launch_preserves_registration_and_rejects_duplicate(tmp_path, monkeypatch, submitted):
    monkeypatch.setattr(module, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(module, "LEASES", tmp_path / "leases.json")
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "test-head\n")
    device = {"index": 0, "uuid": "test-device", "processes": [], "errors": [],
              "memory_used_bytes": 0, "utilization_percent": 0}

    class Monitor:
        def snapshot(self):
            return [device]

        def close(self):
            pass

    monkeypatch.setattr(module, "DeviceMonitor", Monitor)
    monkeypatch.setattr(sys, "argv", ["gpu_job.py", "launch", "--run-id", "interrupted-launch",
                                     "--gpus", "0", "--", sys.executable, "-c", "pass"])
    directory = tmp_path / "runs/interrupted-launch"
    children, launch_calls = [], []

    def interrupted_submit(command, **kwargs):
        spec = json.loads((directory / "job.json").read_text())
        assert spec["run_id"] == "interrupted-launch"
        assert spec["gpu_uuids"] == ["test-device"]
        assert command[:3] == ["tmux", "new-session", "-d"]
        launch_calls.append(command)
        if submitted:
            # A real detached CPU child models a successful submission with lost ACK.
            children.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"],
                                             start_new_session=True))
        raise KeyboardInterrupt("Submission response was interrupted")

    monkeypatch.setattr(module.subprocess, "run", interrupted_submit)
    try:
        with pytest.raises(KeyboardInterrupt, match="response was interrupted"):
            module.main()
        registered = (directory / "job.json").read_bytes()
        assert not (directory / "state.json").exists()
        if submitted:
            assert alive(identity(children[0].pid))
        with pytest.raises(FileExistsError):
            module.main()
        assert len(launch_calls) == 1
        assert (directory / "job.json").read_bytes() == registered
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
