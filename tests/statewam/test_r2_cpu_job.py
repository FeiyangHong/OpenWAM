import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.statewam.r2.cpu_job import alive, identity

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/statewam/r2/cpu_job.py"


def invoke(root, operation, run_id, *command):
    args = [sys.executable, str(SCRIPT), operation, "--root", str(root), "--run-id", run_id]
    return subprocess.run(args + list(command), capture_output=True, text=True)


def wait_for_state(root, run_id, terminal=False):
    deadline = time.monotonic() + 20
    path = root / run_id / "state.json"
    while time.monotonic() < deadline:
        if path.exists():
            state = json.loads(path.read_text())
            if not terminal or state["state"] in ("completed", "failed", "stopped"):
                return state
        time.sleep(0.05)
    raise AssertionError("CPU job did not reach expected state")


def test_live_identity_rejects_stale_process():
    record = identity(__import__("os").getpid())
    assert alive(record)
    assert not alive(dict(record, created_at=record["created_at"] - 1))


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux unavailable")
def test_detached_cpu_job_logs_and_rejects_duplicate(tmp_path):
    code = 'import time; print("start",flush=True); time.sleep(3); print("end",flush=True)'
    launched = invoke(tmp_path, "launch", "detached", "--", sys.executable, "-c", code)
    assert launched.returncode == 0, launched.stderr
    started = wait_for_state(tmp_path, "detached")
    assert alive(started["supervisor"])
    duplicate = invoke(tmp_path, "launch", "detached", "--", sys.executable, "-c", code)
    assert duplicate.returncode != 0
    state = wait_for_state(tmp_path, "detached", terminal=True)
    assert state["state"] == "completed" and state["exit_code"] == 0
    assert not alive(state["child"])
    assert (tmp_path / "detached/console.log").read_text().splitlines() == ["start", "end"]
    samples = [json.loads(line) for line in (tmp_path / "detached/resource_samples.jsonl").read_text().splitlines()]
    assert len(samples) >= 2 and all(sample["gpu_uuids"] == [] for sample in samples)


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux unavailable")
def test_cpu_stop_and_failed_exit_are_not_completed(tmp_path):
    launched = invoke(tmp_path, "launch", "stop", "--", sys.executable, "-c", "import time; time.sleep(18)")
    assert launched.returncode == 0
    wait_for_state(tmp_path, "stop")
    assert invoke(tmp_path, "stop", "stop").returncode == 0
    state = wait_for_state(tmp_path, "stop", terminal=True)
    assert state["state"] == "stopped"
    assert not alive(state["child"])
    launched = invoke(tmp_path, "launch", "failure", "--", sys.executable, "-c", "raise SystemExit(7)")
    assert launched.returncode == 0
    state = wait_for_state(tmp_path, "failure", terminal=True)
    assert state["state"] == "failed" and state["exit_code"] == 7
