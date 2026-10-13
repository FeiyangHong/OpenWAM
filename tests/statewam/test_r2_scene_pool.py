import json

import pytest

from benchmarks.robotwin import r2_scene_pool as pool
from scripts.statewam.r2.common import atomic_json


def reports(tmp_path):
    assets, environment = tmp_path / "assets.json", tmp_path / "env.json"
    atomic_json(assets, {})
    atomic_json(environment, {})
    return {"asset_report": assets, "environment_report": environment, "stop_file": tmp_path / "STOP"}


def test_pool_stop_before_launch_preserves_budget_and_snapshot(tmp_path, monkeypatch):
    options = reports(tmp_path)
    options["stop_file"].touch()
    monkeypatch.setattr(pool.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Must not launch after STOP"))
    output = tmp_path / "pool"
    assert pool.run_pool(output, **options) == "stopped"
    assert (output / "source_snapshot/benchmarks/robotwin/r2_scene_probe.py").exists()
    assert not (output / "COMPLETE.json").exists()
    assert pool.run_pool(output, **options, resume=True) == "stopped"


def test_pool_publishes_only_after_four_verified_lanes(tmp_path, monkeypatch):
    options = reports(tmp_path)

    class Child:
        pid = 1

        def __init__(self, command, **kwargs):
            from pathlib import Path
            directory = Path(command[command.index("--output") + 1])
            directory.mkdir(parents=True)
            atomic_json(directory / "COMPLETE.json", {})

        def poll(self):
            return 0

    monkeypatch.setattr(pool.subprocess, "Popen", Child)
    monkeypatch.setattr(pool, "process_identity", lambda pid: {"pid": pid, "created_at": 1})
    monkeypatch.setattr(pool, "verify_lane", lambda directory, lane: {"lane": lane})
    output = tmp_path / "pool"
    assert pool.run_pool(output, **options) == "completed"
    assert len(json.loads((output / "manifest.json").read_text())["lanes"]) == 4
    with pytest.raises(ValueError, match="already completed"):
        pool.run_pool(output, **options, resume=True)


def test_lane_failure_stops_pool_without_retrying(tmp_path, monkeypatch):
    options = reports(tmp_path)
    calls = []

    class Child:
        pid = 1

        def __init__(self, command, **kwargs):
            calls.append(command)

        def poll(self):
            return 1

    monkeypatch.setattr(pool.subprocess, "Popen", Child)
    monkeypatch.setattr(pool, "process_identity", lambda pid: {"pid": pid, "created_at": 1})
    output = tmp_path / "pool"
    assert pool.run_pool(output, **options) == "failed"
    assert len(calls) == 4 and options["stop_file"].exists()
    assert not (output / "COMPLETE.json").exists()
