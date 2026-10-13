import json

import pytest

from benchmarks.robotwin import r2_episode_recovery as module
from scripts.statewam.r2.common import atomic_json


@pytest.fixture
def terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "alive", lambda row: bool(row))
    state = {"state": "stopped", "supervisor": None, "child": None, "processes": [],
             "gpu_uuids": ["gpu0", "gpu1"], "gpu_release_snapshot": [
                 {"uuid": "gpu0", "processes": []}, {"uuid": "gpu1", "processes": []},
                 {"uuid": "unrelated", "processes": [{"pid": 999}]}]}
    atomic_json(tmp_path / "state.json", state)
    return tmp_path


def test_recovery_requires_own_devices_released_not_other_users(terminal):
    assert module.released(terminal)["state"] == "stopped"


@pytest.mark.parametrize("case", ["supervisor", "child", "processes", "running", "graphics", "missing_device"])
def test_recovery_rejects_live_or_unverified_resources(terminal, case):
    path = terminal / "state.json"
    state = json.loads(path.read_text())
    if case in ("supervisor", "child"):
        state[case] = {"pid": 123, "created_at": 1}
    elif case == "processes":
        state[case] = [{"pid": 123, "created_at": 1}]
    elif case == "running":
        state["state"] = "running"
    elif case == "graphics":
        state["gpu_release_snapshot"][0]["processes"] = [{"pid": 123, "kinds": ["graphics"]}]
    else:
        state["gpu_release_snapshot"].pop(0)
    atomic_json(path, state)
    with pytest.raises(ValueError):
        module.released(terminal)
