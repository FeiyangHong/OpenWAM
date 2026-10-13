import copy
import json
from pathlib import Path

import pytest

from benchmarks.robotwin import r2_pilot_smoke as module
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256
from scripts.statewam.r2.evaluation_plan import EXECUTION, scheduled_episodes
from tests.statewam.test_r2_evaluation_plan import scene_fixture


def plan_fixture(tmp_path, monkeypatch):
    matrix = tmp_path / "matrix"
    files = [tmp_path / "environment.json", tmp_path / "assets.json", matrix / "initialization.json", matrix / "COMPLETE.json"]
    for group in ("E0", "E3"):
        files.extend(matrix / group / name for name in ("checkpoint_step_0.safetensors", "config.yaml", "normalization_stats.npy"))
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    rows = [row for row in scene_fixture()["rows"]["development"]
            if row["task"] == "adjust_bottle" and row["seed"] in (200000, 200001)]
    for row in rows:
        row["probe"] = str(tmp_path / row["condition"] / str(row["seed"]))
    plan = {"goal_id": GOAL_ID, "protocol_id": module.PROTOCOL, "purpose": "engineering", "max_episodes": 8,
            "automatic_retries": 0, "groups": ["E0", "E3"], "execution": copy.deepcopy(EXECUTION),
            "matrix": str(matrix), "scenes": rows, "source_sha256": {},
            "environment": {"path": str(files[0]), "sha256": sha256(files[0])},
            "assets": {"path": str(files[1]), "sha256": sha256(files[1])},
            "files": {str(path): sha256(path) for path in files}}
    monkeypatch.setattr(module, "executable_paths", lambda: [])
    def inspect(path, **kwargs):
        row = next(row for row in rows if Path(row["probe"]) == path)
        return {key: value for key, value in row.items() if key not in ("scene_id", "task", "condition")}
    monkeypatch.setattr(module, "inspect_probe", inspect)
    return plan


def test_eight_episode_plan_covers_shared_lanes_reuse_and_rotation(tmp_path, monkeypatch):
    plan = plan_fixture(tmp_path, monkeypatch)
    path = tmp_path / "plan.json"
    atomic_json(path, plan)
    assert module.load_plan(path) == plan
    schedule = module.schedule_plan(plan, sha256(path))
    episodes = scheduled_episodes(schedule)
    assert len(episodes) == len({row["episode_key"] for row in episodes}) == 8
    assert len(schedule["waves"]) == 2
    assert [[block["group"] for block in wave["blocks"]] for wave in schedule["waves"]] == [["E0", "E3"], ["E3", "E0"]]
    assert all(row["identity"]["split"] == "development" for row in episodes)
    assert all(len(block["episodes"]) == 2 for wave in schedule["waves"] for block in wave["blocks"])


@pytest.mark.parametrize("case", ["budget", "retries", "source", "final", "count", "input", "execution"])
def test_engineering_plan_refuses_expanded_or_changed_inputs(tmp_path, monkeypatch, case):
    plan = plan_fixture(tmp_path, monkeypatch)
    if case == "budget":
        plan["max_episodes"] = 9
    elif case == "retries":
        plan["automatic_retries"] = 1
    elif case == "source":
        plan["source_sha256"] = {"changed": "hash"}
    elif case == "final":
        plan["scenes"][0]["seed"] = 400000
    elif case == "count":
        plan["scenes"].pop()
    elif case == "input":
        Path(plan["environment"]["path"]).write_bytes(b"changed")
    elif case == "execution":
        plan["execution"]["episode_seconds"] *= 2
    path = tmp_path / "plan.json"
    atomic_json(path, plan)
    with pytest.raises(ValueError):
        module.load_plan(path)


def test_engineering_schedule_does_not_accept_duplicate_scenes(tmp_path, monkeypatch):
    plan = plan_fixture(tmp_path, monkeypatch)
    plan["scenes"][1] = copy.deepcopy(plan["scenes"][0])
    with pytest.raises(ValueError, match="distinct"):
        module.schedule_plan(plan, "a" * 64)
