import copy
import json

import pytest

from scripts.statewam.r2 import freeze_protocol as module
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256
from scripts.statewam.r2.formal_training import validate_recipe
from tests.statewam.test_r2_formal_training import recipe
from tests.statewam.test_r2_evaluation_plan import scene_fixture


def test_protocol_requires_frozen_evaluation_execution():
    plan = recipe()
    validate_recipe(plan)
    del plan["evaluation_execution"]
    with pytest.raises(ValueError, match="execution"):
        validate_recipe(plan)


def test_failed_gate_never_publishes_final_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "make_plan", lambda artifacts: recipe())
    def fail(path):
        raise ValueError("Missing physical gate")
    monkeypatch.setattr(module, "load_protocol", fail)
    output = tmp_path / "protocol.yaml"
    with pytest.raises(ValueError, match="physical"):
        module.freeze(output)
    assert not output.exists() and not (tmp_path / "protocol.COMPLETE.json").exists()


def test_protocol_publish_is_immutable_after_validated_staging(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "make_plan", lambda artifacts: recipe())
    seen = []
    monkeypatch.setattr(module, "load_protocol", lambda path: seen.append(path))
    output = tmp_path / "protocol.yaml"
    complete = module.freeze(output)
    assert seen and complete == {"goal_id": GOAL_ID, "protocol_sha256": sha256(output)}
    assert json.loads((tmp_path / "protocol.COMPLETE.json").read_text()) == complete
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.freeze(output)
    assert output.read_bytes() == original


def test_existing_partial_protocol_is_not_overwritten(tmp_path):
    output = tmp_path / "protocol.yaml"
    output.write_text("partial\n")
    with pytest.raises(FileExistsError):
        module.freeze(output)
    assert output.read_text() == "partial\n"


def test_evaluation_estimate_uses_original_limits_and_separates_assumptions(tmp_path):
    scenes = scene_fixture()
    probe = tmp_path / "probe"
    atomic_json(probe / "reset_0/initial.json", {"metadata": {"step_lim": 400}})
    for rows in scenes["rows"].values():
        for row in rows:
            row["probe"] = str(probe)
    proxies = [{"policy_actions": 400, "service_seconds": 300, "startup_seconds": 150,
                "preflight_and_teardown_residual_seconds": 400} for _ in range(3)]
    estimate = module.estimate_evaluation(scenes, proxies)
    assert estimate["planning_wall_hours"] == pytest.approx((400 + 90 * 150 + 1020 * 300 * 1.5) / 3600)
    assert estimate["planning_allocated_gpu_hours"] == pytest.approx(3 * estimate["planning_wall_hours"])
    assert [row["assumed_rollout_slowdown_factor"] for row in estimate["sensitivity_scenarios_not_confidence_intervals"]] == [1, 1.5, 2]
    atomic_json(probe / "reset_0/initial.json", {"metadata": {"step_lim": 800}})
    doubled = module.estimate_evaluation(scenes, proxies)
    assert doubled["planning_wall_hours"] == pytest.approx((400 + 90 * 150 + 1020 * 600 * 1.5) / 3600)
    proxies[0]["service_seconds"] = float("nan")
    with pytest.raises(ValueError, match="Invalid measured"):
        module.estimate_evaluation(scenes, proxies)
