import json

import pytest

from benchmarks.robotwin import r2_cross_process_reset as module
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256, stable_hash
from tests.statewam.test_r2_evaluation_scenes import rows


def test_reset_selection_is_fixed_first_development_scene(rows):
    selected = module.selected_rows({"rows": rows})
    assert len(selected) == 20
    assert all(row["seed"] % 10000 == 0 and row["scene_id"].startswith("development/") for row in selected)
    rows["development"].reverse()
    assert selected == module.selected_rows({"rows": rows})


@pytest.fixture
def completed(tmp_path, rows, monkeypatch):
    scenes_path = tmp_path / "scenes.json"
    scenes = {"rows": rows, "rows_sha256": stable_hash(rows)}
    output = tmp_path / "reset"
    source = output / "source_snapshot/fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("fixture\n")
    atomic_json(output / "source_snapshot/manifest.json", {"fixture.py": sha256(source)})
    initial = {"metadata": {"step_lim": 400}, "rgb": {"head": "fixture"}}
    for index, row in enumerate(module.selected_rows(scenes)):
        probe = tmp_path / f"probe_{index}"
        row["probe"] = str(probe)
        atomic_json(probe / "reset_0/initial.json", initial)
        target = output / f"scene_{index:02d}"
        atomic_json(output / f"row_{index:02d}.json", row)
        atomic_json(target / "initial/initial.json", initial)
        (target / "initial/rgb.npz").write_bytes(b"fixture")
        owner = {"pid": index + 100, "created_at": index + 0.5}
        atomic_json(target / "result.json", {"row_sha256": sha256(output / f"row_{index:02d}.json"),
            "owner": owner, "passed": True, "policy_actions": 0,
            "initial_sha256": sha256(target / "initial/initial.json"), "rgb_sha256": sha256(target / "initial/rgb.npz")})
        atomic_json(output / f"child_{index:02d}.json", {"owner": owner, "exit_code": 0})
    atomic_json(scenes_path, scenes)
    atomic_json(output / "plan.json", {"scenes": str(scenes_path), "scenes_sha256": sha256(scenes_path),
                                      "environment_sha256": "env", "asset_sha256": "assets"})
    for name in ("environment_before.json", "environment_after.json"):
        atomic_json(output / name, {"environment_report_sha256": "env", "asset_report_sha256": "assets"})
    monkeypatch.setattr(module, "alive", lambda owner: False)
    return output


def test_audit_covers_all_pairs_without_policy_actions(completed):
    result = module.audit(completed)
    assert result["passed"] and len(result["checks"]) == 20 and result["goal_id"] == GOAL_ID


@pytest.mark.parametrize("case", ["live", "exit", "initial", "source", "environment", "row", "missing"])
def test_audit_rejects_partial_or_changed_reset_evidence(completed, monkeypatch, case):
    if case == "live":
        monkeypatch.setattr(module, "alive", lambda owner: True)
    elif case == "exit":
        path = completed / "child_00.json"
        atomic_json(path, {**json.loads(path.read_text()), "exit_code": 1})
    elif case == "initial":
        atomic_json(completed / "scene_00/initial/initial.json", {"changed": True})
    elif case == "source":
        (completed / "source_snapshot/fixture.py").write_text("changed\n")
    elif case == "environment":
        atomic_json(completed / "environment_after.json", {"changed": True})
    elif case == "row":
        atomic_json(completed / "row_00.json", {"changed": True})
    else:
        (completed / "child_19.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        module.audit(completed)
