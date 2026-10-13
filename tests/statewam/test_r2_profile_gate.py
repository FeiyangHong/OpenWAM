import json

import pytest

from scripts.statewam.r2.common import GOAL_ID, sha256
from scripts.statewam.r2.profile_training import load_plan
from tests.statewam.test_r2_profile_plan import plan


def files(tmp_path):
    data = plan()
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    manifest = matrix / "initialization.json"
    manifest.write_text("{}")
    digest = sha256(manifest)
    capacity = tmp_path / "capacity.json"
    capacity.write_text(json.dumps({"goal_id": GOAL_ID, "workload_audit_passed": True,
                                    "state_effective_parameter_counts_equal": True,
                                    "groups": {group: {"identity": {"matrix_sha256": digest}}
                                               for group in ("E0", "E1", "E2", "E3")}}))
    data.update(matrix=str(matrix), matrix_sha256=digest, capacity_report=str(capacity),
                capacity_report_sha256=sha256(capacity))
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data))
    return path, capacity, data


def test_profile_requires_passing_intact_capacity_and_matrix(tmp_path):
    path, capacity, data = files(tmp_path)
    assert load_plan(path)["updates"] == 200
    evidence = json.loads(capacity.read_text())
    evidence["workload_audit_passed"] = False
    capacity.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="report changed"):
        load_plan(path)
    data["capacity_report_sha256"] = sha256(capacity)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="gate has not passed"):
        load_plan(path)


def test_profile_rejects_other_matrix_identity(tmp_path):
    path, capacity, data = files(tmp_path)
    evidence = json.loads(capacity.read_text())
    evidence["groups"]["E2"]["identity"]["matrix_sha256"] = "other"
    capacity.write_text(json.dumps(evidence))
    data["capacity_report_sha256"] = sha256(capacity)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="another initialization"):
        load_plan(path)


def test_profile_rejects_partial_capacity_evidence(tmp_path):
    path, capacity, data = files(tmp_path)
    evidence = json.loads(capacity.read_text())
    del evidence["groups"]["E1"]
    capacity.write_text(json.dumps(evidence))
    data["capacity_report_sha256"] = sha256(capacity)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="all four groups"):
        load_plan(path)
