import copy

import pytest

from scripts.statewam.r2.common import TASKS
from scripts.statewam.r2.evaluation_scenes import validate_rows


@pytest.fixture
def rows():
    result = {}
    for split, count, start in (("development", 5, 200000), ("final", 20, 400000)):
        result[split] = [
            {"task": task, "condition": condition, "seed": start + index * 10000 + offset,
             "scene_id": f"{split}/{task}/{condition}/{start + index * 10000 + offset}", "expert_eligible": True}
            for index, task in enumerate(TASKS) for condition in ("demo_clean", "demo_randomized") for offset in range(count)]
    return result


def test_complete_fixed_split_task_condition_counts(rows):
    validate_rows(rows)


@pytest.mark.parametrize("case", ["split", "missing", "duplicate", "wrong_split", "unqualified", "wrong_task_count"])
def test_scene_matrix_rejects_missing_mixed_or_reselected_evidence(rows, case):
    if case == "split":
        del rows["development"]
    elif case == "missing":
        rows["final"].pop()
    elif case == "duplicate":
        rows["final"][1] = copy.deepcopy(rows["final"][0])
    elif case == "wrong_split":
        rows["final"][0] = copy.deepcopy(rows["development"][0])
    elif case == "unqualified":
        rows["final"][0]["expert_eligible"] = False
    elif case == "wrong_task_count":
        value = copy.deepcopy(rows["final"][20])
        value["seed"] += 100
        value["scene_id"] = f"final/{value['task']}/{value['condition']}/{value['seed']}"
        rows["final"][0] = value
    with pytest.raises(ValueError):
        validate_rows(rows)
