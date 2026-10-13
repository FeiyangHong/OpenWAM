import copy

import numpy as np
import pytest

from scripts.statewam.r2 import paired_statistics as module
from scripts.statewam.r2.evaluation_plan import scheduled_episodes
from tests.statewam.test_r2_evaluation_plan import schedule_fixture


def records_fixture(schedule):
    return {row["episode_key"]: {"identity": row["identity"], "outcome": "failure", "audited": True}
            for row in scheduled_episodes(schedule)}


def test_exact_pairing_not_independent_group_bootstrap():
    values = np.broadcast_to(np.arange(20) % 2, (4, 10, 20))
    report = module.paired_bootstrap(values, rng=np.random.default_rng(1), samples=300)
    for row in report["differences_pp"].values():
        assert row == {"estimate": 0, "ci95": {"lower": 0, "upper": 0, "degenerate": True}}
    assert report["rates"]["E0"]["estimate"] == 0.5
    assert report["rates"]["E0"]["ci95"]["lower"] < report["rates"]["E0"]["ci95"]["upper"]


def test_task_equal_point_estimate_and_all_prespecified_differences():
    values = np.zeros((4, 10, 20), dtype=np.uint8)
    values[3, :3] = 1
    report = module.paired_bootstrap(values, rng=np.random.default_rng(1), samples=300)
    assert report["rates"]["E3"]["estimate"] == 0.3
    assert report["differences_pp"]["E3-E0"]["estimate"] == 30
    assert set(report["differences_pp"]) == {"E3-E0", "E3-E1", "E2-E1", "E3-E2"}


@pytest.mark.parametrize("change", ["missing", "extra", "invalid", "unaudited", "identity"])
def test_no_silent_pair_drop_or_invalid_to_failure(change):
    schedule = schedule_fixture()
    records = records_fixture(schedule)
    first = next(iter(records))
    if change == "missing":
        records.pop(first)
    elif change == "extra":
        records["extra"] = records[first]
    elif change == "invalid":
        records[first]["outcome"] = "invalid"
    elif change == "unaudited":
        records[first]["audited"] = False
    else:
        records[first]["identity"] = {**records[first]["identity"], "policy_seed": 123}
    with pytest.raises(ValueError):
        module.summarize(schedule, records)


def test_split_denominators_repeatability_excluded_and_reproducibility(monkeypatch):
    schedule = schedule_fixture()
    records = records_fixture(schedule)
    original = module.paired_bootstrap
    monkeypatch.setattr(module, "paired_bootstrap", lambda value, **kw: original(value, samples=100, **kw))
    report = module.summarize(schedule, records)
    assert report == module.summarize(schedule, copy.deepcopy(records))
    assert len(report["repeatability"]) == 20
    for key, row in report["conditions"].items():
        assert row["denominator_per_group"] == (50 if key.startswith("development") else 200)
        assert row["macro"]["rates"]["E0"]["ci95"]["degenerate"]
        assert len(row["per_task"]) == 10
