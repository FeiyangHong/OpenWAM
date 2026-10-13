import pytest

from scripts.statewam.r2.select_profile import select


def record(batch, speed, headroom=20):
    return {"micro_batch": batch, "workers": 4, "samples_per_second": speed,
            "minimum_headroom_bytes": headroom * 2**30}


def test_selection_prefers_smaller_batch_within_five_percent():
    small, large = record(4, 10), record(8, 10.4)
    assert select([small, large]) is small
    large["samples_per_second"] = 11
    assert select([small, large]) is large


def test_selection_requires_peak_headroom_even_for_fast_candidate():
    small, large = record(4, 10), record(16, 20, headroom=9)
    assert select([small, large]) is small
    with pytest.raises(ValueError, match="ten GiB"):
        select([large])
