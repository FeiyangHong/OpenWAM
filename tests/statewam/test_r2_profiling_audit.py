from copy import deepcopy

import pytest

from scripts.statewam.r2.audit_profiling import validate_steps
from scripts.statewam.r2.exposure_sampler import ExposureBatchSampler


def evidence():
    identity = {"updates": 2, "micro_batch": 2, "accumulation": 2, "global_batch": 16}
    sampler = ExposureBatchSampler(updates=2, micro_batch=2, accumulation=2, world_size=4, rank=1)
    rows = []
    for index, occurrences in enumerate(sampler):
        update = (index + 1) // 2
        phases = {"forward": {"calls": 1}, "engine_backward": {"calls": 1}}
        if (index + 1) % 2 == 0:
            phases["engine_step"] = {"calls": 1}
        rows.append({"position": {"optimizer_update": update, "next_occurrence": update * 16},
                     "samples": [{"occurrence": i} for i in occurrences], "noise_hashes": [{}, {}],
                     "metrics": {"loss": 0.5}, "gradients": {"action_backbone.weight": [1.0, 1]},
                     "phase_timings": phases})
    return identity, rows


def test_profile_audit_respects_accumulation_and_occurrence_order():
    identity, rows = evidence()
    validate_steps(rows, identity, 1)
    broken = deepcopy(rows)
    broken[1]["samples"].reverse()
    with pytest.raises(ValueError, match="sample/noise"):
        validate_steps(broken, identity, 1)
    with pytest.raises(ValueError, match="microstep count"):
        validate_steps(rows[:-1], identity, 1)


@pytest.mark.parametrize("key", ["engine_step", "engine_backward", "forward"])
def test_missing_actual_engine_phases_are_rejected(key):
    identity, rows = evidence()
    del rows[1]["phase_timings"][key]
    with pytest.raises((ValueError, KeyError)):
        validate_steps(rows, identity, 1)


def test_profile_audit_rejects_nonfinite_loss_or_frozen_world_gradients():
    identity, rows = evidence()
    rows[0]["metrics"]["loss"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_steps(rows, identity, 1)
    rows[0]["metrics"]["loss"] = 0.5
    rows[0]["gradients"]["video_backbone.weight"] = [1.0, 1]
    with pytest.raises(ValueError, match="gradient evidence"):
        validate_steps(rows, identity, 1)
