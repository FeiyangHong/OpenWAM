from pathlib import Path
import subprocess
import sys

import pytest

from scripts.statewam.r2.common import GOAL_ID
from scripts.statewam.r2.profile_training import validate_plan


def plan():
    return {"goal_id": GOAL_ID, "purpose": "profiling", "groups": ["E0", "E3"], "updates": 200,
            "world_size": 4, "micro_batch": 4, "accumulation": 1, "workers": 4, "save_every": 100,
            "global_batch": 16, "seed": 20260914, "warmup_updates": 10, "protocol_id": "r2-profile-v1"}


def test_valid_bounded_plan():
    assert validate_plan(plan())["updates"] == 200


@pytest.mark.parametrize("key,value", [("updates", 199), ("updates", 501), ("world_size", 5),
                                      ("groups", ["E3"]), ("purpose", "training"), ("global_batch", 32),
                                      ("micro_batch", 0), ("workers", -1), ("save_every", 201),
                                      ("seed", 42), ("warmup_updates", 0), ("protocol_id", "formal-v1")])
def test_invalid_scope_or_budget(key, value):
    data = plan()
    data[key] = value
    with pytest.raises(ValueError):
        validate_plan(data)


def test_direct_engineering_entry_import_does_not_shadow_standard_library():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "scripts/statewam/r2/train.py", "--help"],
                            cwd=root, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "--audit-workload" in result.stdout
