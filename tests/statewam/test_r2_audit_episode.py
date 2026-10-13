import copy

import numpy as np
import pytest

from benchmarks.robotwin.r2_audit_episode import audit_actions, require_conversion_environment
from benchmarks.robotwin.r2_policy_protocol import chunk_seeds, reset_identity
from benchmarks.utils.action_conversion import eef20d_to_ee16d
from scripts.statewam.r2.common import GOAL_ID, stable_hash


def test_exact_conversion_rejects_different_numpy_environment():
    import sys

    environment = {"goal_id": GOAL_ID, "passed": True, "python": sys.executable, "packages": {"numpy": np.__version__}}
    require_conversion_environment(environment)
    environment["packages"]["numpy"] = "different-version"
    with pytest.raises(ValueError, match="simulator Python/NumPy"):
        require_conversion_environment(environment)


@pytest.fixture
def trace():
    identity = {"goal_id": GOAL_ID, "group": "E0", "task": "adjust_bottle", "condition": "demo_clean",
        "split": "development", "scene_id": "engineering/adjust/200001", "policy_seed": 123,
        "replicate": 0, "protocol_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
        "scene_manifest_sha256": "c" * 64, "instruction_sha256": "d" * 64}
    expected = reset_identity(identity, attempt=1)
    requests = [{"type": "reset_ack", **expected, "next_action": 0, "next_chunk": 0,
        "cleared": {"executor": True, "prompt": True, "vace": True, "velocity": True, "temporary_state": True}}]
    actions = []
    value = np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0, 0] * 2, dtype=np.float32)
    for index in range(33):
        seeds = chunk_seeds(identity, index // 32)
        response = {"type": "action", "episode_key": expected["episode_key"], "attempt": 1,
            "action_index": index, "chunk_index": index // 32, "seeds": seeds, "action": value.tolist(),
            "generation": None if index % 32 else {"chunk_index": index // 32, "seeds": seeds,
                                                   "actions_sha256": stable_hash([value.tolist()] * 32)}}
        actions.append({"action_index": index, "eef20": value.tolist(), "ee16": eef20d_to_ee16d(value).tolist(),
            "response": copy.deepcopy(response), "physics_steps": 4, "control_seconds": 0.1, "request_seconds": 0.01})
        requests.append(response)
    return actions, requests, identity, {"policy_actions": 33, "step_limit": 33, "physics_steps": 132}


def test_audit_binds_ordered_actions_chunks_conversion_and_physics(trace):
    actions, requests, identity, stats = trace
    audit_actions(actions, requests, identity, attempt=1, stats=stats, outcome="failure", reason="action_limit")


@pytest.mark.parametrize("case", ["order", "noise", "conversion", "chunk", "physics", "missing", "early_failure", "reset"])
def test_audit_rejects_inconsistent_physical_evidence(trace, case):
    actions, requests, identity, stats = trace
    if case == "order":
        actions[0]["action_index"] = 1
    elif case == "noise":
        requests[1]["seeds"]["world"] += 1
    elif case == "conversion":
        actions[0]["ee16"][0] += 1
    elif case == "chunk":
        actions[0]["response"]["generation"]["actions_sha256"] = "f" * 64
        requests[1] = copy.deepcopy(actions[0]["response"])
    elif case == "physics":
        stats["physics_steps"] = 0
    elif case == "missing":
        requests.pop()
    elif case == "early_failure":
        stats["step_limit"] += 1
    elif case == "reset":
        requests[0]["cleared"]["executor"] = False
    with pytest.raises(ValueError):
        audit_actions(actions, requests, identity, attempt=1, stats=stats, outcome="failure", reason="action_limit")
