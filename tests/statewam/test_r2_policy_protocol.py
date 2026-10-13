import pytest

from benchmarks.robotwin.r2_policy_protocol import chunk_seeds, reset_identity, validate_action_response, validate_reset_ack
from scripts.statewam.r2.common import GOAL_ID


@pytest.fixture
def identity():
    return {"goal_id": GOAL_ID, "group": "E0", "task": "adjust_bottle", "condition": "demo_clean",
            "split": "development", "scene_id": "development/adjust_bottle/demo_clean/200001",
            "policy_seed": 123, "replicate": 0, "protocol_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
            "scene_manifest_sha256": "c" * 64, "instruction_sha256": "d" * 64}


def test_all_groups_checkpoints_and_replicates_share_scene_chunk_noise(identity):
    for index in range(20):
        expected = chunk_seeds(identity, index)
        assert expected["world"] != expected["action"]
        assert all(0 <= seed < 2**63 for seed in expected.values())
        for group in ("E0", "E1", "E2", "E3"):
            assert chunk_seeds({**identity, "group": group, "checkpoint_sha256": "f" * 64}, index) == expected
        assert chunk_seeds({**identity, "split": "repeatability", "replicate": 1}, index) == expected
    assert len({seed for index in range(20) for seed in chunk_seeds(identity, index).values()}) == 40
    assert chunk_seeds({**identity, "policy_seed": 124}, 0) != chunk_seeds(identity, 0)
    assert chunk_seeds({**identity, "condition": "demo_randomized"}, 0) != chunk_seeds(identity, 0)


def test_reset_attempts_differ_but_noise_does_not(identity):
    first, retry = [reset_identity(identity, attempt=i) for i in (1, 2)]
    assert first["episode_key"] == retry["episode_key"] and first["attempt"] != retry["attempt"]
    repeated = {**identity, "split": "repeatability"}
    assert reset_identity(repeated, attempt=1)["episode_key"] != reset_identity({**repeated, "replicate": 1}, attempt=1)["episode_key"]
    with pytest.raises(ValueError):
        reset_identity(identity, attempt=3)


def test_reset_ack_requires_all_state_clearing_and_exact_identity(identity):
    expected = reset_identity(identity, attempt=1)
    ack = {"type": "reset_ack", **expected, "next_action": 0, "next_chunk": 0,
           "cleared": {"executor": True, "prompt": True, "vace": True, "velocity": True, "temporary_state": True}}
    validate_reset_ack(ack, expected)
    ack["cleared"]["prompt"] = False
    with pytest.raises(ValueError):
        validate_reset_ack(ack, expected)
    with pytest.raises(ValueError):
        validate_reset_ack({"type": "reset_ack"}, expected)


def test_action_response_rejects_previous_attempt_and_duplicate_indices(identity):
    expected = reset_identity(identity, attempt=2)
    response = {"type": "action", "episode_key": expected["episode_key"], "attempt": 2,
                "action_index": 32, "chunk_index": 1, "seeds": chunk_seeds(identity, 1)}
    validate_action_response(response, expected, 32)
    for change in ({"attempt": 1}, {"action_index": 31}, {"chunk_index": 0}, {"seeds": chunk_seeds(identity, 0)}):
        with pytest.raises(ValueError):
            validate_action_response({**response, **change}, expected, 32)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_chunk_indices_are_rejected(identity, value):
    with pytest.raises(ValueError):
        chunk_seeds(identity, value)
