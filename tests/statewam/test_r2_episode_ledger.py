import json
import multiprocessing

import pytest

from benchmarks.robotwin.r2_episode_ledger import EpisodeLedger, RetryExhausted
from scripts.statewam.r2.common import GOAL_ID


@pytest.fixture
def identity():
    return {"goal_id": GOAL_ID, "group": "E0", "task": "adjust_bottle", "condition": "demo_clean",
            "split": "development", "scene_id": "development/adjust_bottle/demo_clean/200001",
            "policy_seed": 123, "replicate": 0, "protocol_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
            "scene_manifest_sha256": "c" * 64, "instruction_sha256": "d" * 64}


def finish(ledger, path, outcome="failure"):
    (path / "video.mp4").write_bytes(b"fixture video; not a simulator validation")
    (path / "rollout.json").write_text("{}")
    ledger.finish(path, outcome=outcome, reason="action_limit" if outcome == "failure" else "goal_reached",
                  stats={"policy_actions": 10, "physics_steps": 100, "wall_seconds": 1.2},
                  artifacts={"video": "video.mp4", "rollout": "rollout.json"})


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_policy_outcomes_are_complete_and_never_retried(tmp_path, identity, outcome):
    with EpisodeLedger(tmp_path, identity) as ledger:
        path = ledger.begin()
        finish(ledger, path, outcome)
    with EpisodeLedger(tmp_path, identity) as ledger:
        assert ledger.begin() is None
        assert len(list(ledger.directory.glob("attempt_*"))) == 1


def test_one_infrastructure_retry_only(tmp_path, identity):
    with EpisodeLedger(tmp_path, identity) as ledger:
        for index in range(2):
            path = ledger.begin()
            assert path.name == f"attempt_{index + 1:03d}"
            (path / "error.log").write_text("documented socket disconnect")
            ledger.finish(path, outcome="invalid", reason="transport_disconnect", stats={}, artifacts={"evidence": "error.log"})
        with pytest.raises(RetryExhausted):
            ledger.begin()
        assert not (ledger.directory / "COMPLETE.json").exists()


def test_model_failure_cannot_be_reclassified_as_infrastructure(tmp_path, identity):
    with EpisodeLedger(tmp_path, identity) as ledger:
        path = ledger.begin()
        (path / "error.log").write_text("NaN action")
        with pytest.raises(ValueError, match="infrastructure"):
            ledger.finish(path, outcome="invalid", reason="nonfinite_action", stats={}, artifacts={"evidence": "error.log"})
        with pytest.raises(RuntimeError, match="Unclassified"):
            ledger.begin()


def test_artifact_corruption_prevents_skip_and_rerun(tmp_path, identity):
    with EpisodeLedger(tmp_path, identity) as ledger:
        path = ledger.begin()
        finish(ledger, path)
        (path / "video.mp4").write_bytes(b"corrupted")
        with pytest.raises(ValueError, match="evidence changed"):
            ledger.begin()


def test_result_survives_interrupted_completion_pointer(tmp_path, identity, monkeypatch):
    with EpisodeLedger(tmp_path, identity) as ledger:
        path = ledger.begin()
        monkeypatch.setattr(ledger, "_publish_complete", lambda path: None)
        finish(ledger, path)
        assert not (ledger.directory / "COMPLETE.json").exists()
    with EpisodeLedger(tmp_path, identity) as ledger:
        assert ledger.begin() is None
        assert (ledger.directory / "COMPLETE.json").exists()


def test_lock_and_live_owner_prevent_duplicate_execution(tmp_path, identity):
    with EpisodeLedger(tmp_path, identity) as ledger:
        path = ledger.begin()
        with pytest.raises(BlockingIOError):
            with EpisodeLedger(tmp_path, identity):
                pass
        (path / "error.log").write_text("not sufficient while owner lives")
        with pytest.raises(RuntimeError, match="still alive"):
            ledger.record_interruption(reason="external_interrupt", evidence="error.log")


def partial_episode(root, identity, ready):
    import time

    with EpisodeLedger(root, identity) as ledger:
        path = ledger.begin()
        (path / "partial.log").write_text("episode started")
        ready.set()
        time.sleep(30)


def test_real_process_termination_requires_evidence_before_one_retry(tmp_path, identity):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    child = context.Process(target=partial_episode, args=(tmp_path, identity, ready))
    child.start()
    try:
        assert ready.wait(15)
        child.kill()
        child.join(timeout=10)
        assert child.exitcode is not None
        with EpisodeLedger(tmp_path, identity) as ledger:
            with pytest.raises(RuntimeError, match="Unclassified"):
                ledger.begin()
            path = ledger.directory / "attempt_001"
            (path / "interruption.json").write_text(json.dumps({"test_child_pid": child.pid, "exit_code": child.exitcode}))
            ledger.record_interruption(reason="external_interrupt", evidence="interruption.json")
            second = ledger.begin()
            assert second.name == "attempt_002"
            finish(ledger, second)
            assert ledger.begin() is None
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=10)


def test_identity_changes_produce_distinct_episode_keys(tmp_path, identity):
    original = EpisodeLedger(tmp_path, identity).key
    for key, value in (("group", "E3"), ("policy_seed", 124), ("checkpoint_sha256", "e" * 64)):
        assert EpisodeLedger(tmp_path, {**identity, key: value}).key != original
    with pytest.raises(ValueError):
        EpisodeLedger(tmp_path, {**identity, "policy_seed": True})


def test_artifacts_must_be_inside_the_attempt(tmp_path, identity):
    external = tmp_path / "outside.log"
    external.write_text("outside")
    with EpisodeLedger(tmp_path, identity) as ledger:
        path = ledger.begin()
        with pytest.raises(ValueError, match="inside its own attempt"):
            ledger.finish(path, outcome="invalid", reason="device_failure", stats={}, artifacts={"evidence": str(external)})


def test_completed_origin_and_pointer_cannot_be_rewritten(tmp_path, identity):
    with EpisodeLedger(tmp_path, identity) as ledger:
        path = ledger.begin()
        finish(ledger, path)
        origin = json.loads((path / "begin.json").read_text())
        origin["started_at"] += 1
        (path / "begin.json").write_text(json.dumps(origin))
        with pytest.raises(ValueError, match="origin changed"):
            ledger.begin()


def test_second_attempt_policy_failure_is_not_a_third_retry(tmp_path, identity):
    with EpisodeLedger(tmp_path, identity) as ledger:
        first = ledger.begin()
        (first / "error.log").write_text("connection failed before action acknowledgement")
        ledger.finish(first, outcome="invalid", reason="transport_disconnect", stats={}, artifacts={"evidence": "error.log"})
        second = ledger.begin()
        finish(ledger, second, "failure")
        assert ledger.begin() is None
        assert json.loads((first / "result.json").read_text())["outcome"] == "invalid"
        assert json.loads((second / "result.json").read_text())["outcome"] == "failure"
