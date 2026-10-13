import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.robotwin.r2_scene_manifest import completed_attempt, inspect_probe, seed_start
from scripts.statewam.r2.common import GOAL_ID, TASKS, stable_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def rejected(tmp_path):
    config = {"task_name": "adjust_bottle", "eval_mode": True}
    result = {"goal_id": GOAL_ID, "task": "adjust_bottle", "condition": "demo_clean", "seed": 200000,
              "config_sha256": stable_hash(config), "completed": True, "expert_eligible": False,
              "rejection": {"kind": "unstable_initial_scene", "detail": "unstable"}}
    write(tmp_path / "scene_probe.json", result)
    write(tmp_path / "config.json", config)
    return result


def test_scene_seed_ranges_are_fixed_and_disjoint():
    ranges = []
    for task in TASKS:
        for split in ("development", "final"):
            start = seed_start(task, split)
            current = set(range(start, start + 1000))
            assert all(not current & previous for previous in ranges)
            ranges.append(current)
    assert seed_start("adjust_bottle", "development") == 200000
    with pytest.raises(ValueError):
        seed_start("unknown", "final")


def test_complete_rejection_is_reused_but_incomplete_exit_zero_is_not(tmp_path):
    directory = tmp_path / "attempt_001"
    write(directory / "attempt.json", {"exit_code": 0})
    assert not completed_attempt(directory)
    result = rejected(directory / "probe")
    assert completed_attempt(directory)
    result.update(completed=False, stopped=True)
    write(directory / "probe/scene_probe.json", result)
    assert not completed_attempt(directory)


def test_rejection_identity_and_config_integrity(tmp_path):
    result = rejected(tmp_path)
    args = {"task": "adjust_bottle", "condition": "demo_clean", "seed": 200000}
    row = inspect_probe(tmp_path, **args)
    assert not row["expert_eligible"] and row["rejection"]["kind"] == "unstable_initial_scene"
    with pytest.raises(ValueError, match="identity mismatch"):
        inspect_probe(tmp_path, **{**args, "seed": 200001})
    result["rejection"]["kind"] = "cuda_oom"
    write(tmp_path / "scene_probe.json", result)
    with pytest.raises(ValueError, match="Infrastructure failure"):
        inspect_probe(tmp_path, **args)


def test_accepted_scene_differences_do_not_silently_select_another_seed(tmp_path):
    result = rejected(tmp_path)
    result.update(expert_eligible=True, numeric_differences=[{"field": "pose", "absolute_difference": 0.1}],
                  rgb_differences={})
    write(tmp_path / "scene_probe.json", result)
    with pytest.raises(ValueError, match="reset differs"):
        inspect_probe(tmp_path, task="adjust_bottle", condition="demo_clean", seed=200000)


def test_saved_accepted_evidence_is_independently_rechecked(tmp_path):
    import hashlib
    result = rejected(tmp_path)
    views = ("head_camera", "left_camera", "right_camera")
    image = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    info = {view: {"shape": list(image.shape), "sha256": hashlib.sha256(image.tobytes()).hexdigest()} for view in views}
    for index in range(2):
        path = tmp_path / f"reset_{index}"
        write(path / "initial.json", {"metadata": {"pose": [1, 2, 3]}, "rgb": info})
        np.savez_compressed(path / "rgb.npz", **{view: image for view in views})
    instruction = {"task": "adjust_bottle", "text": "Adjust the bottle", "text_sha256": stable_hash("Adjust the bottle")}
    instruction["identity_sha256"] = stable_hash(instruction)
    write(tmp_path / "instruction.json", instruction)
    result.update(expert_eligible=True, numeric_differences=[],
                  rgb_differences={view: {"max": 0, "mean": 0, "fraction_changed": 0} for view in views})
    write(tmp_path / "scene_probe.json", result)
    args = {"task": "adjust_bottle", "condition": "demo_clean", "seed": 200000}
    row = inspect_probe(tmp_path, **args)
    assert row["expert_eligible"] and row["instruction"]["text"] == instruction["text"]
    write(tmp_path / "reset_1/initial.json", {"metadata": {"pose": [1, 2, 4]}, "rgb": info})
    with pytest.raises(ValueError, match="not exactly repeated"):
        inspect_probe(tmp_path, **args)


def test_shard_resume_reuses_complete_scenes_and_retries_interrupted_candidate(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from benchmarks.robotwin import eval_policy_wrapper, r2_scene_manifest as module

    monkeypatch.setenv("ROBOTWIN_PATH", str(tmp_path / "robotwin"))
    monkeypatch.setattr(eval_policy_wrapper, "_robotwin_git", lambda root, *args:
                        eval_policy_wrapper.VERIFIED_ROBOTWIN_COMMIT if args[0] == "rev-parse" else "")
    monkeypatch.setitem(module.COUNTS, "development", 2)
    monkeypatch.setattr(module, "inspect_probe", lambda directory, **kwargs: {"seed": kwargs["seed"], "expert_eligible": True})
    first_stop, second_stop = tmp_path / "stop_1", tmp_path / "stop_2"
    calls = []

    def execute(command, **kwargs):
        seed = int(command[command.index("--seed") + 1])
        output = Path(command[command.index("--output") + 1])
        calls.append(seed)
        interrupted = len(calls) == 2
        write(output / "scene_probe.json", {"completed": not interrupted, "stopped": interrupted})
        if interrupted:
            first_stop.touch()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", execute)
    output = tmp_path / "shard"
    args = {"task": "adjust_bottle", "condition": "demo_clean", "split": "development"}
    assert module.run_shard(output, **args, stop_file=first_stop) == "stopped"
    assert not (output / "COMPLETE.json").exists()
    assert module.run_shard(output, **args, stop_file=second_stop, resume=True) == "completed"
    assert calls == [200000, 200001, 200001]
    assert (output / "candidates/200001/attempt_001/probe/scene_probe.json").exists()
    assert (output / "candidates/200001/attempt_002/probe/scene_probe.json").exists()
    with pytest.raises(ValueError, match="already complete"):
        module.run_shard(output, **args, stop_file=second_stop, resume=True)
