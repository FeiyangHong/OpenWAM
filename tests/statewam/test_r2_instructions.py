import json
import random

import pytest

from benchmarks.robotwin.r2_instructions import freeze_instruction, replay_instruction


@pytest.fixture
def instruction_root(tmp_path):
    for directory in ("utils", "task_instruction", "objects_description"):
        (tmp_path / "description" / directory).mkdir(parents=True)
    (tmp_path / "description/utils/generate_episode_instructions.py").write_text(
        'import random\n'
        'def generate_episode_descriptions(task, episodes, max_descriptions):\n'
        '    return [{"unseen": ["Place it " + str(random.random()) for _ in range(4)]}]\n')
    (tmp_path / "description/task_instruction/adjust_bottle.json").write_text('{}')
    return tmp_path


def test_generation_is_reproducible_and_preserves_environment_rng(instruction_root):
    before = random.getstate()
    first = freeze_instruction(instruction_root, "adjust_bottle", {}, seed=42)
    assert random.getstate() == before
    random.random()
    second = freeze_instruction(instruction_root, "adjust_bottle", {}, seed=42)
    assert first == second
    assert replay_instruction(first, robotwin_root=instruction_root, task="adjust_bottle",
                              episode_parameters={}) == first["text"]


def test_replay_rejects_text_scene_and_source_drift(instruction_root):
    first = freeze_instruction(instruction_root, "adjust_bottle", {}, seed=42)
    args = dict(robotwin_root=instruction_root, task="adjust_bottle", episode_parameters={})
    with pytest.raises(ValueError, match="integrity"):
        replay_instruction({**first, "text": "a different instruction"}, **args)
    with pytest.raises(ValueError, match="parameters"):
        replay_instruction(first, **{**args, "episode_parameters": {"{a}": "left"}})
    (instruction_root / "description/task_instruction/adjust_bottle.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="source"):
        replay_instruction(first, **args)


def test_records_upstream_unseen_object_fallback(instruction_root):
    (instruction_root / "description/objects_description/bottle.json").write_text(
        json.dumps({"seen": ["bottle"], "unseen": []}))
    record = freeze_instruction(instruction_root, "adjust_bottle", {"{A}": "bottle"}, seed=42)
    assert record["unseen_object_fallback_to_seen"] == ["description/objects_description/bottle.json"]
