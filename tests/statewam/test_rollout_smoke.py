import json
import sys
from types import SimpleNamespace

import pytest

from scripts.statewam.audit_rollouts import validate_frame_count
from scripts.statewam.rollout_smoke import run


@pytest.mark.parametrize("success,frames", [(True, 13), (False, 12)])
def test_video_frame_count_matches_upstream_success_frame(success, frames):
    episode = {"action_steps": 12, "step_limit": 12, "success": success}
    assert validate_frame_count(episode, frames) == frames


@pytest.mark.parametrize("steps,limit,success,frames", [
    (12, 12, True, 12), (12, 12, False, 13), (0, 12, False, 0), (13, 12, True, 14),
])
def test_video_frame_count_rejects_incomplete_or_invalid_episode(steps, limit, success, frames):
    with pytest.raises(ValueError):
        validate_frame_count({"action_steps": steps, "step_limit": limit, "success": success}, frames)


def test_rollout_records_actual_count_and_not_expert_episode(monkeypatch, tmp_path):
    env = SimpleNamespace(eval_success=False, take_action_cnt=0, step_lim=12)
    env.setup_demo = lambda **kw: None
    env.close_env = lambda **kw: None
    env.get_instruction = lambda: "Adjust the bottle"

    def evaluate(env, model, observation):
        env.take_action_cnt = 12

    module = SimpleNamespace(eval_function_decorator=lambda policy, name: evaluate)

    def original_eval(task, env, cfg, model, seed, test_num, **kw):
        assert test_num == 1
        env.setup_demo(seed=seed)
        env.close_env()
        env.setup_demo(seed=seed)
        module.eval_function_decorator("policy", "eval")(env, model, {})
        env.close_env()
        return seed + 1, 0

    module.eval_policy = original_eval
    module.main = lambda config: module.eval_policy("adjust_bottle", env, {"eval_video_save_dir": tmp_path}, None, 100001)
    monkeypatch.setitem(sys.modules, "eval_policy_wrapper", SimpleNamespace(bootstrap_robotwin_module=lambda: module))
    args = SimpleNamespace(task="adjust_bottle", condition="demo_clean", label="test", episodes=1, port=9000)
    run(args, tmp_path)
    result = json.loads((tmp_path / "smoke_result.json").read_text())
    assert result["pass"] is True
    assert result["count"] == 1 and result["successes"] == 0
    assert result["episodes"][0]["seed"] == 100001
    assert result["episodes"][0]["action_steps"] == 12
