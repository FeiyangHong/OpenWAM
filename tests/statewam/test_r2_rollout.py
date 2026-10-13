import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.robotwin import r2_rollout as rollout
from benchmarks.robotwin.r2_policy_protocol import reset_identity
from scripts.statewam.r2.common import GOAL_ID, atomic_json, stable_hash


def test_physics_counter_calls_original_once_and_restores_on_error():
    class Scene:
        def __init__(self):
            self.calls = 0

        def step(self):
            self.calls += 1
            return self.calls

    scene, other, original = Scene(), Scene(), Scene.step
    with pytest.raises(RuntimeError):
        with rollout.count_physics_steps(scene) as counter:
            assert scene.step() == 1
            other.step()
            scene.step()
            assert counter["steps"] == 2
            raise RuntimeError("test")
    assert Scene.step is original and scene.calls == 2 and other.calls == 1


def test_actual_h264_video_closes_and_decodes_all_frames(tmp_path):
    frame = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3)
    video = rollout.EpisodeVideo(tmp_path / "episode.mp4", shape=frame.shape)
    for i in range(4):
        video.write(np.roll(frame, i, axis=0))
    result = video.close()
    assert result["frames"] == 4 and result["stream"]["nb_read_frames"] == "4"
    assert result["stream"]["width"] == 48


def test_initial_scene_mismatch_is_not_a_policy_failure():
    expected = {"metadata": {"pose": [0., 1.]}, "rgb": {"head": "a"}}
    rollout.verify_initial(expected, expected)
    with pytest.raises(ValueError, match="Initial scene differs"):
        rollout.verify_initial(expected, {**expected, "rgb": {"head": "b"}})


@pytest.mark.parametrize("success_after,expected", [(2, "success"), (None, "failure")])
def test_episode_uses_original_limits_real_ledger_video_and_no_duplicate_rollout(tmp_path, monkeypatch, success_after, expected):
    frame = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3)
    observation = {"observation": {name: {"rgb": frame} for name in ("head_camera", "left_camera", "right_camera")},
                   "endpose": {"left_endpose": [0, 0, 0, 0, 0, 0, 1], "right_endpose": [0, 0, 0, 0, 0, 0, 1],
                               "left_gripper": 0., "right_gripper": 0.}}
    instruction, config = "Adjust the bottle.", {"task_name": "adjust_bottle"}
    identity = {"goal_id": GOAL_ID, "group": "E0", "task": "adjust_bottle", "condition": "demo_clean",
                "split": "development", "scene_id": "development/adjust_bottle/demo_clean/200001", "policy_seed": 123,
                "replicate": 0, "protocol_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
                "scene_manifest_sha256": "c" * 64, "instruction_sha256": stable_hash(instruction)}
    probe = tmp_path / "probe"
    initial = {"metadata": {"pose": [0., 1.]}, "rgb": {"head": "a"}}
    atomic_json(probe / "reset_0/initial.json", initial)
    row = {"probe": str(probe), "scene_id": identity["scene_id"], "seed": 200001, "policy_seed": 123,
           "expert_eligible": True, "config_sha256": stable_hash(config),
           "instruction": {"text_sha256": stable_hash(instruction), "episode_parameters": {}}}
    monkeypatch.setattr(rollout, "inspect_probe", lambda *args, **kw: {k: v for k, v in row.items() if k != "scene_id"})
    monkeypatch.setattr(rollout, "configuration", lambda *args: (config, 3))
    monkeypatch.setattr(rollout, "replay_instruction", lambda *args, **kw: instruction)

    class Scene:
        render_system = SimpleNamespace(device=SimpleNamespace(cuda_id=0))

        def step(self):
            pass

        def get_timestep(self):
            return 1 / 250

    class Env:
        def __init__(self):
            self.scene, self.step_lim, self.take_action_cnt, self.eval_success = Scene(), 3, 0, False
            self.obs_calls, self.actions, self.closed = 0, [], 0
            self.now_obs = observation

        def setup_demo(self, **kwargs):
            assert kwargs["seed"] == 200001 and kwargs["is_test"]

        def set_instruction(self, value):
            self.instruction = value

        def get_instruction(self):
            return self.instruction

        def get_obs(self):
            self.obs_calls += 1
            return observation

        def take_action(self, action, action_type):
            assert action.shape == (16,) and action_type == "ee"
            self.actions.append(action)
            self.take_action_cnt += 1
            for _ in range(4):
                self.scene.step()
            self.eval_success = self.take_action_cnt == success_after

        def close_env(self):
            self.closed += 1

    class Client:
        def __init__(self, url):
            pass

        def reset(self, value, **kwargs):
            return reset_identity(value, attempt=kwargs["attempt"])

        def predict(self, **kwargs):
            assert kwargs["instruction"] == instruction and kwargs["state"].shape == (20,)
            return np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0] * 2, dtype=np.float32), {"type": "action"}

        def close(self):
            pass

    def save_initial(env, directory):
        obs = env.get_obs()
        atomic_json(directory / "initial.json", initial)
        return obs, {}, {}

    env = Env()
    monkeypatch.setattr(rollout, "initial_observation", save_initial)
    monkeypatch.setattr(rollout, "PairedClient", Client)
    options = dict(robotwin_root=tmp_path, scene_row=row, identity=identity, ledger_root=tmp_path / "ledger",
                   service_url="unused", stop_file=tmp_path / "STOP")
    module = SimpleNamespace(class_decorator=lambda task: env)
    assert rollout.run_episode(module, **options) == expected
    result = json.loads(next((tmp_path / "ledger").glob("*/attempt_001/result.json")).read_text())
    count = success_after or 3
    assert result["stats"]["policy_actions"] == count and result["stats"]["physics_steps"] == count * 4
    assert env.obs_calls == count and env.closed == 1
    assert rollout.run_episode(module, **options) == "already_complete"
    assert len(env.actions) == count
