from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.robotwin.r2_scene_probe import initial_observation, json_value, numeric_difference, record_probe_error


def test_only_expert_instability_is_an_eligibility_rejection():
    class UnstableError(Exception):
        pass

    record = {"completed": False}
    assert record_probe_error(record, UnstableError("object moved"), unstable_error=UnstableError, phase="expert")
    assert record["completed"] and not record["expert_eligible"]
    assert record["rejection"]["kind"] == "unstable_initial_scene"
    for phase, error in (("replay", UnstableError("object moved")), ("expert", RuntimeError("renderer failed"))):
        record = {"completed": False}
        assert not record_probe_error(record, error, unstable_error=UnstableError, phase=phase)
        assert not record["completed"] and "error" in record and "rejection" not in record


def test_scene_difference_rejects_identity_and_nonfinite_values():
    assert numeric_difference({"pose": [1.0, 2.0]}, {"pose": [1.0, 2.1]})[0]["field"] == "metadata.pose[1]"
    with pytest.raises(ValueError, match="identity"):
        numeric_difference({"asset": "a"}, {"asset": "b"})
    with pytest.raises(ValueError, match="count"):
        numeric_difference([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="Nonfinite"):
        json_value(np.array([np.nan]))


@pytest.mark.parametrize("phase,with_pose,expected", [("expert", False, True), ("replay", False, False), ("expert", True, False)])
def test_no_grasp_classifier_requires_exact_upstream_frames_and_none_values(tmp_path, phase, with_pose, expected):
    root = tmp_path / "robotwin"
    action = root / "envs/utils/action.py"
    base = root / "envs/_base_task.py"
    action.parent.mkdir(parents=True)
    action.write_text('def __init__(target_pose):\n    assert target_pose is not None, "target_pose cannot be None for move action."\n')
    base.write_text('def grasp_actor():\n    pre_grasp_pose = None\n    grasp_pose = override\n    __init__(pre_grasp_pose)\n')
    namespace = {"override": [1] if with_pose else None}
    exec(compile(action.read_text(), str(action), "exec"), namespace)
    exec(compile(base.read_text(), str(base), "exec"), namespace)
    try:
        namespace["grasp_actor"]()
    except AssertionError as error:
        record = {"completed": False}
        assert record_probe_error(record, error, unstable_error=KeyError, phase=phase, robotwin_root=root) is expected
        if expected:
            assert record["rejection"]["classifier"] == "r2-expert-no-grasp-pose-v1"
            assert len(record["rejection"]["evidence"]) == 2
        else:
            assert not record["completed"]
        assert not record_probe_error({"completed": False}, error, unstable_error=KeyError, phase=phase, robotwin_root=tmp_path / "wrong")


def test_bare_matching_assertion_is_not_enough_to_reject_a_scene(tmp_path):
    record = {"completed": False}
    assert not record_probe_error(record, AssertionError("target_pose cannot be None for move action."),
                                  unstable_error=KeyError, phase="expert", robotwin_root=tmp_path)
    assert not record["completed"]


def test_initial_observation_calls_renderer_once_and_preserves_three_views(tmp_path):
    calls = []
    rgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    observation = {"observation": {name: {"rgb": rgb, "intrinsic_cv": np.eye(3)}
                                   for name in ("head_camera", "left_camera", "right_camera")},
                   "endpose": {}, "joint_action": {}}

    def get_obs():
        calls.append(1)
        return observation

    scene = SimpleNamespace(get_all_actors=lambda: [], get_all_articulations=lambda: [],
                            get_timestep=lambda: 1 / 250, ambient_light=np.ones(3))
    env = SimpleNamespace(get_obs=get_obs, scene=scene, direction_light_lst=[], point_light_lst=[],
                          table_z_bias=0.0, crazy_random_light=False, info={}, step_lim=300)
    first, metadata, images = initial_observation(env, tmp_path / "initial")
    assert calls == [1]
    assert first is observation
    assert len(images) == 3
    assert metadata["step_lim"] == 300
    with np.load(tmp_path / "initial/rgb.npz", allow_pickle=False) as saved:
        assert set(saved.files) == set(images)
        assert np.array_equal(saved["head_camera"], rgb)


def test_renderer_binding_bypasses_wrapper_that_drops_device(monkeypatch):
    import sys
    from types import ModuleType
    from benchmarks.robotwin.r2_scene_probe import bind_renderer

    calls = []

    class Native:
        def __init__(self, device=None):
            calls.append(device)

    class Legacy(Native):
        def __init__(self, **kwargs):
            super().__init__()

    device = SimpleNamespace(cuda_id=0, can_render=lambda: True, pci_string="leased", name="test")
    sapien = ModuleType("sapien")
    core = ModuleType("sapien.core")
    core.Engine = type("Engine", (), {})
    core.SapienRenderer = Legacy
    sapien.Device = lambda name: device
    sapien.pysapien = SimpleNamespace(render=SimpleNamespace(SapienRenderer=Native))
    sapien.core = core
    scene = ModuleType("sapien.wrapper.scene")
    scene.Scene = object
    monkeypatch.setitem(sys.modules, "sapien", sapien)
    monkeypatch.setitem(sys.modules, "sapien.core", core)
    monkeypatch.setitem(sys.modules, "sapien.wrapper.scene", scene)
    bind_renderer()
    core.SapienRenderer()
    assert calls == [device]
