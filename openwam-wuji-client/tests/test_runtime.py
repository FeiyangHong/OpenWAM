import numpy as np
import pytest
from openwam_robot_client.geometry import pose7_to_eef9_column
from openwam_robot_client.robot import RobotController
from openwam_robot_client.runtime import _validated_actions


def test_action_chunk_validation():
    value = _validated_actions({"actions": np.zeros((4, 58)).tolist()}, 4)
    assert value.shape == (4, 58)


def test_action_chunk_rejects_wrong_width():
    with pytest.raises(ValueError, match="shape"):
        _validated_actions({"actions": np.zeros((4, 80)).tolist()}, 4)


def test_hand_commands_are_independent_from_arm_enable():
    pose = np.array([0.4, 0.2, 0.9, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    class Robot:
        arm_left_name = "left"
        arm_right_name = "right"

        def get_current_cartesian_pose(self, names, frame):
            return [pose.copy(), pose.copy()]

        def set_cartesian_pose(self, *args, **kwargs):
            raise AssertionError("arm command must remain disabled")

    class State:
        def get(self):
            return np.zeros(20, dtype=np.float32)

    class Hands:
        left = State()
        right = State()

        def __init__(self):
            self.sent = []

        def send(self, side, joints):
            self.sent.append((side, joints.copy()))

    hands = Hands()
    controller = RobotController(
        Robot(),
        hands,
        enable_arms=False,
        eef_frame="chassis",
        control_way="filter",
        use_wbc=False,
        workspace_radius=np.ones(3),
        max_eef_step=0.03,
        max_rotation_step_rad=0.1,
    )
    action = np.concatenate(
        (pose7_to_eef9_column(pose), np.zeros(20), pose7_to_eef9_column(pose), np.ones(20))
    )
    controller.execute(action)
    assert [side for side, _ in hands.sent] == ["left", "right"]
