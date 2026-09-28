from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import clip_pose_step, eef9_column_to_pose7, pose7_to_eef9_column

RAW_ACTION_DIM = 58


@dataclass(frozen=True)
class Action58:
    left_eef: np.ndarray
    left_hand: np.ndarray
    right_eef: np.ndarray
    right_hand: np.ndarray

    @classmethod
    def parse(cls, value: Any) -> "Action58":
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (RAW_ACTION_DIM,) or not np.isfinite(array).all():
            raise ValueError(f"action must be finite with shape (58,), got {array.shape}")
        return cls(array[:9], array[9:29], array[29:38], array[38:58])


def _pair(values: Any, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(values, dict):
        values = [values[name] for name in names]
    if len(values) < 2:
        raise ValueError("Astribot did not return both arm poses")
    result = tuple(np.asarray(value, dtype=np.float32) for value in values[:2])
    if any(value.shape != (7,) or not np.isfinite(value).all() for value in result):
        raise ValueError("Astribot arm poses must be two finite pose7 values")
    return result


class RobotController:
    def __init__(
        self,
        robot: Any,
        hands: Any,
        *,
        enable_arms: bool,
        eef_frame: str,
        control_way: str,
        use_wbc: bool,
        workspace_radius: np.ndarray,
        max_eef_step: float,
        max_rotation_step_rad: float,
    ):
        self.robot = robot
        self.hands = hands
        self.enable_arms = enable_arms
        self.eef_frame = eef_frame
        self.control_way = control_way
        self.use_wbc = use_wbc
        self.arm_names = [robot.arm_left_name, robot.arm_right_name]
        self.previous = self.read_arms()
        radius = np.asarray(workspace_radius, dtype=np.float32)
        self.workspace_min = (self.previous[0][:3] - radius, self.previous[1][:3] - radius)
        self.workspace_max = (self.previous[0][:3] + radius, self.previous[1][:3] + radius)
        self.max_eef_step = max_eef_step
        self.max_rotation_step_rad = max_rotation_step_rad

    def read_arms(self) -> tuple[np.ndarray, np.ndarray]:
        return _pair(self.robot.get_current_cartesian_pose(self.arm_names, frame=self.eef_frame), self.arm_names)

    def state_vector(self) -> list[float]:
        left_pose, right_pose = self.read_arms()
        state = np.concatenate(
            (
                pose7_to_eef9_column(left_pose),
                self.hands.left.get(),
                pose7_to_eef9_column(right_pose),
                self.hands.right.get(),
            )
        ).astype(np.float32)
        if state.shape != (RAW_ACTION_DIM,):
            raise RuntimeError(f"built invalid state shape: {state.shape}")
        return state.tolist()

    def _project_pose(self, target: np.ndarray, side: int) -> np.ndarray:
        target = clip_pose_step(target, self.previous[side], self.max_eef_step, self.max_rotation_step_rad)
        target[:3] = np.clip(target[:3], self.workspace_min[side], self.workspace_max[side])
        return target

    def execute(self, raw_action: Any) -> None:
        action = Action58.parse(raw_action)
        left = self._project_pose(eef9_column_to_pose7(action.left_eef), 0)
        right = self._project_pose(eef9_column_to_pose7(action.right_eef), 1)
        if self.enable_arms:
            poses = [left.tolist(), right.tolist()]
            try:
                self.robot.set_cartesian_pose(
                    self.arm_names,
                    poses,
                    control_way=self.control_way,
                    use_wbc=self.use_wbc,
                    remote_wbc_option=self.use_wbc,
                )
            except TypeError:
                self.robot.set_cartesian_pose(
                    self.arm_names, poses, control_way=self.control_way, use_wbc=self.use_wbc
                )
        # WujiHands owns its independent enable_commands gate.
        self.hands.send("left", action.left_hand)
        self.hands.send("right", action.right_hand)
        self.previous = (left, right)
