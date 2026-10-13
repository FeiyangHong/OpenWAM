"""Prepare immutable SDK targets for RTC, preserving live tracking guards.

The synchronous RobotController/WujiHands paths are unchanged. Planned hand
targets are never silently clipped against a newer measurement: if the live
guard would alter a committed target, RTC stops and the pending chunk is stale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import clip_pose_step, eef9_column_to_pose7, pose7_to_eef9_column
from .robot import Action58, RobotController
from .hands import WujiHands


@dataclass(frozen=True)
class PreparedAction:
    raw: np.ndarray
    left: np.ndarray
    right: np.ndarray


class RTCWujiHands(WujiHands):
    def validate_prepared(self, side, target):
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (20,) or not np.isfinite(target).all() or np.any(np.abs(target) > 1.57 + 1e-6):
            raise ValueError("Invalid prepared Wuji command")
        if self.enable_commands and self.max_step > 0:
            actual = self._connections[side].state.get()
            if np.any(np.abs(target - actual) > self.max_step + 1e-6):
                raise RuntimeError("RTC hand tracking guard: committed target would require re-clipping; stopping")

    def send_prepared(self, side, target):
        self.validate_prepared(side, target)
        if self.enable_commands:
            from wuji_sdk import JointCommand
            self._connections[side].publisher.send([JointCommand(float(v), 0., 0.) for v in target])


class RTCController(RobotController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_hands = (self.hands.left.get(), self.hands.right.get())

    def prepare(self, actions):
        """Project once, without publishing or changing committed controller state."""
        previous = tuple(p.copy() for p in self.previous)
        hands = tuple(p.copy() for p in self.previous_hands)
        planned = []
        for raw in actions:
            a = Action58.parse(raw)
            poses = []
            targets = []
            for side, (eef, hand) in enumerate(((a.left_eef, a.left_hand), (a.right_eef, a.right_hand))):
                pose = clip_pose_step(eef9_column_to_pose7(eef), previous[side],
                                      self.max_eef_step, self.max_rotation_step_rad)
                pose[:3] = np.clip(pose[:3], self.workspace_min[side], self.workspace_max[side])
                target = np.clip(hand, -1.57, 1.57)
                if self.hands.max_step > 0:
                    target = hands[side] + np.clip(target - hands[side], -self.hands.max_step, self.hands.max_step)
                poses.append(pose)
                targets.append(target)
            value = np.concatenate((pose7_to_eef9_column(poses[0]), targets[0],
                                    pose7_to_eef9_column(poses[1]), targets[1])).astype(np.float32)
            for array in (value, *poses):
                array.setflags(write=False)
            planned.append(PreparedAction(value, *poses))
            previous, hands = poses, targets
        return planned

    def execute_prepared(self, action):
        a = Action58.parse(action.raw)
        # Check both hands before publishing either arm or hand command.
        self.hands.validate_prepared("left", a.left_hand)
        self.hands.validate_prepared("right", a.right_hand)
        if self.enable_arms:
            poses = [action.left.tolist(), action.right.tolist()]
            try:
                self.robot.set_cartesian_pose(self.arm_names, poses, control_way=self.control_way,
                                               use_wbc=self.use_wbc, remote_wbc_option=self.use_wbc)
            except TypeError:
                self.robot.set_cartesian_pose(self.arm_names, poses, control_way=self.control_way, use_wbc=self.use_wbc)
        self.hands.send_prepared("left", a.left_hand)
        self.hands.send_prepared("right", a.right_hand)
        self.previous = (action.left.copy(), action.right.copy())
        self.previous_hands = (a.left_hand.copy(), a.right_hand.copy())
