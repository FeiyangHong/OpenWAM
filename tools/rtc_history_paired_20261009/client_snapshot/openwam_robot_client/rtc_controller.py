"""Prepare immutable SDK targets for RTC, with a separate live tracking guard.

The synchronous RobotController/WujiHands paths are unchanged. RTC limits each
committed command step and checks actuator lag without altering the prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .geometry import clip_pose_step, eef9_column_to_pose7, pose7_to_eef9_column
from .robot import Action58, RobotController
from .hands import WujiHands
from .emergency_left_thumb import EmergencyWujiHands, THUMB_JOINTS


@dataclass(frozen=True)
class PreparedAction:
    raw: np.ndarray
    left: np.ndarray
    right: np.ndarray


def _check_tracking(hands, side, target, actual, start=0):
    if not hands.enable_commands:
        return
    error = np.abs(target[start:] - actual[start:])
    index = int(np.argmax(error)) + start
    if error[index - start] > hands.tracking_tolerance + 1e-6:
        raise RuntimeError(
            f"RTC hand tracking guard: {side} joint {index} target={target[index]:.3f} "
            f"actual={actual[index]:.3f} error={error[index - start]:.3f} "
            f"exceeds tolerance={hands.tracking_tolerance:.3f}; stopping"
        )


class RTCWujiHands(WujiHands):
    tracking_tolerance = 0.5

    def validate_prepared(self, side, target):
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (20,) or not np.isfinite(target).all() or np.any(np.abs(target) > 1.57 + 1e-6):
            raise ValueError("Invalid prepared Wuji command")
        if self.enable_commands:
            actual = self._connections[side].state.get()
            _check_tracking(self, side, target, actual)

    def send_prepared(self, side, target):
        self.validate_prepared(side, target)
        if self.enable_commands:
            from wuji_sdk import JointCommand
            self._connections[side].publisher.send([JointCommand(float(v), 0., 0.) for v in target])


class RTCEmergencyWujiHands(EmergencyWujiHands):
    """RTC's immutable-target path with only the offline left thumb masked."""

    tracking_tolerance = 0.5

    def project_prepared_target(self, side, target):
        target = np.asarray(target, dtype=np.float32).copy()
        if side == "left" and self._thumb_state_mode == "fixed":
            target[:THUMB_JOINTS] = self._synthetic_thumb
        return target

    def validate_prepared(self, side, target):
        if side != "left":
            return RTCWujiHands.validate_prepared(self, side, target)
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (20,) or not np.isfinite(target).all() or np.any(np.abs(target) > 1.57 + 1e-6):
            raise ValueError("Invalid prepared Wuji command")
        actual = self.left.get()  # Also checks that the 16 measured joints are fresh.
        if self._thumb_state_mode == "fixed":
            if np.any(np.abs(target[:THUMB_JOINTS] - self._synthetic_thumb) > 1e-6):
                raise RuntimeError("RTC left thumb target differs from the fixed virtual state")
        elif self.max_step > 0 and np.any(
            np.abs(target[:THUMB_JOINTS] - self._synthetic_thumb) > self.max_step + 1e-6
        ):
            raise RuntimeError("RTC left thumb virtual target exceeds the committed step limit")
        _check_tracking(self, side, target, actual, start=THUMB_JOINTS)

    def send_prepared(self, side, target):
        if side != "left":
            return RTCWujiHands.send_prepared(self, side, target)
        self.validate_prepared(side, target)
        target = np.asarray(target, dtype=np.float64).copy()
        with self._left_lock:
            next_thumb = target[:THUMB_JOINTS].astype(np.float32).copy()
            target[:THUMB_JOINTS] = self._synthetic_thumb
            if self.enable_commands:
                self._left_controller.set_joint_target_position(target.reshape(5, 4))
            if self._thumb_state_mode == "action":
                self._synthetic_thumb = next_thumb
                self.left.replace_thumb(next_thumb)


class RTCController(RobotController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_hands = (self.hands.left.get(), self.hands.right.get())

    def read_arms(self):
        if not getattr(self, "diagnostics", None):
            return super().read_arms()
        start = time.monotonic_ns()
        poses = super().read_arms()
        end = time.monotonic_ns()
        self.diagnostics_arm_read = dict(start_monotonic_ns=start, end_monotonic_ns=end,
                                         left_pose7=poses[0].tolist(), right_pose7=poses[1].tolist(),
                                         measurement_timestamp_available=False)
        return poses

    def align_start_hands(self):
        """Anchor the first planned chunk after warmup/inference, before any command."""
        self.previous_hands = (self.hands.left.get(), self.hands.right.get())

    def prepare(self, actions):
        """Project once, without publishing or changing committed controller state."""
        return self.prepare_from(actions, self.previous, self.previous_hands)

    def prepare_from(self, actions, seed_poses, seed_hands):
        """Plan against the immutable endpoint of a committed RTC prefix."""
        previous = tuple(p.copy() for p in seed_poses)
        hands = tuple(p.copy() for p in seed_hands)
        planned = []
        for raw in actions:
            a = Action58.parse(raw)
            poses = []
            targets = []
            for side, (eef, hand) in enumerate(((a.left_eef, a.left_hand), (a.right_eef, a.right_hand))):
                if side == 0 and self.left_init_pose is not None:
                    eef_target, hand = self.left_init_pose, self.left_init_hand
                else:
                    eef_target = eef9_column_to_pose7(eef)
                pose = clip_pose_step(eef_target, previous[side],
                                      self.max_eef_step, self.max_rotation_step_rad)
                pose[:3] = np.clip(pose[:3], self.workspace_min[side], self.workspace_max[side])
                target = np.clip(hand, -1.57, 1.57)
                if self.hands.max_step > 0:
                    target = hands[side] + np.clip(target - hands[side], -self.hands.max_step, self.hands.max_step)
                project_target = getattr(self.hands, "project_prepared_target", None)
                if project_target is not None:
                    target = project_target("left" if side == 0 else "right", target)
                poses.append(pose)
                targets.append(target)
            value = np.concatenate((pose7_to_eef9_column(poses[0]), targets[0],
                                    pose7_to_eef9_column(poses[1]), targets[1])).astype(np.float32)
            for array in (value, *poses):
                array.setflags(write=False)
            planned.append(PreparedAction(value, *poses))
            previous, hands = poses, targets
        return planned

    def arm_command(self, action):
        """SDK targets for this controller; default retains the dual-arm path."""
        return self.arm_names, [action.left.tolist(), action.right.tolist()], {}

    def execute_prepared(self, action):
        a = Action58.parse(action.raw)
        diagnostics = getattr(self, "diagnostics", None)
        timings = {}
        arm_call_attempts = []
        phase = "validate_left"
        phase_started = time.monotonic()
        # Check both hands before publishing either arm or hand command.
        try:
            self.hands.validate_prepared("left", a.left_hand)
            if diagnostics:
                timings[phase + "_ms"] = (time.monotonic() - phase_started) * 1000
            phase, phase_started = "validate_right", time.monotonic()
            self.hands.validate_prepared("right", a.right_hand)
            if diagnostics:
                timings[phase + "_ms"] = (time.monotonic() - phase_started) * 1000
            phase, phase_started = "arm_send", time.monotonic()
            if self.enable_arms:
                names, poses, arm_options = self.arm_command(action)
                try:
                    call_start = time.monotonic_ns() if diagnostics else None
                    try:
                        self.robot.set_cartesian_pose(names, poses, control_way=self.control_way,
                                                       use_wbc=self.use_wbc, remote_wbc_option=self.use_wbc,
                                                       **arm_options)
                    finally:
                        if diagnostics:
                            arm_call_attempts.append(dict(start_monotonic_ns=call_start,
                                                          end_monotonic_ns=time.monotonic_ns()))
                except TypeError:
                    call_start = time.monotonic_ns() if diagnostics else None
                    try:
                        self.robot.set_cartesian_pose(names, poses, control_way=self.control_way,
                                                       use_wbc=self.use_wbc, **arm_options)
                    finally:
                        if diagnostics:
                            arm_call_attempts.append(dict(start_monotonic_ns=call_start,
                                                          end_monotonic_ns=time.monotonic_ns()))
            if diagnostics:
                timings[phase + "_ms"] = (time.monotonic() - phase_started) * 1000
            phase, phase_started = "left_hand_send", time.monotonic()
            self.hands.send_prepared("left", a.left_hand)
            if diagnostics:
                timings[phase + "_ms"] = (time.monotonic() - phase_started) * 1000
            phase, phase_started = "right_hand_send", time.monotonic()
            self.hands.send_prepared("right", a.right_hand)
            if diagnostics:
                timings[phase + "_ms"] = (time.monotonic() - phase_started) * 1000
                published_at = time.monotonic()
                left_delta = action.left[:3] - self.previous[0][:3]
                right_delta = action.right[:3] - self.previous[1][:3]
                left_hand_delta = a.left_hand - self.previous_hands[0]
                right_hand_delta = a.right_hand - self.previous_hands[1]
                prior = getattr(self, "_diag_prior_delta", None)
                context = getattr(self, "diagnostics_context", (None, None, False))
                last_published = getattr(self, "_diag_last_published", None)
                def rotation_step(left, right):
                    dot = float(np.clip(abs(np.dot(left[3:], right[3:])), 0., 1.))
                    return math.degrees(2 * math.acos(dot))
                diagnostics.record(
                    "command", step=context[0], chunk=context[1], boundary=context[2],
                    scheduled_deadline_monotonic_ns=getattr(self, "diagnostics_deadline_ns", None),
                    arms_enabled=self.enable_arms, arm_call_attempts=arm_call_attempts,
                    interval_ms=None if last_published is None else (published_at - last_published) * 1000,
                    left_xyz_step_m=float(np.linalg.norm(left_delta)),
                    right_xyz_step_m=float(np.linalg.norm(right_delta)),
                    left_rotation_step_deg=rotation_step(action.left, self.previous[0]),
                    right_rotation_step_deg=rotation_step(action.right, self.previous[1]),
                    left_hand_step_max_rad=float(np.max(np.abs(left_hand_delta))),
                    right_hand_step_max_rad=float(np.max(np.abs(right_hand_delta))),
                    left_xyz_delta_change_m=None if prior is None else float(np.linalg.norm(left_delta - prior[0])),
                    right_xyz_delta_change_m=None if prior is None else float(np.linalg.norm(right_delta - prior[1])),
                    left_hand_delta_change_max_rad=None if prior is None else float(np.max(np.abs(left_hand_delta - prior[2]))),
                    right_hand_delta_change_max_rad=None if prior is None else float(np.max(np.abs(right_hand_delta - prior[3]))),
                    left_pose7=action.left.tolist(), right_pose7=action.right.tolist(),
                    left_hand20=a.left_hand.tolist(), right_hand20=a.right_hand.tolist(),
                    **timings,
                )
                self._diag_prior_delta = (left_delta, right_delta, left_hand_delta, right_hand_delta)
                self._diag_last_published = published_at
            self.previous = (action.left.copy(), action.right.copy())
            self.previous_hands = (a.left_hand.copy(), a.right_hand.copy())
        except BaseException as exc:
            if diagnostics:
                diagnostics.record("sdk_error", phase=phase,
                                   ms=(time.monotonic() - phase_started) * 1000,
                                   error=repr(exc), arm_call_attempts=arm_call_attempts,
                                   context=getattr(self, "diagnostics_context", None))
            raise
