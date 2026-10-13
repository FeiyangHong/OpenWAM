"""Fixed virtual left state, right-hand-only connection and right-arm-only targets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .emergency_left_thumb import RIGHT_USB_SERIAL
from .geometry import pose7_to_eef9_column
from .rtc_controller import RTCController, RTCWujiHands

DEFAULT_INIT_FILE = Path(__file__).resolve().parent.parent / 'config' / 'rtc_fixed_left_init.json'


def read_fixed_left_init(path=DEFAULT_INIT_FILE):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    values = []
    for key, size in [('torso_joints', 4), ('left_arm_joints', 7), ('left_hand_joints', 20)]:
        value = np.asarray(data[key], dtype=np.float32)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f'{key} must contain {size} finite values')
        if key == 'left_hand_joints' and np.any(np.abs(value) > 1.57):
            raise ValueError('left_hand_joints must be within [-1.57, 1.57] radians')
        values.append(value)
    return tuple(values)


class FixedJointState:
    def __init__(self, value):
        self._value = np.asarray(value, dtype=np.float32).copy()
        if self._value.shape != (20,) or not np.isfinite(self._value).all() or np.any(np.abs(self._value) > 1.57):
            raise ValueError('fixed left hand must be finite 20-D radians within [-1.57, 1.57]')
        self._value.setflags(write=False)

    def get(self):
        return self._value.copy()


class FixedLeftHands(RTCWujiHands):
    def __init__(self, *, fixed_left_hand, shadow_diagnostics=None, **kwargs):
        self.shadow_diagnostics = shadow_diagnostics
        self.fixed_left = FixedJointState(fixed_left_hand)
        serial = kwargs.get('right_serial') or RIGHT_USB_SERIAL
        # This isolated deployment only opens the physically identified right device.
        if serial != RIGHT_USB_SERIAL:
            raise ValueError(f'Fixed-left client requires verified right USB serial {RIGHT_USB_SERIAL}')
        kwargs['right_serial'] = serial
        super().__init__(**kwargs)

    def _poll_hand1(self, connection):
        if self.shadow_diagnostics is None:
            return super()._poll_hand1(connection)
        from .rtc_shadow_capture import RecordingSubscription
        connection.subscription = RecordingSubscription(connection.subscription, self.shadow_diagnostics)
        return super()._poll_hand1(connection)

    def _poll_hand2(self, connection):
        if self.shadow_diagnostics is not None:
            self.shadow_diagnostics.record('shadow_hand_unsupported', reason='Hand2 partial-joint frames require per-joint timing')
        return super()._poll_hand2(connection)

    def _connect(self, model, left_serial, right_serial, effort_limit, lowpass_hz, mit_kp, mit_kd):
        self.left = self.fixed_left
        # Parent filters scan results by serial BEFORE connect(), limits or controllers.
        super()._connect(model, '', right_serial, effort_limit, lowpass_hz, mit_kp, mit_kd,
                         only_serials=frozenset((right_serial,)), required_sides=frozenset(('right',)))

    def project_prepared_target(self, side, target):
        return self.left.get() if side == 'left' else target

    def validate_prepared(self, side, target):
        if side == 'left':
            value = np.asarray(target, dtype=np.float32)
            if value.shape != (20,) or not np.array_equal(value, self.left.get()):
                raise ValueError('Left hand target must equal its fixed virtual state')
            return
        return super().validate_prepared(side, target)

    def send_prepared(self, side, target):
        if side == 'left':
            self.validate_prepared(side, target)
            return
        return super().send_prepared(side, target)

    def send(self, side, joints):
        if side == 'left':
            self.validate_prepared(side, joints)
            return
        return super().send(side, joints)


class FixedLeftController(RTCController):
    def hold_left_init(self, pose, hand):
        super().hold_left_init(pose, hand)
        # Seed planning with the same virtual values used in observations/prefixes.
        self.previous = (self.left_init_pose.copy(), self.previous[1])
        self.previous_hands = (self.left_init_hand.copy(), self.previous_hands[1])

    def state_vector(self):
        if self.left_init_pose is None:
            raise RuntimeError('Fixed-left initial pose has not been validated')
        state = super().state_vector()  # Retain SDK reads and normal right-hand feedback.
        state[:9] = pose7_to_eef9_column(self.left_init_pose).tolist()
        state[9:29] = self.left_init_hand.tolist()
        return state

    def arm_command(self, action):
        if self.use_wbc:
            raise ValueError('Fixed-left client does not permit whole-body control')
        if self.left_init_pose is None or not np.allclose(action.left, self.left_init_pose, atol=1e-6, rtol=0):
            raise ValueError('Prepared left arm target differs from fixed initial pose')
        return [self.robot.arm_right_name], [action.right.tolist()], {'add_default_torso': False}
