"""Opt-in, device-specific fallback for a Wuji Hand 1 with an offline left thumb."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from .hands import JOINT_COUNT, JointState, WujiHands


logger = logging.getLogger(__name__)

LEFT_USB_SERIAL = "336536623434"
LEFT_PRODUCT_SERIAL = "LQSQJL.260507.003"
RIGHT_USB_SERIAL = "387E36643533"
THUMB_JOINTS = 4
STATE_MAX_AGE = 0.5


class _FreshJointState(JointState):
    def __init__(self) -> None:
        super().__init__()
        self._updated_at = 0.0
        self._error: BaseException | None = None

    def update(self, value: Any) -> None:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape != (JOINT_COUNT,) or not np.isfinite(array).all():
            raise ValueError(f"expected {JOINT_COUNT} finite hand joints, got {array.shape}")
        with self._lock:
            self._value = array.copy()
            self._updated_at = time.monotonic()

    def fail(self, error: BaseException) -> None:
        with self._lock:
            self._error = error

    def raise_if_failed(self) -> None:
        with self._lock:
            if self._error is not None:
                raise RuntimeError("emergency left hand state failed") from self._error

    def get(self) -> np.ndarray:
        with self._lock:
            if self._error is not None:
                raise RuntimeError("emergency left hand state failed") from self._error
            if self._value is None:
                raise RuntimeError("emergency left hand state is not ready")
            if time.monotonic() - self._updated_at > STATE_MAX_AGE:
                raise TimeoutError("emergency left hand state is stale")
            return self._value.copy()

    def replace_thumb(self, thumb: np.ndarray) -> None:
        """Replace only virtual joints without refreshing real-state freshness."""
        with self._lock:
            if self._value is None or self._error is not None:
                raise RuntimeError("emergency left hand state is not ready")
            self._value[:THUMB_JOINTS] = thumb


class EmergencyWujiHands(WujiHands):
    """Keep the right SDK path and replace only the disconnected left thumb path."""

    def __init__(
        self,
        *,
        thumb_state: Any = (0.0, 0.0, 0.0, 0.0),
        thumb_state_mode: str = "fixed",
        **kwargs: Any,
    ) -> None:
        synthetic = np.asarray(thumb_state, dtype=np.float32)
        if synthetic.shape != (THUMB_JOINTS,) or not np.isfinite(synthetic).all():
            raise ValueError("thumb_state must contain four finite joint angles")
        if np.any(np.abs(synthetic) > 1.57):
            raise ValueError("thumb_state entries must be within [-1.57, 1.57] radians")
        if thumb_state_mode not in ("fixed", "action"):
            raise ValueError("thumb_state_mode must be fixed or action")
        self._thumb_state_mode = thumb_state_mode
        self._synthetic_thumb = synthetic.copy()
        self._left_lock = threading.RLock()
        self._left_hand: Any | None = None
        self._left_controller: Any | None = None
        self._left_enabled = False
        requested_enable = bool(kwargs.pop("enable_commands", False))
        # Connect both hands and obtain live state before enabling either one.
        super().__init__(enable_commands=False, **kwargs)
        if requested_enable:
            self.enable_commands = True
            try:
                self._enable_hands()
            except BaseException:
                self.close()
                raise

    def _connect(self, model, left_serial, right_serial, effort_limit, lowpass_hz, mit_kp, mit_kd):
        if model not in ("auto", "wuji_hand"):
            raise ValueError("--ignore-left-thumb supports Wuji Hand 1 only")
        if left_serial and left_serial != LEFT_USB_SERIAL:
            raise ValueError(
                f"emergency left USB serial must be {LEFT_USB_SERIAL}, got {left_serial}"
            )
        if right_serial and right_serial != RIGHT_USB_SERIAL:
            raise ValueError(
                f"emergency right USB serial must be {RIGHT_USB_SERIAL}, got {right_serial}"
            )
        # The normal SDK must never try to initialize the left hand here.
        super()._connect(
            model,
            left_serial,
            RIGHT_USB_SERIAL,
            effort_limit,
            lowpass_hz,
            mit_kp,
            mit_kd,
            skip_serials=frozenset((LEFT_USB_SERIAL,)),
            only_serials=frozenset((RIGHT_USB_SERIAL,)),
            required_sides=frozenset(("right",)),
        )
        try:
            import wujihandpy
        except ImportError as exc:
            raise RuntimeError(
                "--ignore-left-thumb requires the patched wujihandpy wheel; "
                "see docs/left_thumb_emergency.zh-CN.md"
            ) from exc

        mask = np.zeros((5, 4), dtype=np.bool_)
        mask[0, :] = True
        hand = wujihandpy.Hand(serial_number=LEFT_USB_SERIAL, mask=mask)
        self._left_hand = hand
        product_serial = hand.get_product_sn()
        if product_serial != LEFT_PRODUCT_SERIAL or int(hand.read_handedness()) != 1:
            raise RuntimeError(
                f"emergency left hand identity mismatch: product={product_serial!r}"
            )
        # The native API is single-threaded unless all calls share an external lock.
        hand.disable_thread_safe_check()
        voltage = np.asarray(hand.read_joint_bus_voltage(), dtype=np.float32)
        if voltage.shape != (5, 4) or not np.isfinite(voltage[1:, :]).all() or np.any(voltage[1:, :] <= 1.0):
            raise RuntimeError("one or more unmasked left-hand joints are not online")
        self.left = _FreshJointState()
        self._left_controller = hand.realtime_controller(
            True, wujihandpy.filter.LowPass(cutoff_freq=lowpass_hz)
        )
        self._left_effort_limit = effort_limit
        thread = threading.Thread(target=self._poll_left, name="wuji_left_emergency_state", daemon=True)
        self._threads.append(thread)
        thread.start()
        logger.warning(
            "LEFT THUMB EMERGENCY MODE: %s joints 0..3 masked; synthetic state=%s",
            LEFT_PRODUCT_SERIAL,
            self._synthetic_thumb.tolist(),
        )

    def _poll_left(self) -> None:
        while not self._stop.is_set():
            try:
                with self._left_lock:
                    positions = np.asarray(self._left_hand.read_joint_actual_position(), dtype=np.float32)
                    if positions.shape != (5, 4) or not np.isfinite(positions[1:, :]).all():
                        raise ValueError("left hand returned invalid state for an unmasked joint")
                    positions[0, :] = self._synthetic_thumb
                    self.left.update(positions.reshape(JOINT_COUNT))
            except Exception as exc:
                self.left.fail(exc)
                logger.error("emergency left hand state stopped: %s", exc)
                return
            self._stop.wait(1.0 / 30.0)

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.left.raise_if_failed()
            try:
                self.left.get()
                self.right.get()
                return
            except RuntimeError:
                time.sleep(0.02)
        self.left.raise_if_failed()
        raise TimeoutError("timed out waiting for Wuji hand state")

    def _enable_hands(self) -> None:
        actual = self.left.get().astype(np.float64).reshape(5, 4)
        enabled = np.ones((5, 4), dtype=np.bool_)
        enabled[0, :] = False
        with self._left_lock:
            self._left_hand.write_joint_effort_limit(float(self._left_effort_limit))
            # Keep the healthy joints at their measured pose when they are enabled.
            self._left_controller.set_joint_target_position(actual)
            self._left_enabled = True
            self._left_hand.write_joint_enabled(enabled)
        self._connections["right"].hand.enable()

    def check_health(self) -> None:
        self.left.get()

    def send(self, side: str, joints: np.ndarray) -> None:
        if side != "left":
            super().send(side, joints)
            return
        if not self.enable_commands and self._thumb_state_mode == "fixed":
            return
        actual = self.left.get().astype(np.float64)
        target = np.asarray(joints, dtype=np.float64)
        if target.shape != (JOINT_COUNT,) or not np.isfinite(target).all():
            raise ValueError("left hand command must be finite with shape (20,)")
        target = np.clip(target, -1.57, 1.57)
        if self.max_step > 0.0:
            target[THUMB_JOINTS:] = actual[THUMB_JOINTS:] + np.clip(
                target[THUMB_JOINTS:] - actual[THUMB_JOINTS:],
                -self.max_step,
                self.max_step,
            )
        with self._left_lock:
            previous_thumb = self._synthetic_thumb.copy()
            if self._thumb_state_mode == "action":
                next_thumb = target[:THUMB_JOINTS].copy()
                if self.max_step > 0.0:
                    next_thumb = previous_thumb + np.clip(
                        next_thumb - previous_thumb, -self.max_step, self.max_step
                    )
            target[:THUMB_JOINTS] = previous_thumb
            if self.enable_commands:
                self._left_controller.set_joint_target_position(target.reshape(5, 4))
            if self._thumb_state_mode == "action":
                self._synthetic_thumb = next_thumb.astype(np.float32)
                self.left.replace_thumb(self._synthetic_thumb)

    def close(self) -> None:
        self._stop.set()
        if self._left_hand is not None and self._left_enabled:
            try:
                with self._left_lock:
                    self._left_hand.write_joint_enabled(False)
            except Exception:
                logger.exception("failed to disable emergency left hand")
            self._left_enabled = False
        try:
            super().close()
        finally:
            if self._left_controller is not None:
                try:
                    with self._left_lock:
                        self._left_controller.close()
                except Exception:
                    logger.exception("failed to close emergency left controller")
                self._left_controller = None
            self._left_hand = None
