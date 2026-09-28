from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

JOINT_COUNT = 20


class JointState:
    def __init__(self) -> None:
        self._value: np.ndarray | None = None
        self._lock = threading.Lock()

    def update(self, value: Any) -> None:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape != (JOINT_COUNT,) or not np.isfinite(array).all():
            raise ValueError(f"expected {JOINT_COUNT} finite hand joints, got {array.shape}")
        with self._lock:
            self._value = array.copy()

    def get(self) -> np.ndarray:
        with self._lock:
            if self._value is None:
                raise RuntimeError("Wuji hand state is not ready")
            return self._value.copy()


@dataclass
class _HandConnection:
    hand: Any
    publisher: Any
    subscription: Any
    state: JointState
    controller_context: Any | None


def _side_name(value: Any) -> str:
    value = str(value or "").lower()
    if value == "left" or value.endswith(".left"):
        return "left"
    if value == "right" or value.endswith(".right"):
        return "right"
    return ""


class WujiHands:
    """Small direct-SDK boundary; no ROS hand topics are created."""

    def __init__(
        self,
        *,
        model: str = "auto",
        left_serial: str = "",
        right_serial: str = "",
        enable_commands: bool = False,
        effort_limit: float = 1.5,
        lowpass_hz: float = 5.0,
        mit_kp: float = 3.0,
        mit_kd: float = 0.05,
        max_step: float = 0.1,
        timeout: float = 5.0,
    ):
        self.enable_commands = enable_commands
        self.max_step = max_step
        self.left = JointState()
        self.right = JointState()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._connections: dict[str, _HandConnection] = {}
        self._manager = None
        try:
            self._connect(model, left_serial, right_serial, effort_limit, lowpass_hz, mit_kp, mit_kd)
            self.wait_ready(timeout)
        except BaseException:
            self.close()
            raise

    def _connect(self, model, left_serial, right_serial, effort_limit, lowpass_hz, mit_kp, mit_kd):
        from wuji_sdk import DeviceType, LowPass, SdkManager, WujiHand, WujiHand2

        allowed = {
            "auto": (DeviceType.WujiHand, DeviceType.WujiHand2),
            "wuji_hand": (DeviceType.WujiHand,),
            "wuji_hand2": (DeviceType.WujiHand2,),
        }
        if model not in allowed:
            raise ValueError(f"unsupported Wuji model: {model}")
        serials = {"left": left_serial, "right": right_serial}
        self._manager = SdkManager.instance()
        devices = [device for device in self._manager.scan() if device.device_type in allowed[model]]
        for index, device in enumerate(devices):
            hand = self._manager.connect(sn=device.sn, device_name=f"openwam_{index}")
            if isinstance(hand, WujiHand2):
                side = _side_name(hand.handedness().get())
                hand.effort_limit().set(effort_limit)
                hand.mit_params().set((mit_kp, mit_kd))
                context = None
                is_hand2 = True
            elif isinstance(hand, WujiHand):
                side = _side_name(hand.handedness_name())
                hand.set_all_effort_limit(effort_limit)
                context = hand.realtime_controller(LowPass(cutoff_hz=lowpass_hz))
                context.__enter__()
                is_hand2 = False
            else:
                hand.disconnect()
                continue
            if side not in serials or side in self._connections or (serials[side] and serials[side] != device.sn):
                hand.disconnect()
                continue
            state = self.left if side == "left" else self.right
            publisher = hand.joint_command().publish()
            subscription = hand.joint_states().subscribe()
            connection = _HandConnection(hand, publisher, subscription, state, context)
            self._connections[side] = connection
            target = self._poll_hand2 if is_hand2 else self._poll_hand1
            thread = threading.Thread(target=target, args=(connection,), name=f"wuji_{side}_state", daemon=True)
            self._threads.append(thread)
            thread.start()
            if self.enable_commands:
                hand.enable()
        missing = sorted({"left", "right"} - set(self._connections))
        if missing:
            raise RuntimeError(f"missing Wuji hand devices: {missing}")

    def _poll_hand1(self, connection: _HandConnection) -> None:
        while not self._stop.is_set():
            value = connection.subscription.recv()
            if value is None:
                time.sleep(0.001)
            else:
                connection.state.update(value.position)

    def _poll_hand2(self, connection: _HandConnection) -> None:
        from wuji_sdk import WujiHand2

        index_by_nid = {
            int(WujiHand2.joint_id_from_bus_node(bus_id, node_id)) + 1: bus_id * 4 + node_id - 1
            for bus_id in range(5)
            for node_id in range(1, 5)
        }
        positions = np.full(JOINT_COUNT, np.nan, dtype=np.float32)
        while not self._stop.is_set():
            frame = connection.subscription.recv()
            if frame is None:
                time.sleep(0.001)
                continue
            for entry in frame.joints:
                index = index_by_nid.get(int(entry.nid))
                if index is not None:
                    positions[index] = float(entry.position)
            if np.isfinite(positions).all():
                connection.state.update(positions)

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.left.get()
                self.right.get()
                return
            except RuntimeError:
                time.sleep(0.02)
        raise TimeoutError("timed out waiting for Wuji hand state")

    def send(self, side: str, joints: np.ndarray) -> None:
        if not self.enable_commands:
            return
        from wuji_sdk import JointCommand

        connection = self._connections[side]
        target = np.asarray(joints, dtype=np.float32)
        if target.shape != (JOINT_COUNT,) or not np.isfinite(target).all():
            raise ValueError(f"{side} hand command must be finite with shape (20,)")
        target = np.clip(target, -1.57, 1.57)
        if self.max_step > 0.0:
            actual = connection.state.get()
            target = actual + np.clip(target - actual, -self.max_step, self.max_step)
        connection.publisher.send([JointCommand(float(value), 0.0, 0.0) for value in target])

    def close(self) -> None:
        self._stop.set()
        for connection in self._connections.values():
            try:
                connection.subscription.close()
            except Exception:
                pass
        for thread in self._threads:
            thread.join(timeout=1.0)
        for connection in self._connections.values():
            try:
                connection.publisher.close()
            except Exception:
                pass
            try:
                if self.enable_commands:
                    connection.hand.disable()
            except Exception:
                pass
            if connection.controller_context is not None:
                try:
                    connection.controller_context.__exit__(None, None, None)
                except Exception:
                    pass
            try:
                connection.hand.disconnect()
            except Exception:
                pass
        self._connections.clear()
        if self._manager is not None:
            try:
                self._manager.disconnect_all()
            except Exception:
                pass
