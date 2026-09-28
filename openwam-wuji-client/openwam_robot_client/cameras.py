from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

CAMERA_TOPICS = {
    "head_camera": "/astribot_camera/head_rgbd/color_compress/compressed",
    "left_wrist_camera": "/astribot_camera/left_wrist_rgbd/color_compress/compressed",
    "right_wrist_camera": "/astribot_camera/right_wrist_rgbd/color_compress/compressed",
}


@dataclass(frozen=True)
class CameraFrame:
    data: bytes
    sequence: int
    received_at: float


class _LatestCompressedFrame:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: CameraFrame | None = None

    def update(self, message: Any) -> None:
        with self._lock:
            sequence = 1 if self._frame is None else self._frame.sequence + 1
            self._frame = CameraFrame(bytes(message.data), sequence, time.monotonic())

    def get(self) -> CameraFrame | None:
        with self._lock:
            return self._frame


class CameraStreams:
    """ROS is limited to receiving the three already-compressed camera topics."""

    def __init__(self, node: Any, topics: dict[str, str] | None = None):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage

        self.node = node
        self.topics = dict(topics or CAMERA_TOPICS)
        self._frames = {name: _LatestCompressedFrame() for name in CAMERA_TOPICS}
        self._executor = None
        self._thread: threading.Thread | None = None
        for name, topic in self.topics.items():
            node.create_subscription(CompressedImage, topic, self._frames[name].update, qos_profile_sensor_data)

    def start(self) -> None:
        if self._thread is not None:
            return
        from rclpy.executors import SingleThreadedExecutor

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(
            target=self._executor.spin, name="openwam_camera_spin", daemon=True
        )
        self._thread.start()

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(stream.get() is not None for stream in self._frames.values()):
                return
            time.sleep(0.02)
        missing = [name for name, stream in self._frames.items() if stream.get() is None]
        raise TimeoutError(f"timed out waiting for camera streams: {missing}")

    def snapshot(self, max_age: float) -> tuple[dict[str, bytes], dict[str, int]]:
        now = time.monotonic()
        frames = {name: stream.get() for name, stream in self._frames.items()}
        missing = [name for name, frame in frames.items() if frame is None]
        stale = [name for name, frame in frames.items() if frame is not None and now - frame.received_at > max_age]
        if missing or stale:
            raise RuntimeError(f"camera snapshot invalid: missing={missing}, stale={stale}")
        return (
            {name: frame.data for name, frame in frames.items()},
            {name: frame.sequence for name, frame in frames.items()},
        )

    def stop(self) -> None:
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception:
                pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._executor = None
