"""RTC-only camera timestamp capture; image bytes and selection stay unchanged."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from .cameras import CAMERA_TOPICS, CameraFrame, CameraStreams


@dataclass(frozen=True)
class TimedCameraFrame(CameraFrame):
    received_monotonic_ns: int
    received_wall_ns: int
    header_stamp_ns: int | None
    frame_id: str | None


class _LatestTimedFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def update(self, message):
        mono, wall = time.monotonic_ns(), time.time_ns()
        header = getattr(message, "header", None)
        stamp = getattr(header, "stamp", None)
        header_ns = None if stamp is None else int(stamp.sec) * 10**9 + int(stamp.nanosec)
        with self._lock:
            sequence = 1 if self._frame is None else self._frame.sequence + 1
            self._frame = TimedCameraFrame(bytes(message.data), sequence, time.monotonic(),
                                          mono, wall, header_ns, getattr(header, "frame_id", None))

    def get(self):
        with self._lock:
            return self._frame


class RTCCameraStreams(CameraStreams):
    def __init__(self, node, topics=None):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage

        self.node = node
        self.topics = dict(topics or CAMERA_TOPICS)
        self._frames = {name: _LatestTimedFrame() for name in CAMERA_TOPICS}
        self._executor = self._thread = None
        for name, topic in self.topics.items():
            node.create_subscription(CompressedImage, topic, self._frames[name].update,
                                     qos_profile_sensor_data)

    def snapshot_with_timing(self, max_age):
        # Hold references to exactly the frames returned for this request.
        now = time.monotonic()
        frames = {name: stream.get() for name, stream in self._frames.items()}
        missing = [name for name, frame in frames.items() if frame is None]
        stale = [name for name, frame in frames.items()
                 if frame is not None and now - frame.received_at > max_age]
        if missing or stale:
            raise RuntimeError(f"camera snapshot invalid: missing={missing}, stale={stale}")
        metadata = {name: dict(sequence=f.sequence, topic=self.topics[name],
                               received_monotonic_ns=f.received_monotonic_ns,
                               received_wall_ns=f.received_wall_ns,
                               header_stamp_ns=f.header_stamp_ns, frame_id=f.frame_id,
                               receipt_age_ms=(now - f.received_at) * 1000,
                               compressed_bytes=len(f.data)) for name, f in frames.items()}
        return ({name: f.data for name, f in frames.items()},
                {name: f.sequence for name, f in frames.items()}, metadata)
