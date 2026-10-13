"""Passive ROS state sidecar. Never imports the robot SDK or creates publishers."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .rtc_diagnostics import RTCDiagnostics


def finite_values(values):
    return [float(x) if math.isfinite(float(x)) else None for x in values]


def info_value(info, name, default=None):
    return info.get(name, default) if isinstance(info, dict) else getattr(info, name, default)


def state_fields(message, info=None):
    """ROS-free extraction used by tests, preserving unknown timestamp semantics."""
    stamp = message.header.stamp
    fields = dict(header_stamp_ns=int(stamp.sec) * 10**9 + int(stamp.nanosec),
                  frame_id=message.header.frame_id,
                  publisher_gid=bytes(info_value(info, "publisher_gid", b"")).hex(),
                  dds_source_timestamp_ns=info_value(info, "source_timestamp"),
                  dds_received_timestamp_ns=info_value(info, "received_timestamp"))
    fields["publisher_gid_available"] = bool(fields["publisher_gid"])
    if hasattr(message, "pose"):
        p, q = message.pose.position, message.pose.orientation
        fields["pose7"] = finite_values([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
    else:
        fields.update(position=finite_values(message.position), names=list(message.name))
    return fields


class StateRecorder:
    """Bound saved samples; counts describe consumed messages, not all publications."""
    def __init__(self, trace, max_hz):
        self.trace = trace
        self.period_ns = int(1e9 / max_hz) if max_hz else 0
        self.stats = defaultdict(lambda: dict(received=0, recorded=0, last_receive_ns=None,
                                             last_record_ns=None, max_receive_gap_ms=0.))

    def receive(self, topic, message, info=None):
        mono, wall = time.monotonic_ns(), time.time_ns()
        gid = bytes(info_value(info, "publisher_gid", b"")).hex()
        stat = self.stats[(topic, gid)]
        stat["received"] += 1
        previous = stat["last_receive_ns"]
        if previous is not None:
            stat["max_receive_gap_ms"] = max(stat["max_receive_gap_ms"], (mono - previous) / 1e6)
        stat["last_receive_ns"] = mono
        if stat["last_record_ns"] is not None and mono - stat["last_record_ns"] < self.period_ns:
            return
        stat["last_record_ns"] = mono
        stat["recorded"] += 1
        self.trace.record("feedback", monotonic_ns=mono, wall_time_ns=wall,
                          topic=topic, received_count=stat["received"],
                          max_receive_gap_ms=stat["max_receive_gap_ms"], **state_fields(message, info))

    def summary(self):
        return [dict(topic=topic, publisher_gid=gid, **stat)
                for (topic, gid), stat in self.stats.items()]


class NativeFeedbackSampler:
    """Poll KEEP_LAST=1 queues instead of deserializing every 250 Hz publication.

    Astribot's rclpy executor drops native MessageInfo. This adapter retains it
    without changing that library and supports its additional take argument.
    """
    def __init__(self):
        self.command_argument = None

    def take(self, sub):
        with sub.handle:
            if self.command_argument is not False:
                try:
                    result = sub.handle.take_message(sub.msg_type, sub.raw, "")
                    self.command_argument = True
                    return result
                except TypeError:
                    if self.command_argument is True:
                        raise
                    self.command_argument = False
            return sub.handle.take_message(sub.msg_type, sub.raw)


class PassiveStateProcess:
    def __init__(self, diagnostics, max_hz=60.):
        self.diagnostics = diagnostics
        self.path = diagnostics.path.with_suffix(".states.jsonl")
        self.ready = diagnostics.path.with_suffix(".states.ready.json")
        self.stderr_path = diagnostics.path.with_suffix(".states.stderr.log")
        self.process = self._stderr = None
        self.max_hz = max_hz
        self._failure_reported = False

    def start(self):
        if any(p.exists() for p in (self.path, self.ready, self.stderr_path)):
            raise FileExistsError("State trace sidecars already exist; use a new diagnostics filename")
        self._stderr = self.stderr_path.open("x", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "openwam_robot_client.rtc_state_trace", "--out", str(self.path),
                 "--ready", str(self.ready), "--max-hz", str(self.max_hz),
                 "--parent-pid", str(os.getpid())],
                stdin=subprocess.DEVNULL, stdout=self._stderr, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 10.
            while not self.ready.exists():
                if self.process.poll() is not None:
                    raise RuntimeError(f"Passive state trace failed; inspect {self.stderr_path}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Passive state trace did not start; inspect {self.stderr_path}")
                time.sleep(.02)
            self.diagnostics.record("state_trace_started", path=str(self.path),
                                    pid=self.process.pid, poll_hz=self.max_hz)
        except BaseException:
            self.close()
            raise

    def check(self):
        if self.process is not None and self.process.poll() is not None and not self._failure_reported:
            self._failure_reported = True
            self.diagnostics.record("state_trace_failed", returncode=self.process.returncode,
                                    stderr_path=str(self.stderr_path))
            logging.getLogger(__name__).error("RTC passive feedback stopped; inspect %s", self.stderr_path)

    def close(self):
        if self.process is not None:
            forced = False
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3.)
                except subprocess.TimeoutExpired:
                    forced = True
                    self.process.kill()
                    self.process.wait(timeout=2.)
            self.diagnostics.record("state_trace_stopped", returncode=self.process.returncode,
                                    forced=forced, path=str(self.path))
            self.process = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ready", type=Path)
    ap.add_argument("--max-hz", type=float, default=60.)
    ap.add_argument("--seconds", type=float, default=0., help="0 means until interrupted")
    ap.add_argument("--parent-pid", type=int)
    args = ap.parse_args(argv)
    if not 1 <= args.max_hz <= 250 or not math.isfinite(args.seconds) or args.seconds < 0:
        ap.error("max-hz must be 1..250 and seconds must be finite and nonnegative")
    # Imports are deliberately here: importing this module does not initialize ROS.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from astribot_msgs.msg import RobotCartesianState, RobotJointState

    rclpy.init(args=[])
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    node = Node('openwam_rtc_state_trace_' + str(os.getpid()),
                enable_rosout=False, start_parameter_services=False)
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                     reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
    trace = RTCDiagnostics(args.out, exclusive=True)
    recorder = StateRecorder(trace, None)  # The polling loop already bounds sampling.

    sampler = NativeFeedbackSampler()
    topics = {}
    for arm in ("astribot_arm_left", "astribot_arm_right"):
        for name in ("endpoint_current_states", "endpoint_desired_states", "sdk_current_states", "sdk_desired_states"):
            topics[f"/{arm}/{name}"] = RobotCartesianState
        for name in ("joint_space_states", "joint_space_command_recv"):
            topics[f"/{arm}/{name}"] = RobotJointState

    def metadata():
        return {topic: [dict(node_name=p.node_name, node_namespace=p.node_namespace,
                             publisher_gid=bytes(p.endpoint_gid).hex(), type=p.topic_type,
                             reliability=str(p.qos_profile.reliability), depth=p.qos_profile.depth)
                        for p in node.get_publishers_info_by_topic(topic)] for topic in topics}

    subscriptions = []
    started, cpu_started = time.monotonic(), time.process_time()
    try:
        trace.record("state_trace_start", pid=os.getpid(), poll_hz=args.max_hz,
                     topics=list(topics), sampling="Poll KEEP_LAST=1; counts exclude intentionally unsampled publications",
                     timestamp_note="Headers/DDS timestamps are not proven sensor measurement times")
        for topic, typ in topics.items():
            subscriptions.append((topic, node.create_subscription(typ, topic, lambda message: None, qos)))
        ready_written = False
        next_health = started + 2.
        deadline = time.monotonic()
        while not stopping and rclpy.ok():
            if args.parent_pid and os.getppid() != args.parent_pid:
                break
            if args.seconds and time.monotonic() - started >= args.seconds:
                break
            for topic, sub in subscriptions:
                result = sampler.take(sub)
                if result is not None:
                    recorder.receive(topic, result[0], result[1])
            if args.ready and not ready_written and recorder.stats:
                temp = args.ready.with_suffix(".tmp")
                temp.write_text(json.dumps(dict(pid=os.getpid(), output=str(args.out),
                                                sampled_topics=sorted({key[0] for key in recorder.stats}))),
                                encoding="utf-8")
                os.replace(temp, args.ready)
                ready_written = True
            if time.monotonic() >= next_health:
                trace.record("state_trace_health", streams=recorder.summary(),
                             publishers=metadata(), elapsed_s=time.monotonic() - started,
                             process_cpu_s=time.process_time() - cpu_started)
                next_health = time.monotonic() + 5.
            deadline += 1. / args.max_hz
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                deadline = time.monotonic()  # Do not issue catch-up polls.
    finally:
        trace.record("state_trace_summary", streams=recorder.summary(), publishers=metadata(),
                     elapsed_s=time.monotonic() - started, process_cpu_s=time.process_time() - cpu_started)
        trace.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
