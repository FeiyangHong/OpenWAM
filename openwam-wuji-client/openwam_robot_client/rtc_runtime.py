"""RTC client scheduler: one local action clock and one nonblocking network worker."""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future

import numpy as np

from .cameras import CameraStreams
from .rtc_controller import RTCController, RTCWujiHands
from .rtc_transport import RTCTransport
from .runtime import KeyboardStop, _validated_actions

logger = logging.getLogger(__name__)


class NetworkWorker:
    """Only this worker accesses transport while active; no unbounded backlog."""
    def __init__(self, transport):
        self.transport = transport
        self.jobs = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name="rtc-network", daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopping.is_set():
            try:
                payload, future = self.jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                response = self.transport.request(payload)
                response["_received_at"] = time.monotonic()
                future.set_result(response)
            except BaseException as exc:
                future.set_exception(exc)

    def submit(self, payload):
        future = Future()
        self.jobs.put_nowait((payload, future))
        return future

    def close(self):
        self.stopping.set()
        # Called only after local robot stop; close_timeout is bounded.
        self.transport.close()
        self.thread.join(timeout=1.)


class RTCScheduler:
    """Tick indices count successfully published commands, never network requests."""
    def __init__(self, controller, worker, horizon=32, execute_steps=16, prefix_steps=10):
        if not 0 < prefix_steps < execute_steps or execute_steps + prefix_steps > horizon:
            raise ValueError("RTC requires 0 < prefix_steps < execution_horizon and d+s <= model horizon")
        self.controller, self.worker = controller, worker
        self.horizon, self.s, self.d = horizon, execute_steps, prefix_steps
        self.buffer = deque()
        self.pending = None
        self.pending_request = None
        self.step = 0
        self.request_id = 0
        self.chunks = 0

    def request(self, observation, warmup=False):
        if self.pending is not None:
            raise RuntimeError("RTC already has an outstanding request")
        prefix = [item.raw.tolist() for item in self.buffer]
        if warmup:
            prefix = [list(observation["state"]) for _ in range(self.d)]
        elif self.step and len(prefix) != self.d:
            raise RuntimeError("RTC request must start with exactly d committed commands remaining")
        self.request_id += 1
        payload = dict(observation, type="rtc_warmup" if warmup else "rtc_chunk",
                       request_id=self.request_id, start_step=self.step, execute_steps=self.s, rtc_prefix=prefix)
        self.pending_request = payload
        self.pending_started = observation.get("observation_time", time.monotonic())
        self.pending = self.worker.submit(payload)

    def finish_warmup(self):
        result = self.pending.result()
        if result.get("type") != "rtc_warmup_ack":
            raise RuntimeError("Invalid RTC warmup response")
        self.pending = self.pending_request = None

    def adopt(self):
        if self.buffer or self.pending is None:
            raise RuntimeError("RTC results may only be adopted at the committed-prefix boundary")
        if not self.pending.done():
            raise TimeoutError("RTC inference exceeded the prefix budget; stopping (increase d or reduce latency)")
        request = self.pending_request
        response = self.pending.result()
        d = len(request["rtc_prefix"])
        if self.step != request["start_step"] + d:
            raise RuntimeError("RTC prefix was not executed exactly once")
        for name, expected in (("start_step", request["start_step"]), ("prefix_steps", d),
                               ("execute_steps", self.s), ("request_id", request["request_id"])):
            if response.get(name) != expected:
                raise RuntimeError(f"RTC response timeline mismatch: {name}")
        actions = _validated_actions(response, self.horizon)
        self.buffer.extend(self.controller.prepare(actions[d:d + self.s]))
        self.pending = self.pending_request = None
        self.chunks += 1
        logger.info("rtc chunk=%d step=%d prefix=%d server_ms=%.1f observation_to_response_ms=%.1f",
                    self.chunks, self.step, d, response.get("latency_ms", 0.),
                    (response.get("_received_at", time.monotonic()) - self.pending_started) * 1000)

    def execute(self):
        if not self.buffer:
            self.adopt()
        action = self.buffer[0]
        self.controller.execute_prepared(action)
        self.buffer.popleft()
        self.step += 1


def _wait(future, stop):
    while not future.done():
        if stop.triggered():
            raise KeyboardInterrupt
        time.sleep(0.01)
    return future.result()


def run(args):
    import rclpy
    from astribot_sdk.core.astribot_api.astribot_client import Astribot

    rclpy.init()
    node = rclpy.create_node("openwam_wuji_rtc_client")
    cameras = CameraStreams(node, {"head_camera": args.head_topic,
                                   "left_wrist_camera": args.left_wrist_topic,
                                   "right_wrist_camera": args.right_wrist_topic})
    transport = RTCTransport(args.server, timeout=args.timeout)
    hands = robot = worker = None
    try:
        # Handshake before acquiring robot command rights.
        contract = transport.ping_and_validate()
        if args.rtc_prefix_steps > contract["max_prefix_steps"] or args.control_hz != contract["control_hz"]:
            raise ValueError("Client prefix/control frequency differs from RTC server capability")
        if args.execution_horizon + args.rtc_prefix_steps > contract["action_horizon"]:
            raise ValueError("Client execution horizon plus prefix exceeds model chunk length")
        transport.reset()
        cameras.start()
        hands = RTCWujiHands(model=args.hand_model, left_serial=args.left_hand_serial,
                             right_serial=args.right_hand_serial, enable_commands=args.enable_hands,
                             effort_limit=args.hand_effort_limit, lowpass_hz=args.hand_lowpass_hz,
                             mit_kp=args.hand_mit_kp, mit_kd=args.hand_mit_kd,
                             max_step=args.max_hand_step, timeout=args.warmup_timeout)
        robot = Astribot(freq=args.control_hz, high_control_rights=args.enable_arms)
        if args.enable_arms and args.control_way == "filter":
            robot.set_filter_parameters(args.filter_scale, args.filter_scale)
        if args.move_home:
            robot.move_to_home()
        controller = RTCController(robot, hands, enable_arms=args.enable_arms, eef_frame=args.eef_frame,
                                   control_way=args.control_way, use_wbc=args.use_wbc,
                                   workspace_radius=np.asarray(args.workspace_radius),
                                   max_eef_step=args.max_eef_step, max_rotation_step_rad=args.max_rotation_step_rad)
        cameras.wait_ready(args.warmup_timeout)
        worker = NetworkWorker(transport)
        scheduler = RTCScheduler(controller, worker, contract["action_horizon"],
                                 args.execution_horizon, args.rtc_prefix_steps)

        def observation():
            observed_at = time.monotonic()
            images, sequences = cameras.snapshot(args.max_camera_age)
            state = controller.state_vector()
            return {"images": images, "state": state, "prompt": args.task,
                    "camera_sequences": sequences, "observation_time": observed_at}

        with KeyboardStop() as stop:
            for _ in range(args.rtc_warmup_rounds):
                scheduler.request(observation(), warmup=True)
                _wait(scheduler.pending, stop)
                scheduler.finish_warmup()
            logger.info("RTC warmup complete; arms=%s hands=%s", args.enable_arms, args.enable_hands)
            scheduler.request(observation())
            _wait(scheduler.pending, stop)
            scheduler.adopt()
            period = 1. / args.control_hz
            deadline = time.monotonic()
            while True:
                if stop.triggered():
                    break
                now = time.monotonic()
                if now < deadline:
                    time.sleep(min(deadline - now, 0.005))
                    continue
                if now - deadline > args.max_tick_lag_ms / 1000:
                    raise TimeoutError("RTC control loop missed its deadline; stopping without catch-up commands")
                tick_started = now
                # Keep stale-camera detection live even between network requests.
                cameras.snapshot(args.max_camera_age)
                if not scheduler.buffer:
                    if args.max_chunks and scheduler.chunks >= args.max_chunks:
                        break
                    scheduler.adopt()
                if (len(scheduler.buffer) == scheduler.d and scheduler.pending is None and
                        (not args.max_chunks or scheduler.chunks < args.max_chunks)):
                    scheduler.request(observation())
                scheduler.execute()
                # Never issue a burst to make up for delayed SDK/network scheduling.
                deadline = max(deadline + period, tick_started + period)
    finally:
        # Stop hardware before touching any possibly-blocked network operation.
        if robot is not None and args.enable_arms:
            try:
                robot.stop_robot()
            except Exception:
                logger.exception("Robot stop failed")
        if hands is not None:
            hands.close()
        if worker is not None:
            worker.close()
        else:
            transport.close()
        cameras.stop()
        rclpy.shutdown()
