"""RTC client scheduler: one local action clock and one nonblocking network worker."""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future
from pathlib import Path

import numpy as np

from .cameras import CameraStreams
from .rtc_controller import RTCController, RTCEmergencyWujiHands, RTCWujiHands
from .rtc_diagnostics import RTCDiagnostics, run_log_path
from .rtc_transport import RTCTransport
from .runtime import KeyboardStop, _validated_actions
from .robot import Action58

logger = logging.getLogger(__name__)


class NetworkWorker:
    """Only this worker accesses transport while active; no unbounded backlog."""
    def __init__(self, transport, diagnostics=None):
        self.transport = transport
        self.diagnostics = diagnostics
        self.jobs = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name="rtc-network", daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopping.is_set():
            try:
                payload, future, submitted_at, planner = self.jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            phase = "network"
            try:
                started = time.monotonic()
                if self.diagnostics:
                    self.diagnostics.record("network_start", request_id=payload["request_id"],
                                            queue_ms=(started - submitted_at) * 1000)
                response = self.transport.request(payload)
                response["_received_at"] = time.monotonic()
                if self.diagnostics:
                    self.diagnostics.record("network_done", request_id=payload["request_id"],
                                            network_ms=(response["_received_at"] - started) * 1000,
                                            server_ms=response.get("latency_ms"),
                                            model_diagnostics=response.get("rtc_diagnostics"),
                                            model_actions58=response.get("actions"))
                if planner is not None:
                    phase = "plan"
                    plan_started = time.monotonic()
                    response["_prepared_actions"] = planner(response)
                    response["_prepare_ms"] = (time.monotonic() - plan_started) * 1000
                    response["_ready_at"] = time.monotonic()
                    if self.diagnostics:
                        self.diagnostics.record("plan_done", request_id=payload["request_id"],
                                                prepare_ms=response["_prepare_ms"])
                future.set_result(response)
            except BaseException as exc:
                if self.diagnostics:
                    self.diagnostics.record("network_error" if phase == "network" else "plan_error",
                                            request_id=payload["request_id"],
                                            error=repr(exc))
                future.set_exception(exc)

    def submit(self, payload, planner=None):
        future = Future()
        self.jobs.put_nowait((payload, future, time.monotonic(), planner))
        return future

    def close(self):
        self.stopping.set()
        # Called only after local robot stop; close_timeout is bounded.
        self.transport.close()
        self.thread.join(timeout=1.)


class RTCScheduler:
    """Tick indices count successfully published commands, never network requests."""
    def __init__(self, controller, worker, horizon=32, execute_steps=16, prefix_steps=10, control_hz=30,
                 diagnostics=None):
        if not 0 < prefix_steps < execute_steps or execute_steps + prefix_steps > horizon:
            raise ValueError("RTC requires 0 < prefix_steps < execution_horizon and d+s <= model horizon")
        self.controller, self.worker = controller, worker
        self.diagnostics = diagnostics
        self.horizon, self.s, self.d = horizon, execute_steps, prefix_steps
        self.prefix_budget_ms = prefix_steps / control_hz * 1000
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
        if self.diagnostics:
            payload["rtc_diagnostics"] = True
        planner = None
        if not warmup and self.step and hasattr(self.controller, "prepare_from"):
            endpoint = self.buffer[-1]
            endpoint_action = Action58.parse(endpoint.raw)
            seed_poses = (endpoint.left.copy(), endpoint.right.copy())
            seed_hands = (endpoint_action.left_hand.copy(), endpoint_action.right_hand.copy())
            d, request_id, start_step = len(prefix), self.request_id, self.step

            def planner(response):
                for name, expected in (("request_id", request_id), ("start_step", start_step),
                                       ("prefix_steps", d), ("execute_steps", self.s)):
                    if response.get(name) != expected:
                        raise RuntimeError(f"RTC response timeline mismatch: {name}")
                actions = _validated_actions(response, self.horizon)
                return self.controller.prepare_from(actions[d:d + self.s], seed_poses, seed_hands)

        self.pending_request = payload
        self.pending_started = observation.get("observation_time", time.monotonic())
        self.pending = self.worker.submit(payload, planner=planner) if planner else self.worker.submit(payload)
        if self.diagnostics:
            self.diagnostics.record("request_submit", request_id=self.request_id,
                                    step=self.step, prefix=len(prefix), warmup=warmup,
                                    shadow_tick_deadline_ns=getattr(self, 'shadow_tick_deadline_ns', None),
                                    prefix_actions58=prefix,
                                    observation_start_monotonic_ns=int(self.pending_started * 1e9),
                                    observation_to_submit_ms=(time.monotonic() - self.pending_started) * 1000,
                                    observed_state58=np.asarray(observation.get("state"), dtype=np.float32).tolist()
                                    if observation.get("state") is not None else None,
                                    prefix_previous58=prefix[-2] if len(prefix) >= 2 else None,
                                    prefix_end58=prefix[-1] if prefix else None)

    def finish_warmup(self):
        result = self.pending.result()
        if result.get("type") != "rtc_warmup_ack":
            raise RuntimeError("Invalid RTC warmup response")
        self.pending = self.pending_request = None

    def adopt(self):
        if self.buffer or self.pending is None:
            raise RuntimeError("RTC results may only be adopted at the committed-prefix boundary")
        if not self.pending.done():
            elapsed_ms = (time.monotonic() - self.pending_started) * 1000
            if self.diagnostics:
                self.diagnostics.record("prefix_timeout", request_id=self.pending_request["request_id"],
                                        step=self.step, elapsed_ms=elapsed_ms,
                                        budget_ms=self.prefix_budget_ms)
            raise TimeoutError(
                f"RTC inference exceeded prefix budget: elapsed={elapsed_ms:.1f} ms, "
                f"budget={self.prefix_budget_ms:.1f} ms (prefix={self.d} steps); stopping"
            )
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
        if self.step == 0:
            align_start_hands = getattr(self.controller, "align_start_hands", None)
            if align_start_hands is not None:
                align_start_hands()
        model_gap = None
        if self.diagnostics:
            first_raw = Action58.parse(actions[d])
            model_gap = dict(
                model_left_xyz_gap_m=float(np.linalg.norm(first_raw.left_eef[:3] - self.controller.previous[0][:3])),
                model_right_xyz_gap_m=float(np.linalg.norm(first_raw.right_eef[:3] - self.controller.previous[1][:3])),
                model_left_hand_gap_rad=float(np.max(np.abs(first_raw.left_hand - self.controller.previous_hands[0]))),
                model_right_hand_gap_rad=float(np.max(np.abs(first_raw.right_hand - self.controller.previous_hands[1]))),
            )
        if not d and response.get("latency_ms", 0) > self.prefix_budget_ms:
            logger.warning("RTC server latency %.1f ms exceeds next-prefix budget %.1f ms; "
                           "continuous execution may stop", response["latency_ms"], self.prefix_budget_ms)
        prepare_start = time.monotonic()
        preplanned = response.get("_prepared_actions")
        if preplanned is None:
            preplanned = self.controller.prepare(actions[d:d + self.s])
        if len(preplanned) != self.s:
            raise RuntimeError("RTC prepared action count differs from execution horizon")
        self.buffer.extend(preplanned)
        adopt_prepare_ms = (time.monotonic() - prepare_start) * 1000
        prepare_ms = response.get("_prepare_ms", adopt_prepare_ms)
        self.pending = self.pending_request = None
        self.chunks += 1
        logger.info("rtc chunk=%d step=%d prefix=%d server_ms=%.1f observation_to_response_ms=%.1f",
                    self.chunks, self.step, d, response.get("latency_ms", 0.),
                    (response.get("_received_at", time.monotonic()) - self.pending_started) * 1000)
        if self.diagnostics:
            self.diagnostics.record("chunk_adopt", chunk=self.chunks, step=self.step,
                                    request_id=request["request_id"], prefix=d,
                                    prepare_ms=prepare_ms, adopt_prepare_ms=adopt_prepare_ms,
                                    preplanned=response.get("_prepared_actions") is not None,
                                    ready_to_adopt_ms=(prepare_start - response.get("_ready_at", prepare_start)) * 1000,
                                    response_to_adopt_ms=(prepare_start - response.get("_received_at", prepare_start)) * 1000,
                                    observation_to_response_ms=(response.get("_received_at", prepare_start) - self.pending_started) * 1000,
                                    server_ms=response.get("latency_ms"),
                                    model_diagnostics=response.get("rtc_diagnostics"),
                                    prepared_actions58=[item.raw.tolist() for item in preplanned], **model_gap)

    def execute(self):
        if not self.buffer:
            self.adopt()
        action = self.buffer[0]
        if self.diagnostics:
            self.controller.diagnostics_context = (self.step, self.chunks, len(self.buffer) == self.s)
        self.controller.execute_prepared(action)
        self.buffer.popleft()
        self.step += 1


def _wait(future, stop):
    while not future.done():
        if stop.triggered():
            raise KeyboardInterrupt
        time.sleep(0.01)
    return future.result()


def run(args, *, hands_factory=None, controller_factory=None, left_init_loader=None):
    import rclpy
    from astribot_sdk.core.astribot_api.astribot_client import Astribot

    rclpy.init()
    diagnostics = None
    if args.rtc_diagnostics_dir:
        root = Path(args.rtc_diagnostics_dir).expanduser()
        if not root.is_absolute():
            root = Path(__file__).resolve().parent.parent / root
        diagnostics = RTCDiagnostics(run_log_path(root, args.task,
                                                 args.rtc_diagnostics_label), exclusive=True)
    elif args.rtc_diagnostics_file:
        diagnostics = RTCDiagnostics(args.rtc_diagnostics_file)
    if diagnostics:
        logger.info("RTC diagnostics: %s", diagnostics.path)
        diagnostics.record("run_start", task=args.task, server=args.server,
                           control_hz=args.control_hz, execution_horizon=args.execution_horizon,
                           prefix_steps=args.rtc_prefix_steps, enable_arms=args.enable_arms,
                           enable_hands=args.enable_hands, schema_version=2,
                           settings=vars(args), timestamp_note="monotonic_ns values share the Orin host clock")
    node = rclpy.create_node("openwam_wuji_rtc_client")
    if diagnostics:
        from .rtc_observation import RTCCameraStreams
        camera_class = RTCCameraStreams
    else:
        camera_class = CameraStreams
    cameras = camera_class(node, {"head_camera": args.head_topic,
                                   "left_wrist_camera": args.left_wrist_topic,
                                   "right_wrist_camera": args.right_wrist_topic})
    transport = RTCTransport(args.server, timeout=args.timeout)
    hands = robot = worker = state_trace = None
    try:
        left_init = None
        if args.hold_left_init:
            from .left_init import read_left_init
            left_init = (left_init_loader or read_left_init)()
            if args.ignore_left_thumb:
                args.left_thumb_state = tuple(float(v) for v in left_init[2][:4])
        # Handshake before acquiring robot command rights.
        contract = transport.ping_and_validate()
        if getattr(args, 'rtc_shadow_align', False):
            diagnostics.record('shadow_contract', contract=contract,
                               note='Publication/receipt-time proxy only; no aligned data is sent to server')
            cameras.enable_shadow_metadata(diagnostics)
        if args.rtc_prefix_steps > contract["max_prefix_steps"] or args.control_hz != contract["control_hz"]:
            raise ValueError("Client prefix/control frequency differs from RTC server capability")
        if args.execution_horizon + args.rtc_prefix_steps > contract["action_horizon"]:
            raise ValueError("Client execution horizon plus prefix exceeds model chunk length")
        transport.reset()
        cameras.start()
        if args.rtc_state_trace:
            from .rtc_state_trace import PassiveStateProcess
            state_trace = PassiveStateProcess(diagnostics)
            state_trace.start()
            logger.info("RTC passive feedback: %s", state_trace.path)
        hands_class = hands_factory or (RTCEmergencyWujiHands if args.ignore_left_thumb else RTCWujiHands)
        hand_options = dict(model=args.hand_model, left_serial=args.left_hand_serial,
                            right_serial=args.right_hand_serial, enable_commands=args.enable_hands,
                            effort_limit=args.hand_effort_limit, lowpass_hz=args.hand_lowpass_hz,
                            mit_kp=args.hand_mit_kp, mit_kd=args.hand_mit_kd,
                            max_step=args.max_hand_step, timeout=args.warmup_timeout)
        if args.ignore_left_thumb:
            hand_options.update(thumb_state=args.left_thumb_state,
                                thumb_state_mode=args.left_thumb_state_mode)
        if getattr(args, 'rtc_shadow_align', False):
            hand_options['shadow_diagnostics'] = diagnostics
        hands = hands_class(**hand_options)
        hands.tracking_tolerance = args.rtc_hand_tracking_tolerance
        robot = Astribot(freq=args.control_hz, high_control_rights=args.enable_arms)
        if args.enable_arms and args.control_way == "filter":
            robot.set_filter_parameters(args.filter_scale, args.filter_scale)
        if args.move_home:
            robot.move_to_home()
        controller = (controller_factory or RTCController)(robot, hands, enable_arms=args.enable_arms, eef_frame=args.eef_frame,
                                   control_way=args.control_way, use_wbc=args.use_wbc,
                                   workspace_radius=np.asarray(args.workspace_radius),
                                   max_eef_step=args.max_eef_step, max_rotation_step_rad=args.max_rotation_step_rad)
        controller.diagnostics = diagnostics
        if left_init is not None:
            from .left_init import left_init_pose
            controller.hold_left_init(
                left_init_pose(robot, left_init[0], left_init[1], eef_frame=args.eef_frame),
                left_init[2],
            )
            logger.warning("RTC left arm/hand init hold enabled; prefix contains replaced commands")
        cameras.wait_ready(args.warmup_timeout)
        worker = NetworkWorker(transport, diagnostics)
        scheduler = RTCScheduler(controller, worker, contract["action_horizon"],
                                 args.execution_horizon, args.rtc_prefix_steps, args.control_hz,
                                 diagnostics=diagnostics)

        def observation():
            observed_at = time.monotonic()
            if state_trace is not None:
                state_trace.check()
            if diagnostics:
                images, sequences, camera_timing = cameras.snapshot_with_timing(args.max_camera_age)
            else:
                images, sequences = cameras.snapshot(args.max_camera_age)
            camera_at = time.monotonic()
            state_start_ns = time.monotonic_ns() if diagnostics else None
            state = controller.state_vector()
            if diagnostics:
                state_end_ns = time.monotonic_ns()
                actual = Action58.parse(state)
                diagnostics.record("observation", step=scheduler.step,
                                   request_id=scheduler.request_id + 1,
                                   observation_start_monotonic_ns=int(observed_at * 1e9),
                                   state_start_monotonic_ns=state_start_ns,
                                   state_end_monotonic_ns=state_end_ns,
                                   sdk_arm_read=getattr(controller, "diagnostics_arm_read", None),
                                   cameras=camera_timing,
                                   camera_ms=(camera_at - observed_at) * 1000,
                                   state_ms=(time.monotonic() - camera_at) * 1000,
                                   observed_left_xyz_m=actual.left_eef[:3].tolist(),
                                   observed_right_xyz_m=actual.right_eef[:3].tolist(),
                                   last_command_left_xyz_m=controller.previous[0][:3].tolist(),
                                   last_command_right_xyz_m=controller.previous[1][:3].tolist(),
                                   left_xyz_tracking_error_m=float(np.linalg.norm(actual.left_eef[:3] - controller.previous[0][:3])),
                                   right_xyz_tracking_error_m=float(np.linalg.norm(actual.right_eef[:3] - controller.previous[1][:3])),
                                   left_hand_tracking_error_rad=float(np.max(np.abs(actual.left_hand - controller.previous_hands[0]))),
                                   right_hand_tracking_error_rad=float(np.max(np.abs(actual.right_hand - controller.previous_hands[1]))))
            if args.ignore_left_thumb:
                logger.info("RTC left thumb virtual state for inference: %s", state[9:13])
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
            wait_started = deadline
            wait_process_cpu_started = time.process_time() if diagnostics else 0.
            wait_thread_cpu_started = time.thread_time() if diagnostics else 0.
            last_tick_end = None
            sleep_calls = 0
            last_sleep_requested_ms = last_sleep_actual_ms = max_sleep_overshoot_ms = 0.
            while True:
                if stop.triggered():
                    break
                now = time.monotonic()
                if now < deadline:
                    requested = min(deadline - now, 0.005)
                    if diagnostics:
                        sleep_started = time.monotonic()
                    time.sleep(requested)
                    if diagnostics:
                        actual_ms = (time.monotonic() - sleep_started) * 1000
                        sleep_calls += 1
                        last_sleep_requested_ms = requested * 1000
                        last_sleep_actual_ms = actual_ms
                        max_sleep_overshoot_ms = max(max_sleep_overshoot_ms, actual_ms - last_sleep_requested_ms)
                    continue
                wait_metrics = dict(
                    wait_elapsed_ms=(now - wait_started) * 1000,
                    since_last_tick_end_ms=(now - last_tick_end) * 1000 if last_tick_end is not None else None,
                    sleep_calls=sleep_calls,
                    last_sleep_requested_ms=last_sleep_requested_ms,
                    last_sleep_actual_ms=last_sleep_actual_ms,
                    max_sleep_overshoot_ms=max_sleep_overshoot_ms,
                    process_cpu_wait_ms=(time.process_time() - wait_process_cpu_started) * 1000,
                    control_thread_cpu_wait_ms=(time.thread_time() - wait_thread_cpu_started) * 1000,
                ) if diagnostics else None
                if now - deadline > args.max_tick_lag_ms / 1000:
                    if diagnostics:
                        diagnostics.record("tick_deadline_miss", step=scheduler.step,
                                           lag_ms=(now - deadline) * 1000, **wait_metrics)
                    raise TimeoutError("RTC control loop missed its deadline; stopping without catch-up commands")
                tick_started = now
                stage = "camera"
                camera_ms = adopt_ms = request_ms = execute_ms = 0.
                tick_ok = False
                try:
                    # Keep stale-camera detection live even between network requests.
                    phase_at = time.monotonic()
                    cameras.snapshot(args.max_camera_age)
                    camera_ms = (time.monotonic() - phase_at) * 1000
                    if not scheduler.buffer:
                        if args.max_chunks and scheduler.chunks >= args.max_chunks:
                            break
                        stage = "adopt"
                        phase_at = time.monotonic()
                        try:
                            scheduler.adopt()
                        except TimeoutError:
                            # In no-actuator checks only, wait briefly for the late
                            # response so its actual latency can guide diagnosis.
                            if not args.enable_arms and not args.enable_hands and scheduler.pending is not None:
                                try:
                                    late = scheduler.pending.result(timeout=min(args.timeout, 10.))
                                    logger.error("RTC late dry-run response: server_ms=%.1f total_ms=%.1f",
                                                 late.get("latency_ms", float("nan")),
                                                 (time.monotonic() - scheduler.pending_started) * 1000)
                                except Exception:
                                    logger.exception("RTC late dry-run response did not arrive within 10 seconds")
                            raise
                        adopt_ms = (time.monotonic() - phase_at) * 1000
                    if (len(scheduler.buffer) == scheduler.d and scheduler.pending is None and
                            (not args.max_chunks or scheduler.chunks < args.max_chunks)):
                        stage = "request"
                        phase_at = time.monotonic()
                        if getattr(args, 'rtc_shadow_align', False):
                            scheduler.shadow_tick_deadline_ns = int(deadline * 1e9)
                        scheduler.request(observation())
                        request_ms = (time.monotonic() - phase_at) * 1000
                    stage = "execute"
                    phase_at = time.monotonic()
                    if diagnostics:
                        controller.diagnostics_deadline_ns = int(deadline * 1e9)
                    scheduler.execute()
                    execute_ms = (time.monotonic() - phase_at) * 1000
                    tick_ok = True
                finally:
                    if diagnostics:
                        diagnostics.record("tick", step=scheduler.step, chunk=scheduler.chunks,
                                           lag_ms=(tick_started - deadline) * 1000,
                                           camera_ms=camera_ms, adopt_ms=adopt_ms,
                                           request_ms=request_ms, execute_ms=execute_ms,
                                           total_ms=(time.monotonic() - tick_started) * 1000,
                                           stage=stage, ok=tick_ok, **wait_metrics)
                # Never issue a burst to make up for delayed SDK/network scheduling.
                deadline = max(deadline + period, tick_started + period)
                if diagnostics:
                    last_tick_end = wait_started = time.monotonic()
                    wait_process_cpu_started = time.process_time()
                    wait_thread_cpu_started = time.thread_time()
                    sleep_calls = 0
                    last_sleep_requested_ms = last_sleep_actual_ms = max_sleep_overshoot_ms = 0.
    finally:
        if diagnostics:
            diagnostics.record("cleanup_start")
        try:
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
            if rclpy.ok():
                rclpy.shutdown()
        finally:
            if state_trace is not None:
                state_trace.close()
            if diagnostics:
                diagnostics.record("cleanup_end")
                diagnostics.close()
