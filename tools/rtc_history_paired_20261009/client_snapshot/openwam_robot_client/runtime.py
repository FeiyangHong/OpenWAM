from __future__ import annotations

import logging
import select
import sys
import termios
import time
import tty
from typing import Any

import numpy as np

from .cameras import CameraStreams
from .hands import WujiHands
from .robot import RAW_ACTION_DIM, RobotController
from .transport import PolicyTransport

logger = logging.getLogger(__name__)


class KeyboardStop:
    def __init__(self) -> None:
        self._fd = None
        self._settings = None

    def __enter__(self):
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_exc):
        if self._fd is not None and self._settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._settings)

    def triggered(self) -> bool:
        if self._fd is None:
            return False
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        return bool(readable and sys.stdin.read(1) == " ")


def _validated_actions(response: dict, expected_count: int) -> np.ndarray:
    actions = np.asarray(response.get("actions"), dtype=np.float32)
    if actions.shape != (expected_count, RAW_ACTION_DIM):
        raise ValueError(
            f"server action chunk must have shape ({expected_count}, {RAW_ACTION_DIM}), got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("server action chunk contains NaN or infinity")
    return actions


def run(args: Any) -> None:
    import rclpy
    from astribot_sdk.core.astribot_api.astribot_client import Astribot

    topics = {
        "head_camera": args.head_topic,
        "left_wrist_camera": args.left_wrist_topic,
        "right_wrist_camera": args.right_wrist_topic,
    }
    rclpy.init()
    node = rclpy.create_node("openwam_wuji_client")
    cameras = CameraStreams(node, topics)
    hands = None
    robot = None
    transport = PolicyTransport(args.server, timeout=args.timeout)
    try:
        left_init = None
        if args.hold_left_init:
            from .left_init import read_left_init
            left_init = read_left_init()
            if args.ignore_left_thumb:
                args.left_thumb_state = tuple(float(v) for v in left_init[2][:4])
        cameras.start()
        if args.ignore_left_thumb:
            from .emergency_left_thumb import EmergencyWujiHands

            hands_class = EmergencyWujiHands
        else:
            hands_class = WujiHands
        hand_options = dict(
            model=args.hand_model,
            left_serial=args.left_hand_serial,
            right_serial=args.right_hand_serial,
            enable_commands=args.enable_hands,
            effort_limit=args.hand_effort_limit,
            lowpass_hz=args.hand_lowpass_hz,
            mit_kp=args.hand_mit_kp,
            mit_kd=args.hand_mit_kd,
            max_step=args.max_hand_step,
            timeout=args.warmup_timeout,
        )
        if args.ignore_left_thumb:
            hand_options["thumb_state"] = args.left_thumb_state
            hand_options["thumb_state_mode"] = args.left_thumb_state_mode
        hands = hands_class(**hand_options)
        robot = Astribot(freq=args.control_hz, high_control_rights=args.enable_arms)
        if args.enable_arms and args.control_way == "filter":
            robot.set_filter_parameters(args.filter_scale, args.filter_scale)
        if args.move_home:
            logger.info("moving Astribot to home pose")
            robot.move_to_home()
        controller = RobotController(
            robot,
            hands,
            enable_arms=args.enable_arms,
            eef_frame=args.eef_frame,
            control_way=args.control_way,
            use_wbc=args.use_wbc,
            workspace_radius=np.asarray(args.workspace_radius),
            max_eef_step=args.max_eef_step,
            max_rotation_step_rad=args.max_rotation_step_rad,
        )
        if left_init is not None:
            from .left_init import left_init_pose
            controller.hold_left_init(
                left_init_pose(robot, left_init[0], left_init[1], eef_frame=args.eef_frame),
                left_init[2],
            )
            logger.warning("Left arm/hand init hold enabled; left policy commands will be replaced")

        cameras.wait_ready(args.warmup_timeout)
        contract = transport.ping_and_validate()
        transport.reset()
        logger.info("server contract validated: %s", contract)
        logger.info(
            "client ready: arms=%s hands=%s; press Space to stop",
            "ENABLED" if args.enable_arms else "dry-run",
            "ENABLED" if args.enable_hands else "dry-run",
        )

        chunk_index = 0
        period = 1.0 / args.control_hz
        with KeyboardStop() as stop:
            while not stop.triggered() and (args.max_chunks == 0 or chunk_index < args.max_chunks):
                images, sequences = cameras.snapshot(args.max_camera_age)
                state = controller.state_vector()
                if args.ignore_left_thumb and args.left_thumb_state_mode == "action":
                    logger.info("left thumb virtual state for inference: %s", state[9:13])
                request_started = time.monotonic()
                response = transport.request_chunk(
                    images=images,
                    prompt=args.task,
                    state=state,
                    count=args.execution_horizon,
                )
                roundtrip_ms = (time.monotonic() - request_started) * 1000.0
                actions = _validated_actions(response, args.execution_horizon)
                logger.info(
                    "chunk=%d frames=%s server_ms=%.2f roundtrip_ms=%.2f",
                    chunk_index,
                    sequences,
                    float(response.get("latency_ms", 0.0)),
                    roundtrip_ms,
                )

                deadline = time.monotonic()
                for action in actions:
                    if stop.triggered():
                        break
                    if args.ignore_left_thumb:
                        hands.check_health()
                    controller.execute(action)
                    deadline += period
                    remaining = deadline - time.monotonic()
                    if remaining > 0.0:
                        time.sleep(remaining)
                chunk_index += 1
    finally:
        transport.close()
        if robot is not None and args.enable_arms:
            try:
                robot.stop_robot()
            except Exception:
                pass
        if hands is not None:
            hands.close()
        cameras.stop()
        try:
            rclpy.shutdown()
        except Exception:
            pass
