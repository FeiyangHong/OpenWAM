from __future__ import annotations

"""Capture one read-only observation for policy debugging."""

import argparse
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import rclpy
from PIL import Image

from openwam_robot_client.cameras import CameraStreams
from openwam_robot_client.geometry import pose7_to_eef9_column
from openwam_robot_client.hands import WujiHands


DEFAULT_PROMPT = "Pick up the spray bottle, pump it to build up pressure, then spray water on the flowers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("debug_input"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-camera-age", type=float, default=1.0)
    parser.add_argument("--eef-frame", choices=("chassis", "world"), default="chassis")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    from astribot_sdk.core.astribot_api.astribot_client import Astribot

    rclpy.init(args=[])
    node = rclpy.create_node("openwam_debug_export")
    cameras = CameraStreams(node)
    hands = None
    robot = None
    try:
        cameras.start()
        robot = Astribot(freq=30.0, high_control_rights=False, node_name="openwam_debug_export")
        hand_error = None
        try:
            hands = WujiHands(enable_commands=False, timeout=args.timeout)
        except Exception as exc:
            hand_error = f"Wuji hand state unavailable: {exc}"
        cameras.wait_ready(args.timeout)
        images, sequences = cameras.snapshot(args.max_camera_age)
        arm_names = [robot.arm_left_name, robot.arm_right_name]
        arm_poses = robot.get_current_cartesian_pose(arm_names, frame=args.eef_frame)
        arm_state = [value for pose in arm_poses for value in pose7_to_eef9_column(pose).tolist()]
        state = None
        if hands is not None:
            state = arm_state[:9] + hands.left.get().tolist() + arm_state[9:] + hands.right.get().tolist()

        image_names = {
            "head_camera": "head.png",
            "left_wrist_camera": "left_wrist.png",
            "right_wrist_camera": "right_wrist.png",
        }
        for camera_name, filename in image_names.items():
            with Image.open(BytesIO(images[camera_name])) as image:
                image.convert("RGB").save(output / filename, format="PNG")

        metadata = {
            "prompt": args.prompt,
            "state": state,
            "state_dim": None if state is None else len(state),
            "state_layout": "left_eef9,left_hand20,right_eef9,right_hand20",
            "arm_state": arm_state,
            "hand_state_available": hands is not None,
            "state_error": hand_error,
            "representation": "eef",
            "rot6d_convention": "column",
            "eef_frame": args.eef_frame,
            "camera_sequences": sequences,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        (output / "state.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"exported debug input to {output}")
        return 0
    finally:
        if hands is not None:
            hands.close()
        cameras.stop()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
