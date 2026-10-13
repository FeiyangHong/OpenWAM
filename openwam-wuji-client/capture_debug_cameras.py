from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import rclpy
from PIL import Image

from openwam_robot_client.cameras import CameraStreams


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture the three current compressed camera streams as PNG")
    parser.add_argument("--output", type=Path, default=Path("debug_input"))
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rclpy.init(args=[])
    node = rclpy.create_node("openwam_debug_camera_export")
    cameras = CameraStreams(node)
    try:
        cameras.start()
        cameras.wait_ready(args.timeout)
        images, _sequences = cameras.snapshot(max_age=1.0)
        for camera_name, filename in {
            "head_camera": "head.png",
            "left_wrist_camera": "left_wrist.png",
            "right_wrist_camera": "right_wrist.png",
        }.items():
            with Image.open(BytesIO(images[camera_name])) as image:
                image.convert("RGB").save(args.output / filename, format="PNG")
        print(f"exported cameras to {args.output}")
        return 0
    finally:
        cameras.stop()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
