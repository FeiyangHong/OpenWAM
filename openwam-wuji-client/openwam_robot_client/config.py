from __future__ import annotations

import argparse
import math

from .cameras import CAMERA_TOPICS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenWAM Astribot + Wuji real-robot client")
    parser.add_argument("--server", default="ws://192.168.0.100:8848")
    parser.add_argument("--task", required=True, help="Prompt used when training the checkpoint")
    parser.add_argument(
        "--execution-horizon",
        "--chunk-size",
        dest="execution_horizon",
        type=int,
        default=32,
        help="Actions executed from each observation (training action horizon: 32)",
    )
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--warmup-timeout", type=float, default=10.0)
    parser.add_argument("--max-camera-age", type=float, default=0.25)
    parser.add_argument("--max-chunks", type=int, default=0, help="0 runs until stopped")
    parser.add_argument("--eef-frame", choices=("chassis", "world"), default="chassis")
    parser.add_argument("--control-way", choices=("direct", "filter"), default="filter")
    parser.add_argument("--filter-scale", type=float, default=0.35)
    parser.add_argument("--use-wbc", action="store_true")
    parser.add_argument("--workspace-radius", type=float, nargs=3, default=(0.4, 0.4, 0.4))
    parser.add_argument("--max-eef-step", type=float, default=0.03)
    parser.add_argument("--max-rotation-step-deg", type=float, default=10.0)
    parser.add_argument("--enable-arms", action="store_true")
    parser.add_argument("--enable-hands", action="store_true")
    parser.add_argument("--move-home", action="store_true")
    parser.add_argument("--hand-model", choices=("auto", "wuji_hand", "wuji_hand2"), default="auto")
    parser.add_argument("--left-hand-serial", default="")
    parser.add_argument("--right-hand-serial", default="")
    parser.add_argument("--hand-effort-limit", type=float, default=1.5)
    parser.add_argument("--hand-lowpass-hz", type=float, default=5.0)
    parser.add_argument("--hand-mit-kp", type=float, default=3.0)
    parser.add_argument("--hand-mit-kd", type=float, default=0.05)
    parser.add_argument("--max-hand-step", type=float, default=0.1)
    parser.add_argument("--head-topic", default=CAMERA_TOPICS["head_camera"])
    parser.add_argument("--left-wrist-topic", default=CAMERA_TOPICS["left_wrist_camera"])
    parser.add_argument("--right-wrist-topic", default=CAMERA_TOPICS["right_wrist_camera"])
    return parser


def validate_args(parser, args):

    if not 1 <= args.execution_horizon <= 64:
        parser.error("--execution-horizon must be in [1, 64]")
    for name in ("control_hz", "timeout", "warmup_timeout", "max_camera_age"):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "filter_scale",
        "max_eef_step",
        "max_rotation_step_deg",
        "hand_effort_limit",
        "hand_lowpass_hz",
        "hand_mit_kp",
        "hand_mit_kd",
        "max_hand_step",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.max_chunks < 0:
        parser.error("--max-chunks must be non-negative")
    if any(radius <= 0.0 for radius in args.workspace_radius):
        parser.error("--workspace-radius entries must be positive")
    if args.move_home and not args.enable_arms:
        parser.error("--move-home requires --enable-arms")
    args.max_rotation_step_rad = math.radians(args.max_rotation_step_deg)
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return validate_args(parser, parser.parse_args(argv))
