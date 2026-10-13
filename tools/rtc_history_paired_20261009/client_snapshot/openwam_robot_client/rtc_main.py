"""Independent CLI for TrainingRTC; synchronous main/runtime are not invoked."""
from __future__ import annotations

import logging
import signal
from pathlib import Path

from .config import build_parser, validate_args
from .rtc_runtime import run


def parse_args(argv=None, *, configure_parser=None):
    parser = build_parser()
    parser.description = "OpenWAM Astribot + Wuji TrainingRTC client"
    parser.set_defaults(execution_horizon=16)
    parser.add_argument("--rtc-prefix-steps", type=int, default=10)
    parser.add_argument("--rtc-warmup-rounds", type=int, default=2,
                        help="Each round warms both no-prefix and prefix paths before action publication")
    parser.add_argument("--max-tick-lag-ms", type=float, default=50.,
                        help="Stop instead of issuing catch-up commands if the control loop is late")
    parser.add_argument("--rtc-hand-tracking-tolerance", type=float, default=0.5,
                        help="Maximum live hand target error in radians; independent of max-hand-step")
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument("--rtc-diagnostics-file", default=None,
                             help="Write TrainingRTC JSONL timing trace to this file")
    diagnostics.add_argument("--rtc-diagnostics-dir", nargs="?", default=None,
                             const=str(Path(__file__).resolve().parent.parent / "logs" / "training_rtc"),
                             help="Write under client/logs/training_rtc by default; optionally specify another root")
    parser.add_argument("--rtc-diagnostics-label", default=None,
                        help="Short task name for the diagnostics directory (default: first word of --task)")
    parser.add_argument("--rtc-state-trace", action="store_true",
                        help="Passively log existing ROS feedback in a separate process; requires diagnostics")
    if configure_parser is not None:
        configure_parser(parser)
    args = validate_args(parser, parser.parse_args(argv))
    if not 0 < args.rtc_prefix_steps < args.execution_horizon:
        parser.error("Require 0 < rtc-prefix-steps < execution-horizon")
    if args.rtc_warmup_rounds < 1 or not 0 < args.max_tick_lag_ms < float("inf"):
        parser.error("Require warmup rounds >= 1 and a finite positive tick lag limit")
    if not 0 < args.rtc_hand_tracking_tolerance <= 3.14:
        parser.error("Require 0 < rtc-hand-tracking-tolerance <= 3.14 radians")
    if args.rtc_diagnostics_label and not args.rtc_diagnostics_dir:
        parser.error("--rtc-diagnostics-label requires --rtc-diagnostics-dir")
    if args.rtc_state_trace and not (args.rtc_diagnostics_dir or args.rtc_diagnostics_file):
        parser.error("--rtc-state-trace requires --rtc-diagnostics-dir or --rtc-diagnostics-file")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    def stop(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        run(args)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("RTC stopped")


if __name__ == "__main__":
    main()
