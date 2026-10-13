"""Independent CLI for TrainingRTC; synchronous main/runtime are not invoked."""
from __future__ import annotations

import logging
import signal

from .config import build_parser, validate_args
from .rtc_runtime import run


def parse_args(argv=None):
    parser = build_parser()
    parser.description = "OpenWAM Astribot + Wuji TrainingRTC client"
    parser.set_defaults(execution_horizon=16)
    parser.add_argument("--rtc-prefix-steps", type=int, default=10)
    parser.add_argument("--rtc-warmup-rounds", type=int, default=2,
                        help="Each round warms both no-prefix and prefix paths before action publication")
    parser.add_argument("--max-tick-lag-ms", type=float, default=50.,
                        help="Stop instead of issuing catch-up commands if the control loop is late")
    args = validate_args(parser, parser.parse_args(argv))
    if not 0 < args.rtc_prefix_steps < args.execution_horizon:
        parser.error("Require 0 < rtc-prefix-steps < execution-horizon")
    if args.rtc_warmup_rounds < 1 or not 0 < args.max_tick_lag_ms < float("inf"):
        parser.error("Require warmup rounds >= 1 and a finite positive tick lag limit")
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
