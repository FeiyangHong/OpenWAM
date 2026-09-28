from __future__ import annotations

import logging
import signal

from .config import parse_args
from .runtime import run


def _stop_on_signal(_signum, _frame) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    signal.signal(signal.SIGTERM, _stop_on_signal)
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("interrupted; stopping")


if __name__ == "__main__":
    main()
