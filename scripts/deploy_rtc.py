"""Separate launcher for the TrainingRTC chunk server."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party"))

from openwam.deploy.rtc_server import main

if __name__ == "__main__":
    main()
