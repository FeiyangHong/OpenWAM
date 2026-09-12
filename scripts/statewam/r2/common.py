"""Small provenance and atomic-output helpers for the R2 pilot."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GOAL_ID = "r2/robotwin-pilot"
REVISION = "785feb15aa4a4f532395ad2b1d2be5f28cb561ad"
REPO_ID = "TianxingChen/RoboTwin2.0"
VARIANT = "aloha-agilex_clean_50"
TASKS = tuple(sorted((
    "adjust_bottle", "pick_dual_bottles", "handover_block", "lift_pot",
    "stack_blocks_two", "stack_bowls_two", "place_mouse_pad",
    "place_object_basket", "open_microwave", "press_stapler",
)))
ARTIFACTS = ROOT / "statewam_artifacts/robotwin/r2-robotwin-pilot"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
