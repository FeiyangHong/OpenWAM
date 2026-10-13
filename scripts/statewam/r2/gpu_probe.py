"""Bounded CUDA lifecycle probe; not model profiling or training recovery evidence."""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.common import atomic_json
from scripts.statewam.r2.cpu_job import identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    if not 0 < args.seconds <= 120:
        raise ValueError("Probe duration must be in (0, 120] seconds")
    directory = Path(os.environ["R2_RUN_DIR"])
    stop = Path(os.environ["R2_STOP_FILE"])
    if torch.cuda.device_count() != 1:
        raise ValueError("This probe requires exactly one visible GPU")
    started = time.monotonic()
    record = {"process": identity(os.getpid()), "heartbeat": time.time(), "updates": 0,
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
    if stop.exists():
        atomic_json(directory / "probe_result.json", {**record, "stopped_cooperatively": True,
                                                      "elapsed_seconds": time.monotonic() - started})
        return
    x = torch.ones((1024, 1024), dtype=torch.bfloat16, device="cuda")
    updates = 0
    while time.monotonic() - started < args.seconds and not stop.exists():
        y = x @ x
        torch.cuda.synchronize()
        if not torch.isfinite(y).all():
            raise ValueError("Nonfinite CUDA output")
        updates += 1
        record = {"process": identity(os.getpid()), "heartbeat": time.time(),
                  "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                  "device_name": torch.cuda.get_device_name(), "updates": updates,
                  "allocated_bytes": torch.cuda.memory_allocated(),
                  "reserved_bytes": torch.cuda.memory_reserved(),
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        atomic_json(directory / "probe_progress.json", record)
        time.sleep(0.25)
    atomic_json(directory / "probe_result.json", {**record, "stopped_cooperatively": stop.exists(),
                                                  "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
