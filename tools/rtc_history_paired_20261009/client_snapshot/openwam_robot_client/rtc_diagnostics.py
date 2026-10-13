"""Opt-in, asynchronous TrainingRTC timing, state and action trace."""
from __future__ import annotations

import json
import itertools
import os
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path


_run_number = itertools.count(1)


def run_log_path(root, task, label=None):
    """Choose a dated run file with a short task label and no prompt hash."""
    if label is None:
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9]*|[\u4e00-\u9fff]+", task)
        label = next((word for word in words if word.lower() not in
                      {"a", "an", "the", "please", "to"}), "task")
        if re.fullmatch(r"[\u4e00-\u9fff]+", label):
            label = label[:4]
    label = re.sub(r"[^\w-]+", "_", label.lower(), flags=re.UNICODE).strip("_-")[:24] or "task"
    now = datetime.now().astimezone()
    filename = f"{now:%H%M%S_%f}_pid{os.getpid()}_run{next(_run_number):04d}.jsonl"
    return Path(root).expanduser() / label / f"{now:%Y-%m-%d}" / filename


class RTCDiagnostics:
    def __init__(self, path, *, exclusive=False):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._output = self.path.open("x" if exclusive else "a", encoding="utf-8")
        self._events = queue.Queue(maxsize=16384)
        self._dropped = 0
        self._thread = threading.Thread(target=self._write, name="rtc-diagnostics", daemon=True)
        self._thread.start()

    def record(self, event, **fields):
        row = {"event": event, "monotonic_ns": time.monotonic_ns(),
               "wall_time_ns": time.time_ns(), **fields}
        try:
            self._events.put_nowait(row)
        except queue.Full:
            self._dropped += 1

    def _write(self):
        with self._output as output:
            while True:
                row = self._events.get()
                if row is None:
                    break
                output.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                output.flush()

    def close(self):
        self.record("diagnostics_end", dropped_events=self._dropped)
        try:
            self._events.put_nowait(None)
        except queue.Full:
            try:
                self._events.get_nowait()
            except queue.Empty:
                pass
            self._events.put_nowait(None)
        self._thread.join(timeout=5)
