"""Asynchronous persistence for decoded deployment videos."""

from __future__ import annotations

import json
import logging
import queue
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DecodedVideoSaver:
    """Write one MP4 for each model generation without blocking control."""

    def __init__(self, output_dir: str, fps: float = 7.5, queue_size: int = 8):
        if fps <= 0:
            raise ValueError("video_fps must be positive")
        self.run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
        self.root = Path(output_dir).expanduser() / self.run_id
        self.fps = float(fps)
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop = object()
        self._closed = False
        self._lock = threading.Lock()
        self._session_episodes: dict[str, int] = {}
        self._session_predictions: dict[str, int] = {}
        self._thread = threading.Thread(target=self._worker, name="openwam-video-saver", daemon=True)
        self._thread.start()

    def reset(self, session_id: str) -> int:
        with self._lock:
            episode = self._session_episodes.get(session_id, 0) + 1
            self._session_episodes[session_id] = episode
            self._session_predictions[session_id] = 0
            return episode

    def submit(self, frames, *, session_id: Optional[str] = None, episode_id: Optional[int] = None) -> None:
        if not frames:
            return
        session = session_id or "direct"
        with self._lock:
            episode = episode_id if episode_id is not None else self._session_episodes.get(session, 0)
            prediction = self._session_predictions.get(session, 0) + 1
            self._session_predictions[session] = prediction
        try:
            self._queue.put_nowait((session, episode, prediction, list(frames)))
        except queue.Full:
            logger.warning("Decoded video queue is full; dropping prediction %s/%s", session, prediction)

    def _worker(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is self._stop:
                    return
                session, episode, prediction, frames = task
                self._write(session, episode, prediction, frames)
            except Exception:
                logger.exception("Failed to save decoded prediction video")
            finally:
                self._queue.task_done()

    def _write(self, session: str, episode: int, prediction: int, frames) -> None:
        import imageio.v2 as imageio

        directory = self.root / session / f"episode_{episode:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"prediction_{prediction:06d}.mp4"
        temporary = target.with_suffix(".tmp.mp4")
        with imageio.get_writer(str(temporary), fps=self.fps, codec="libx264") as writer:
            for frame in frames:
                if hasattr(frame, "convert"):
                    frame = frame.convert("RGB")
                writer.append_data(np.asarray(frame))
        temporary.replace(target)

        record = {
            "session_id": session,
            "episode_id": episode,
            "prediction_id": prediction,
            "path": str(target.relative_to(self.root)),
            "frames": len(frames),
            "fps": self.fps,
        }
        with (self.root / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.join()
        self._queue.put(self._stop)
        self._thread.join()
