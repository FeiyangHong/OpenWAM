"""TrainingRTC execution with a fixed, committed prefix and one worker.

One predict_action call means one consumed action tick. At request time the
remaining d commands are frozen and become the clean prefix of the new chunk.
Only its [d:d+s] suffix is adopted. If late, wait at the boundary instead of
executing unconditioned commands or skipping additional predicted actions.
"""
import copy
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TrainingRTCExecutor:
    def __init__(self, engine, inference_horizon, inference_delay_steps):
        self.engine = engine
        self.horizon = inference_horizon
        self.delay = inference_delay_steps
        if self.horizon is None or self.delay is None or not 0 < self.delay < self.horizon:
            raise ValueError("TrainingRTC requires explicit 0 < inference_delay_steps < inference_horizon")
        config = getattr(getattr(engine, "architecture", None), "training_rtc", None)
        if config is None or not config.enabled or self.delay > config.max_prefix_steps:
            raise ValueError("TrainingRTC delay exceeds the checkpoint trained prefix limit or checkpoint lacks RTC")
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training-rtc")
        self._lock = threading.Lock()
        self._buffer = deque()
        self._pending = None
        self._closed = False
        self.late_chunks = 0

    def _generate(self, conditions):
        with torch.no_grad():
            result = self.engine.generate(conditions)
        actions = np.array(result["actions"], copy=True)
        if actions.ndim != 2 or not np.isfinite(actions).all():
            raise ValueError("RTC engine must return finite [H,D] actions")
        if len(actions) < self.delay + self.horizon:
            raise ValueError("RTC requires generated H >= inference_delay_steps + inference_horizon")
        # The committed prefix must match the final commands sent by WAMPolicy.
        dims = getattr(getattr(self.engine, "architecture", None), "binary_command_dims", ()) or ()
        if dims:
            actions[:, dims] = np.where(actions[:, dims] > 0.5, 1., -1.)
        return actions

    def predict_action(self, conditions):
        with self._lock:
            if self._closed:
                raise RuntimeError("TrainingRTC executor is shut down")
            if not self._buffer:
                if self._pending is None:
                    actions = self._generate(conditions)
                    self._buffer.extend(actions[:self.horizon])
                else:
                    future = self._pending
                    self._pending = None
                    if not future.done():
                        self.late_chunks += 1
                        logger.warning("TrainingRTC chunk late; waiting at the committed-prefix boundary")
                    actions = future.result()
                    self._buffer.extend(actions[self.delay:self.delay + self.horizon])
            if self._pending is None and len(self._buffer) == self.delay:
                frozen = copy.deepcopy(conditions)
                frozen["rtc_prefix"] = np.stack(list(self._buffer)).copy()
                self._pending = self._pool.submit(self._generate, frozen)
            # Never return a view into a future prefix buffer.
            return np.array(self._buffer.popleft(), copy=True)

    def reset(self):
        with self._lock:
            self._buffer.clear()
            future, self._pending = self._pending, None
            if future is not None:
                # Drain before a new episode: no concurrent engine use or stale adoption.
                try:
                    future.result()
                except Exception:
                    logger.exception("Discarding failed RTC generation during reset")
            self.late_chunks = 0

    def shutdown(self):
        self.reset()
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=True)
