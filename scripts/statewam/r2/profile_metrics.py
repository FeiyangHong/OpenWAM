"""Bounded training timings with DeepSpeed backward/update attribution."""

from contextlib import contextmanager
import math
import statistics
import time

import torch


class StepTimings:
    def __init__(self):
        self.pending = []

    @contextmanager
    def during(self, name):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = time.perf_counter()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.pending.append((name, start, end, time.perf_counter() - before))

    def resolve(self):
        # Caller synchronizes once at the end of the microstep.
        result = {}
        for name, start, end, wall in self.pending:
            row = result.setdefault(name, {"cuda_event_seconds": 0.0, "host_seconds": 0.0, "calls": 0})
            row["cuda_event_seconds"] += start.elapsed_time(end) / 1000
            row["host_seconds"] += wall
            row["calls"] += 1
        self.pending.clear()
        return result

    @contextmanager
    def engine_phases(self, engine):
        original = {name: getattr(engine, name) for name in ("backward", "step")}
        had_instance_value = {name: name in vars(engine) for name in original}
        for name, method in original.items():
            def timed(*args, method=method, name=name, **kwargs):
                with self.during(f"engine_{name}"):
                    return method(*args, **kwargs)
            setattr(engine, name, timed)
        try:
            yield
        finally:
            for name, method in original.items():
                if had_instance_value[name]:
                    setattr(engine, name, method)
                else:
                    delattr(engine, name)


def distribution(values):
    values = sorted(values)
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Require finite nonnegative timing samples")
    def quantile(q):
        position = (len(values) - 1) * q
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)
    return {"count": len(values), "mean": statistics.mean(values), "p50": quantile(0.5),
            "p95": quantile(0.95), "max": values[-1], "sum": sum(values)}


def timing_sample(row):
    """Retain only summary inputs; large gradient/sample evidence stays on disk."""
    return {key: row[key] for key in ("microstep_wall_seconds", "phase_timings", "wait_seconds",
                                     "peak_allocated_bytes", "peak_reserved_bytes")}


def summarize_steps(rows, *, accumulation, micro_batch, world_size, warmup_updates=10):
    if accumulation < 1 or min(micro_batch, world_size) < 1 or len(rows) % accumulation:
        raise ValueError("Incomplete optimizer-update timing window")
    if len(rows) <= warmup_updates * accumulation:
        raise ValueError("No steady samples after warmup")
    steady = rows[warmup_updates * accumulation:]
    windows = [steady[index:index + accumulation] for index in range(0, len(steady), accumulation)]
    updates = [sum(row["microstep_wall_seconds"] for row in window) for window in windows]
    phases = sorted({key for row in steady for key in row["phase_timings"]})
    result = {"warmup_updates_excluded": warmup_updates, "microsteps": len(rows),
              "optimizer_updates": len(rows) // accumulation,
              "microstep_wall_seconds": distribution([row["microstep_wall_seconds"] for row in steady]),
              "optimizer_update_wall_seconds_excluding_save": distribution(updates),
              "global_samples_per_second_excluding_save": len(steady) * micro_batch * world_size / sum(updates),
              "data_wait_seconds": distribution([row["wait_seconds"] for row in steady]),
              "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rows),
              "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in rows),
              "phases": {}}
    for phase in phases:
        values = [row["phase_timings"][phase] for row in steady if phase in row["phase_timings"]]
        result["phases"][phase] = {key: distribution([row[key] for row in values])
                                   for key in ("cuda_event_seconds", "host_seconds")}
    result["limits"] = ("Rank-local wall times include explicit end-microstep synchronization and audit overhead. "
                        "CUDA event spans are stream-local estimates, not aggregate GPU kernel time. "
                        "DeepSpeed engine_step includes clipping/optimizer/zeroing and any scheduler. "
                        "Nested wrapper scopes must not be summed. Checkpoint times are separate; "
                        "amortized end-to-end throughput requires full run wall-clock.")
    return result
