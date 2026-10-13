from contextlib import contextmanager

import pytest

from scripts.statewam.r2.profile_metrics import StepTimings, distribution, summarize_steps


def test_distribution_small_samples_and_invalid_values():
    assert distribution([1, 3])["p50"] == 2
    assert distribution([1, 3])["p95"] == pytest.approx(2.9)
    for values in ([], [float("nan")], [-1]):
        with pytest.raises(ValueError):
            distribution(values)


def test_profile_summary_accumulation_warmup_and_global_exposures():
    row = {"microstep_wall_seconds": 2, "wait_seconds": 0.1,
           "peak_allocated_bytes": 50, "peak_reserved_bytes": 60,
           "phase_timings": {"forward": {"cuda_event_seconds": 0.5, "host_seconds": 0.2}}}
    rows = [row] * 8
    result = summarize_steps(rows, accumulation=2, micro_batch=4, world_size=4, warmup_updates=1)
    assert result["optimizer_updates"] == 4
    assert result["optimizer_update_wall_seconds_excluding_save"]["mean"] == 4
    assert result["optimizer_update_wall_seconds_excluding_save"]["count"] == 3
    assert result["global_samples_per_second_excluding_save"] == 8
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_steps(rows[:-1], accumulation=2, micro_batch=4, world_size=4)
    with pytest.raises(ValueError, match="No steady"):
        summarize_steps(rows, accumulation=2, micro_batch=4, world_size=4)


def test_engine_timing_wraps_actual_backward_and_step_and_restores_on_error(monkeypatch):
    phases = []
    @contextmanager
    def during(name):
        phases.append(name)
        yield

    class Engine:
        def backward(self, loss, *, scale):
            return loss * scale

        def step(self):
            raise RuntimeError("test failure")

    engine, timings = Engine(), StepTimings()
    monkeypatch.setattr(timings, "during", during)
    with pytest.raises(RuntimeError, match="test failure"):
        with timings.engine_phases(engine):
            assert engine.backward(3, scale=2) == 6
            engine.step()
    assert phases == ["engine_backward", "engine_step"]
    assert "step" not in vars(engine) and "backward" not in vars(engine)
