from copy import deepcopy

import pytest

from scripts.statewam.r2.audit_workload import summarize_rank, validate_source_geometry


def evidence():
    call = {"phase": "forward", "scope": "backbone_or_preprocessing", "dense_pairs": 32,
            "allowed_pairs": 16, "dense_qk_av_flops": 1024, "allowed_qk_av_flops": 512}
    trace = {"parameter_inventory": {"action.weight": {"trainable": True, "numel": 4},
                                     "video.weight": {"trainable": False, "numel": 12}},
             "sdpa_calls": [call, {**call, "phase": "backward_recomputation"}], "state_sources": []}
    steps = [{"gradients": {"action.weight": [0.0, 1]}, "compute_seconds": 2,
              "peak_allocated_bytes": 64, "peak_reserved_bytes": 96},
             {"gradients": {"action.weight": [1.0, 1]}, "compute_seconds": 1,
              "peak_allocated_bytes": 80, "peak_reserved_bytes": 96}]
    return trace, steps


def test_counts_separate_gradient_activity_from_total_and_recomputation():
    trace, steps = evidence()
    result = summarize_rank(trace, steps)
    assert result["parameter_counts"]["total"] == 16
    assert result["parameter_counts"]["trainable"] == result["parameter_counts"]["backward_participating"] == 4
    assert result["parameter_counts"]["ever_nonzero_gradient"] == 4
    assert result["attention_totals_across_microsteps"]["forward/backbone_or_preprocessing"]["allowed_pairs"] == 16
    assert result["attention_totals_across_microsteps"]["backward_recomputation/backbone_or_preprocessing"]["calls"] == 1
    validate_source_geometry("E0", result)


@pytest.mark.parametrize("gradient", [[float("nan"), 1], [float("inf"), 1], [1, 0], [-1, 1]])
def test_reject_invalid_gradient_evidence(gradient):
    trace, steps = evidence()
    steps[1]["gradients"]["action.weight"] = gradient
    with pytest.raises(ValueError, match="Invalid gradient"):
        summarize_rank(trace, steps)


def test_reject_frozen_gradient_and_missing_recomputation():
    trace, steps = evidence()
    steps[0]["gradients"]["video.weight"] = [0, 1]
    with pytest.raises(ValueError, match="Frozen"):
        summarize_rank(trace, steps)
    del steps[0]["gradients"]["video.weight"]
    trace["sdpa_calls"].pop()
    with pytest.raises(ValueError, match="Missing forward"):
        summarize_rank(trace, steps)


def test_missing_state_coverage_cannot_pass_as_matched_capacity():
    trace, steps = evidence()
    trace["parameter_inventory"]["interaction_state.layers.4.gate"] = {"trainable": True, "numel": 1}
    with pytest.raises(ValueError, match="State parameters inactive"):
        summarize_rank(trace, steps)
    for step in steps:
        step["gradients"]["interaction_state.layers.4.gate"] = [1, 1]
    with pytest.raises(ValueError, match="State source trace missing"):
        summarize_rank(trace, steps)


def test_source_geometry_requires_all_six_layers_and_both_phases():
    summary = {"microsteps": 2, "state_source_histogram": [
        {"phase": phase, "layer": f"interaction_state.layers.{layer}", "calls": 2,
         "video_shape": [1, 120, 3072], "context_shape": [1, 513, 3072],
         "action_source_shape": [1, 32, 1024], "writer_shape": [1, 32, 1024]}
        for phase in ("forward", "backward_recomputation") for layer in (4, 9, 14, 19, 24, 29)
    ]}
    validate_source_geometry("E1", summary)
    broken = deepcopy(summary)
    broken["state_source_histogram"].pop()
    with pytest.raises(ValueError, match="Missing State layer"):
        validate_source_geometry("E1", broken)
    broken = deepcopy(summary)
    broken["state_source_histogram"][0]["calls"] = 1
    with pytest.raises(ValueError, match="invocation count"):
        validate_source_geometry("E1", broken)
    with pytest.raises(ValueError, match="E0 must not"):
        validate_source_geometry("E0", summary)
