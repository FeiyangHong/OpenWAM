from copy import deepcopy

import pytest

from scripts.statewam.r2.capacity_report import comparison


def groups():
    records = {}
    for group in ("E0", "E1", "E2", "E3"):
        source = {"phase": "forward", "video_shape": [1, 120 if group == "E1" else 360, 3072],
                  "context_shape": [1, 513, 3072], "action_source_shape": [1, 32, 1024],
                  "writer_shape": [1, 32, 1024], "video_valid": [120], "context_valid": [33],
                  "action_valid": [11 if group == "E3" else 32]}
        rank = {"parameter_counts": {"trainable": 100}, "microsteps": 2,
                "state_source_histogram": [] if group == "E0" else [source],
                "instrumented_compute_seconds": [1, 2], "peak_allocated_bytes": 32, "peak_reserved_bytes": 40,
                "attention_totals_across_microsteps": {"forward/state": {"calls": 4, "dense_pairs": 64}}}
        records[group] = {"ranks": [deepcopy(rank) for _ in range(4)], "identity": {"micro_batch": 1},
                          "wall_seconds": 10, "allocated_gpu_hours": 4 / 360, "save_seconds_by_rank": [2] * 4}
    return records


def test_comparison_normalizes_per_sample_without_hiding_source_or_mask_differences():
    result = comparison(groups())
    assert result["E1"]["source_token_counts"]["world"] == [120]
    assert result["E2"]["source_token_counts"]["world"] == [360]
    assert result["E1"]["valid_token_counts"]["action_valid"] == [32]
    assert result["E3"]["valid_token_counts"]["action_valid"] == [11]
    assert result["E3"]["sdpa_per_sample"]["forward/state"]["dense_pairs"] == 32
    assert result["E2"]["source_semantics"]["action_reader"] != result["E3"]["source_semantics"]["action_reader"]


def test_comparison_rejects_wrong_source_geometry_on_any_rank():
    records = groups()
    records["E2"]["ranks"][2]["state_source_histogram"][0]["video_shape"][1] = 120
    with pytest.raises(ValueError, match="Unexpected source dimensions: E2"):
        comparison(records)
