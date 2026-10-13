"""Readable comparison of audited real-model capacity and SDPA workload."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.audit_workload import GROUPS, audit
from scripts.statewam.r2.common import atomic_json

SOURCES = {
    "E0": {"world_reader": None, "action_reader": None},
    "E1": {"world_reader": "current_frame_only", "action_reader": "masked_current_frame_resampling"},
    "E2": {"world_reader": "current_and_noisy_generated_future", "action_reader": "masked_current_frame_resampling"},
    "E3": {"world_reader": "current_and_noisy_generated_future", "action_reader": "noisy_action_hidden_tokens"},
}


def comparison(groups):
    rows = {}
    for group in GROUPS:
        record = groups[group]
        ranks = record["ranks"]
        sources = [source for rank in ranks for source in rank["state_source_histogram"] if source["phase"] == "forward"]
        row = {"parameter_counts": ranks[0]["parameter_counts"], "source_semantics": SOURCES[group],
               "source_token_counts": {name: sorted({source[field][1] for source in sources if source[field]})
                                       for name, field in (("world", "video_shape"), ("language_proprio", "context_shape"),
                                                           ("auxiliary_action_reader", "action_source_shape"), ("action_writer", "writer_shape"))},
               "valid_token_counts": {name: sorted({value for source in sources for value in source[name]})
                                      for name in ("video_valid", "context_valid", "action_valid")},
               "instrumented_compute_seconds_by_rank": [rank["instrumented_compute_seconds"] for rank in ranks],
               "peak_allocated_bytes_by_rank": [rank["peak_allocated_bytes"] for rank in ranks],
               "peak_reserved_bytes_by_rank": [rank["peak_reserved_bytes"] for rank in ranks],
               "wall_seconds": record["wall_seconds"], "allocated_gpu_hours": record["allocated_gpu_hours"],
               "save_seconds_by_rank": record["save_seconds_by_rank"], "sdpa_per_sample": {}}
        samples = sum(rank["microsteps"] for rank in ranks) * record["identity"]["micro_batch"]
        for phase in ("preprocessing", "forward", "backward_recomputation"):
            for region in ("state", "backbone_or_preprocessing"):
                key = f"{phase}/{region}"
                totals = [rank["attention_totals_across_microsteps"].get(key, {}) for rank in ranks]
                row["sdpa_per_sample"][key] = {field: sum(value.get(field, 0) for value in totals) / samples
                                                for field in ("calls", "dense_pairs", "allowed_pairs", "dense_qk_av_flops", "allowed_qk_av_flops")}
        rows[group] = row
    # These are the fixed Foundation/RoboTwin token dimensions observed in this audit.
    for group, world in (("E1", 120), ("E2", 360), ("E3", 360)):
        if rows[group]["source_token_counts"] != {"world": [world], "language_proprio": [513],
                                                    "auxiliary_action_reader": [32], "action_writer": [32]}:
            raise ValueError(f"Unexpected source dimensions: {group}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="GROUP=run-directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous capacity evidence")
    runs = dict(row.split("=", 1) for row in args.run)
    if len(runs) != len(args.run):
        raise ValueError("Duplicate group")
    result = audit({group: Path(path) for group, path in runs.items()})
    result["comparison"] = comparison(result["groups"])
    result["capacity_decision"] = {
        "parameter_control": "E1/E2 use the same trainable State projections/readers/writers as E3, driven only by pooled current visual tokens in the auxiliary reader",
        "effective_state_parameters_matched": True,
        "compute_matched": False,
        "main_comparison": "optimizer-update and sample-exposure matched; report actual runtime/memory separately",
        "remaining_compute_differences": ["E1 reads 120 current tokens instead of 360 world tokens",
                                          "E1/E2 auxiliary pooling adds non-SDPA work; their validity masks can differ from E3 action masks",
                                          "SDPA arithmetic excludes projections/FFN/softmax and actual backward kernels"],
        "interpretation": "Parameter participation supports this pilot capacity control, not a proof that information source is the sole cause of an outcome. E3-E2 only isolates the added direct read edge; the shared mutual backbone can already carry action information.",
        "pending_before_formal_training": ["200-500 update profiling", "final recipe recovery coverage", "frozen protocol and paired scene/evaluation gates"],
    }
    atomic_json(args.output, result)
    print(json.dumps({"workload_audit_passed": result["workload_audit_passed"], "compute_matched": False,
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
