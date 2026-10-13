"""Audit paired real-model workload traces; explicitly not a compute-match claim."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256
from scripts.statewam.r2.cpu_job import alive

GROUPS = ("E0", "E1", "E2", "E3")


def summarize_rank(trace, steps):
    inventory = trace["parameter_inventory"]
    trainable = {name for name, row in inventory.items() if row["trainable"]}
    used, nonzero = set(), set()
    for step in steps:
        gradients = step["gradients"]
        if set(gradients) - trainable:
            raise ValueError("Frozen or unknown parameter received gradient")
        for name, (norm, finite) in gradients.items():
            if finite != 1 or not math.isfinite(norm) or norm < 0:
                raise ValueError("Invalid gradient evidence")
            used.add(name)
            if norm > 0:
                nonzero.add(name)
    state = {name for name in trainable if name.startswith("interaction_state.")}
    if state and not state <= {name for name, value in steps[-1]["gradients"].items() if value[0] > 0}:
        raise ValueError("State parameters inactive after gate warm start")
    count = lambda names: sum(inventory[name]["numel"] for name in names)
    counts = {"total": count(inventory), "trainable": count(trainable),
              "backward_participating": count(used), "ever_nonzero_gradient": count(nonzero),
              "state_trainable": count(state), "state_backward_participating": count(used & state),
              "trainable_without_gradient": sorted(trainable - used),
              "trainable_never_nonzero_gradient": sorted(trainable - nonzero)}
    geometry = Counter()
    totals = defaultdict(Counter)
    for call in trace["sdpa_calls"]:
        phase, scope = call["phase"], call["scope"]
        if phase not in {"preprocessing", "forward", "backward_recomputation"}:
            raise ValueError("Unattributed attention phase")
        if not 0 <= call["allowed_pairs"] <= call["dense_pairs"]:
            raise ValueError("Invalid attention pair count")
        geometry[json.dumps(call, sort_keys=True)] += 1
        region = "state" if scope.startswith("interaction_state.") else "backbone_or_preprocessing"
        key = f"{phase}/{region}"
        totals[key]["calls"] += 1
        for field in ("dense_pairs", "allowed_pairs", "dense_qk_av_flops", "allowed_qk_av_flops"):
            totals[key][field] += call[field]
    if not any(key.startswith("forward/") for key in totals) or not any(key.startswith("backward_recomputation/") for key in totals):
        raise ValueError("Missing forward or recomputation trace")
    sources = Counter(json.dumps(row, sort_keys=True) for row in trace["state_sources"])
    if bool(state) != bool(sources):
        raise ValueError("State source trace missing or unexpected")
    return {"parameter_counts": counts, "microsteps": len(steps),
            "attention_totals_across_microsteps": dict(totals),
            "attention_geometry_histogram": [{**json.loads(row), "calls": count} for row, count in sorted(geometry.items())],
            "state_source_histogram": [{**json.loads(row), "calls": count} for row, count in sorted(sources.items())],
            "instrumented_compute_seconds": [row["compute_seconds"] for row in steps],
            "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in steps),
            "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in steps)}


def validate_source_geometry(group, summary):
    rows = summary["state_source_histogram"]
    if group == "E0":
        if rows:
            raise ValueError("E0 must not read State sources")
        return
    expected_layers = {f"interaction_state.layers.{layer}" for layer in (4, 9, 14, 19, 24, 29)}
    for phase in ("forward", "backward_recomputation"):
        selected = [row for row in rows if row["phase"] == phase]
        if {row["layer"] for row in selected} != expected_layers:
            raise ValueError("Missing State layer or phase coverage")
        for layer in expected_layers:
            if sum(row["calls"] for row in selected if row["layer"] == layer) != summary["microsteps"]:
                raise ValueError("Unexpected State invocation count")
    for row in rows:
        video, action, writer = row["video_shape"], row["action_source_shape"], row["writer_shape"]
        if action != writer or not row["context_shape"] or video[-1] != 3072 or writer[-1] != 1024:
            raise ValueError("Unexpected real-model State geometry")


def audit(runs):
    if set(runs) != set(GROUPS):
        raise ValueError("All four groups are required")
    records, paired, identities = {}, {}, {}
    for group in GROUPS:
        path = runs[group]
        state = json.loads((path / "state.json").read_text())
        if state["state"] != "completed" or state["exit_code"] != 0 or any(
            alive(row) for row in [state.get("supervisor"), state.get("child"), *state.get("processes", [])]
        ):
            raise ValueError(f"Workload has not completed and released processes: {group}")
        training = path / "training"
        identity = json.loads((training / "identity.json").read_text())
        if (identity["group"] != group or identity["world_size"] != 4 or identity["updates"] != 2
                or identity["micro_batch"] != 1 or identity["accumulation"] != 1 or not identity["workload_trace"]):
            raise ValueError("Require registered real-model two-update workload protocol")
        identities[group] = identity
        ranks, paired[group], hashes = [], [], {}
        for rank in range(4):
            trace_path, steps_path = training / f"workload_rank{rank}.json", training / f"steps_rank{rank}.jsonl"
            trace = json.loads(trace_path.read_text())
            steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
            completion = json.loads((training / f"summary_rank{rank}.json").read_text())
            if trace["identity"] != identity or completion["identity"] != identity or completion["stopped"]:
                raise ValueError("Trace/completion identity mismatch")
            if len(steps) != 2 or [row["position"] for row in steps] != [
                {"optimizer_update": 1, "next_occurrence": 4}, {"optimizer_update": 2, "next_occurrence": 8}
            ] or completion["position"] != steps[-1]["position"]:
                raise ValueError("Incomplete step evidence")
            summary = summarize_rank(trace, steps)
            validate_source_geometry(group, summary)
            ranks.append(summary)
            paired[group].append(steps)
            hashes[trace_path.name], hashes[steps_path.name] = sha256(trace_path), sha256(steps_path)
        if any(row["parameter_counts"] != ranks[0]["parameter_counts"] for row in ranks):
            raise ValueError("Rank effective parameter coverage differs")
        performance = json.loads((path / "performance_summary.json").read_text())
        records[group] = {"run": str(path), "identity": identity, "evidence_sha256": hashes, "ranks": ranks,
                          "wall_seconds": performance["wall_seconds"],
                          "allocated_gpu_hours": performance["allocated_gpu_hours"],
                          "save_seconds_by_rank": [json.loads((training / f"save_update2_rank{rank}.json").read_text())["seconds"] for rank in range(4)]}
    fields = ("protocol_id", "matrix_sha256", "seed", "episode_manifest_sha256", "statistics_sha256",
              "updates", "global_batch", "deterministic_algorithms")
    for group in GROUPS[1:]:
        if any(identities[group][key] != identities["E0"][key] for key in fields):
            raise ValueError("Unpaired workload protocol identities")
        for a, b in zip(paired["E0"], paired[group]):
            if any(x[key] != y[key] for x, y in zip(a, b) for key in ("samples", "noise_hashes")):
                raise ValueError("Workload samples/noise are not paired")
            if a[0]["metrics"] != b[0]["metrics"]:
                raise ValueError("Zero-gate baseline loss differs")
    counts = [records[group]["ranks"][0]["parameter_counts"] for group in GROUPS[1:]]
    if not all(row["state_trainable"] == row["state_backward_participating"] == counts[0]["state_trainable"] for row in counts):
        raise ValueError("State effective parameter budgets differ")
    return {"goal_id": GOAL_ID, "workload_audit_passed": True, "compute_matched": False,
            "state_effective_parameter_counts_equal": True, "groups": records,
            "limits": ["Two instrumented updates per group; not steady profiling or convergence",
                       "SDPA observations exclude other attention implementations and nonattention arithmetic",
                       "Backward phase counts checkpoint forward recomputation, not backward kernel FLOPs",
                       "Allowed mask pairs do not imply sparse execution; E1 reads less world context than E2/E3",
                       "Gradient coverage demonstrates backward participation, not equal information or capacity in a theoretical sense",
                       "Input-source isolation is tested separately; geometry alone cannot prove semantic isolation",
                       "Allocated GPU-hours are lease cost; NVML process attribution limitations remain"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="GROUP=run-directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing audit evidence")
    runs = dict(row.split("=", 1) for row in args.run)
    if len(runs) != len(args.run):
        raise ValueError("Duplicate group")
    atomic_json(args.output, audit({group: Path(path) for group, path in runs.items()}))


if __name__ == "__main__":
    main()
