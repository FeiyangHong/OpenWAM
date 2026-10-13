"""Audit finite formal training chains, retaining actual recomputation costs."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.checkpoints import validate_checkpoint
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256, stable_hash
from scripts.statewam.r2.cpu_job import alive
from scripts.statewam.r2.exposure_sampler import ExposureBatchSampler
from scripts.statewam.r2.formal_training import GROUPS, load_protocol


def validate_row(row, *, update, occurrences, global_batch):
    if row["position"] != {"optimizer_update": update, "next_occurrence": update * global_batch}:
        raise ValueError("Formal update/exposure cursor differs")
    if ([sample["occurrence"] for sample in row["samples"]] != occurrences
            or len(row["noise_hashes"]) != len(occurrences)):
        raise ValueError("Formal samples/noise have gaps")
    if (not row["gradients"] or "loss" not in row["metrics"]
            or any(not math.isfinite(value) for value in row["metrics"].values())):
        raise ValueError("Missing gradients or nonfinite formal loss")
    for name, (norm, finite) in row["gradients"].items():
        if finite != 1 or not math.isfinite(norm) or norm < 0 or name.startswith("video_backbone."):
            raise ValueError("Invalid formal gradients or frozen world was updated")
    phases = row["phase_timings"]
    if any(phases.get(name, {}).get("calls") != 1 for name in ("forward", "engine_backward", "engine_step")):
        raise ValueError("Formal optimizer update lacks timing evidence")


def pairing_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(bytes.fromhex(stable_hash({key: row[key] for key in ("samples", "noise_hashes")})))
    return digest.hexdigest()


def audit_group(paths, plan, protocol_hash, group):
    recipe = plan["training"]
    paths = [Path(path).resolve() for path in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Require a nonempty chain of distinct invocations")
    if recipe["accumulation"] != 1:
        raise ValueError("Formal audit requires the frozen accumulation-one recipe")
    logical = [dict() for _ in range(4)]
    identities, run_records, checkpoints = [], [], {}
    raw_exposures = 0
    for run_index, run in enumerate(paths):
        state = json.loads((run / "state.json").read_text())
        if (state["state"] not in ("completed", "stopped", "failed")
                or any(alive(row) for row in [state.get("supervisor"), state.get("child"), *state.get("processes", [])])):
            raise ValueError("Formal training chain contains a live run")
        if run_index == len(paths) - 1 and (state["state"], state["exit_code"]) != ("completed", 0):
            raise ValueError("Formal group lacks a successfully completed final invocation")
        training = run / "training"
        identity = json.loads((training / "identity.json").read_text())
        if (identity.get("purpose") != "training" or identity["group"] != group
                or identity["plan_identity"]["plan"] != plan
                or identity["plan_identity"]["protocol_sha256"] != protocol_hash
                or identity["protocol_id"] != plan["protocol_id"]
                or identity["matrix_sha256"] != plan["matrix"]["sha256"]
                or any(identity[key] != recipe[key] for key in
                       ("updates", "seed", "micro_batch", "accumulation", "global_batch", "world_size", "workers"))):
            raise ValueError("Formal training identity differs from frozen recipe")
        if identities and identity != identities[0]:
            raise ValueError("Resume changed training identity")
        identities.append(identity)
        sources = json.loads((training / "source_snapshot/manifest.json").read_text())
        expected = {name: digest for name, digest in plan["source_sha256"].items()
                    if name.startswith(("openwam/", "scripts/", "configs/"))}
        if sources != expected or any(sha256(training / "source_snapshot" / name) != digest for name, digest in sources.items()):
            raise ValueError("Formal training executable snapshot differs")
        per_rank = []
        common_invocation = None
        for rank in range(4):
            invocation = json.loads((training / f"invocation_rank{rank}.json").read_text())
            if common_invocation is not None and invocation != common_invocation:
                raise ValueError("Ranks resumed different invocation boundaries or checkpoints")
            common_invocation = invocation
            start = invocation["start_update"]
            if (invocation["purpose"] != "training" or invocation["end_update"] != recipe["updates"]
                    or not 0 <= start < recipe["updates"] or (run_index == 0 and start != 0)):
                raise ValueError("Formal invocation changed its total budget or start")
            if run_index and not invocation["resume"]:
                raise ValueError("Subsequent invocation did not resume a full checkpoint")
            if invocation["resume"]:
                source = Path(invocation["resume"]).resolve()
                if source not in checkpoints or checkpoints[source]["optimizer_update"] != start:
                    raise ValueError("Resume source is not a prior validated checkpoint in this chain")
            if set(logical[rank]).intersection(range(1, start + 1)) != set(range(1, start + 1)):
                raise ValueError("Resume checkpoint lacks a complete recorded prefix")
            # Drop abandoned suffixes, while accounting their physical work separately.
            logical[rank] = {update: row for update, row in logical[rank].items() if update <= start}
            sampler = iter(ExposureBatchSampler(updates=recipe["updates"], micro_batch=recipe["micro_batch"],
                accumulation=1, world_size=4, rank=rank, start_update=start))
            count = 0
            with (training / f"steps_rank{rank}.jsonl").open() as stream:
                for count, line in enumerate(stream, 1):
                    row = json.loads(line)
                    update = start + count
                    expected_occurrences = next(sampler, None)
                    if expected_occurrences is None:
                        raise ValueError("Formal run exceeded its finite update budget")
                    validate_row(row, update=update, occurrences=expected_occurrences, global_batch=recipe["global_batch"])
                    raw_exposures += len(row["samples"])
                    logical[rank][update] = {key: row[key] for key in ("samples", "noise_hashes", "metrics")}
            per_rank.append({"start_update": start, "updates_recorded": count,
                             "steps_sha256": sha256(training / f"steps_rank{rank}.jsonl")})
            if run_index == len(paths) - 1:
                summary = json.loads((training / f"summary_rank{rank}.json").read_text())
                final = {"optimizer_update": recipe["updates"], "next_occurrence": recipe["exposures"]}
                if (summary["identity"] != identity or summary["position"] != final or summary["stopped"]
                        or start + count != recipe["updates"]):
                    raise ValueError("Formal final rank did not finish the budget")
        for checkpoint in sorted((training / "checkpoints").glob("update_*")):
            if checkpoint.name.endswith(".partial"):
                continue
            position = validate_checkpoint(checkpoint, identity=identity)
            if (checkpoint.name != f"update_{position['optimizer_update']:08d}"
                    or any(not row["start_update"] < position["optimizer_update"]
                           <= row["start_update"] + row["updates_recorded"] for row in per_rank)):
                raise ValueError("Full checkpoint is outside the recorded invocation progress")
            checkpoints[checkpoint.resolve()] = position
        if run_index == len(paths) - 1:
            final_checkpoint = training / "checkpoints" / f"update_{recipe['updates']:08d}"
            if final_checkpoint not in checkpoints:
                raise ValueError("Formal final full checkpoint is missing")
            latest = json.loads((training / "checkpoints/latest-complete.json").read_text())
            if latest != {"checkpoint": final_checkpoint.name, "complete_sha256": sha256(final_checkpoint / "COMPLETE.json")}:
                raise ValueError("Formal latest-complete pointer differs")
        performance = json.loads((run / "performance_summary.json").read_text())
        run_records.append({"run": str(run), "rank_invocations": per_rank,
                            "performance_sha256": sha256(run / "performance_summary.json"),
                            "wall_seconds": performance["wall_seconds"],
                            "allocated_gpu_hours": performance["allocated_gpu_hours"]})
    required = set(range(1, recipe["updates"] + 1))
    if not identities or any(set(rank) != required for rank in logical):
        raise ValueError("Formal logical update chain has gaps")
    rows = [[rank[update] for update in sorted(rank)] for rank in logical]
    counts = Counter(str(sample["task_index"]) for rank in rows for row in rank for sample in row["samples"])
    if sum(counts.values()) != recipe["exposures"]:
        raise ValueError("Formal logical exposures differ from frozen budget")
    saved_updates = {position["optimizer_update"] for position in checkpoints.values()}
    required_saves = set(range(recipe["save_every"], recipe["updates"] + 1, recipe["save_every"])) | {recipe["updates"]}
    if not required_saves.issubset(saved_updates):
        raise ValueError("Registered full checkpoint save nodes are missing")
    losses = []
    for start in range(0, recipe["updates"], 500):
        end = min(start + 500, recipe["updates"])
        keys = set(rows[0][0]["metrics"])
        losses.append({"start_update": start + 1, "end_update": end,
            "mean_training_losses": {key: sum(row["metrics"][key] for rank in rows for row in rank[start:end]) / (4 * (end - start))
                                     for key in keys}})
    return {"identity": identities[0], "runs": run_records, "logical_exposures": sum(counts.values()),
            "actual_recorded_exposures": raw_exposures, "recorded_recomputed_exposures": raw_exposures - sum(counts.values()),
            "task_exposures": dict(counts), "paired_trace_sha256_by_rank": [pairing_digest(rank) for rank in rows],
            "first_loss_by_rank": [rank[0]["metrics"] for rank in rows], "loss_windows": losses,
            "checkpoint_count": len(checkpoints), "saved_updates": sorted(saved_updates),
            "final_checkpoint": str(final_checkpoint),
            "limits": "Actual recorded exposures exclude work lost before log persistence; loss is training-only, not convergence or policy efficacy"}


def audit(protocol, chains):
    plan = load_protocol(protocol)
    if set(chains) != set(GROUPS) or any(not chains[group] for group in GROUPS):
        raise ValueError("Require ordered invocation chains for all four groups")
    reports = {group: audit_group(chains[group], plan, sha256(protocol), group) for group in GROUPS}
    baseline = reports["E0"]
    for group in GROUPS[1:]:
        if reports[group]["paired_trace_sha256_by_rank"] != baseline["paired_trace_sha256_by_rank"]:
            raise ValueError("Formal group sample/augmentation/noise sequences are not paired")
        if reports[group]["first_loss_by_rank"] != baseline["first_loss_by_rank"]:
            raise ValueError("Formal zero-gate first losses differ")
    return {"goal_id": GOAL_ID, "passed": True, "protocol_sha256": sha256(protocol), "groups": reports,
            "all_samples_noise_paired": True, "first_loss_exact": True,
            "limits": "One training seed, fixed step/sample budget; not compute matching or a convergence claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--chains", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior formal training audits")
    atomic_json(args.output, audit(args.protocol, json.loads(args.chains.read_text())))


if __name__ == "__main__":
    main()
