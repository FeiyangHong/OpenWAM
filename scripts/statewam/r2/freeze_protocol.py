"""Publish the finite R2 protocol only after verifying every required gate."""

import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.common import ARTIFACTS, GOAL_ID, ROOT, TASKS, atomic_json, sha256
from scripts.statewam.r2.evaluation_plan import EXECUTION
from scripts.statewam.r2.cpu_job import alive
from scripts.statewam.r2.evaluation_scenes import validate_rows
from scripts.statewam.r2.formal_training import (
    GROUPS, PROTOCOL, TRAINING, INFERENCE, EVALUATION, executable_paths, load_protocol,
)

GATES = {
    "p0": "p0_acceptance.json", "capacity": "capacity_report.json",
    "profiling": "profiling_paired_001.json", "resume": "batch16_resume_audit_001.json",
    "training_faults": "model_fault_audit_001.json", "physical_recovery": "physical_recovery_001.json",
    "scenes": "evaluation_scenes_001.json", "cross_process_reset": "cross_process_reset_001.json",
    "pilot_lifecycle": "pilot_lifecycle_001.json",
}


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def timing_proxies(artifacts):
    proxies = []
    for filename in ("physical_episode_002.json", "physical_stop_pair_001.json", "physical_kill_pair_001.json"):
        report_path = artifacts / "reports" / filename
        report = json.loads(report_path.read_text())
        episode = report.get("resumed_episode_audit", report)
        run = Path(episode["run"])
        service_path = run / "physical/service/state.json"
        state_path = run / "state.json"
        state = json.loads(state_path.read_text())
        service = json.loads(service_path.read_text())
        coordinator_path = run / "physical/state.json"
        coordinator = json.loads(coordinator_path.read_text())
        performance_path = run / "performance_summary.json"
        performance = json.loads(performance_path.read_text())
        if (report["passed"] is not True or episode["passed"] is not True
                or (state["state"], state["exit_code"]) != ("completed", 0)
                or any(alive(row) for row in [state.get("supervisor"), state.get("child"), *state.get("processes", [])])
                or service["state"] != "stopped" or episode["stats"]["policy_actions"] != 400
                or episode["stats"]["step_limit"] != 400 or episode["identity"]["group"] != "E0"
                or episode["identity"]["task"] != "adjust_bottle"):
            raise ValueError("Timing proxy lacks complete registered engineering evidence")
        startup = service["started_at"] - coordinator["children"]["service"]["owner"]["created_at"]
        residual = performance["wall_seconds"] - startup - service["wall_seconds"]
        proxies.append({"identity": episode["identity"], "policy_actions": 400,
                        "service_seconds": service["wall_seconds"], "startup_seconds": startup,
                        "preflight_and_teardown_residual_seconds": residual,
                        "evidence": [reference(path) for path in (report_path, service_path, state_path,
                                                                   coordinator_path, performance_path)]})
    return proxies


def estimate_evaluation(scenes, proxies):
    """Planning scenarios from completed non-final episodes, not a measured pilot ETA."""
    validate_rows(scenes["rows"])
    if len(proxies) != 3:
        raise ValueError("Require the three registered complete timing proxies")
    fields = ("policy_actions", "service_seconds", "startup_seconds", "preflight_and_teardown_residual_seconds")
    if any(not math.isfinite(row[key]) or row[key] <= 0 for row in proxies for key in fields):
        raise ValueError("Invalid measured timing proxy")
    limits = {}
    for split, rows in scenes["rows"].items():
        for row in rows:
            path = Path(row["probe"]) / "reset_0/initial.json"
            limit = json.loads(path.read_text())["metadata"]["step_lim"]
            if type(limit) is not int or limit <= 0:
                raise ValueError("Missing original scene action limit for timing estimate")
            limits.setdefault((split, row["task"], row["condition"]), []).append((row["seed"], limit))
    pairs = [(task, condition) for task in TASKS for condition in ("demo_clean", "demo_randomized")]
    diagnostics = [2 * min(limits[("development", *pair)])[1] for pair in pairs]
    wave_limits = [max(diagnostics[index:index + 2]) for index in range(0, len(diagnostics), 2)]
    for split in ("development", "final"):
        for pair in pairs:
            total = sum(limit for _, limit in limits[(split, *pair)])
            wave_limits.extend([total, total])
    if len(wave_limits) != 90:
        raise ValueError("Timing estimate differs from the frozen 90-wave schedule")
    seconds_per_action = statistics.median(row["service_seconds"] / row["policy_actions"] for row in proxies)
    startup = statistics.median(row["startup_seconds"] for row in proxies)
    residual = statistics.median(row["preflight_and_teardown_residual_seconds"] for row in proxies)
    scenarios = []
    for factor in (1.0, 1.5, 2.0):
        seconds = residual + len(wave_limits) * startup + sum(wave_limits) * seconds_per_action * factor
        scenarios.append({"assumed_rollout_slowdown_factor": factor, "wall_hours": seconds / 3600,
                          "allocated_gpu_hours": EXECUTION["physical_device_count"] * seconds / 3600})
    return {"method": "original-action-limit-scaled engineering service wall time, 90 two-lane barrier waves",
            "planning_wall_hours": scenarios[1]["wall_hours"],
            "planning_allocated_gpu_hours": scenarios[1]["allocated_gpu_hours"],
            "sensitivity_scenarios_not_confidence_intervals": scenarios, "proxies": proxies,
            "median_service_seconds_per_action": seconds_per_action,
            "median_process_startup_seconds": startup, "preflight_and_teardown_residual_seconds": residual,
            "assumptions": ["Every episode reaches its original limit; no early-success speedup assumed",
                "E0 step-zero adjust_bottle clean service timing is a proxy for all groups/tasks/conditions",
                "Startup/setup inside service time scales with action limit; per-block model launch charged separately",
                "Two lanes share physical simulator GPU 0; 1.5x rollout slowdown is an assumption, not measured",
                "One uninterrupted evaluation invocation; no faults, queue waits, export or audits included"],
            "limits": "Low-confidence planning proxy only; neither ten-task throughput nor shared-GPU concurrency is measured here. Re-estimate from fixed completed blocks without changing the experiment budget"}


def make_plan(artifacts=ARTIFACTS):
    artifacts = Path(artifacts).resolve()
    candidate_path = artifacts / "manifests/training_budget_candidate_001.json"
    candidate = json.loads(candidate_path.read_text())
    if (candidate["updates_per_group"] != TRAINING["updates"] or candidate["train_seed"] != TRAINING["seed"]
            or candidate["logical_exposures_per_group"] != TRAINING["exposures"]
            or candidate["groups"] != GROUPS):
        raise ValueError("Budget candidate no longer matches the common finite recipe")
    for name, digest in candidate["sources"].items():
        path = artifacts / name if name.startswith("reports/") else ROOT / name
        if sha256(path) != digest:
            raise ValueError("Training budget reference changed")
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in executable_paths()}
    estimate = copy.deepcopy(candidate["training_estimate"])
    scenes = json.loads((artifacts / "reports/evaluation_scenes_001.json").read_text())
    estimate["evaluation_estimate"] = estimate_evaluation(scenes, timing_proxies(artifacts))
    estimate["evaluation_estimate_reason"] = "Evidence-backed engineering proxy with explicit unmeasured task/concurrency assumptions"
    estimate["training_plus_evaluation_planning_wall_hours"] = (
        estimate["four_group_serial_wall_hours"] + estimate["evaluation_estimate"]["planning_wall_hours"])
    estimate["training_plus_evaluation_planning_allocated_gpu_hours"] = (
        estimate["four_group_allocated_gpu_hours"] + estimate["evaluation_estimate"]["planning_allocated_gpu_hours"])
    # Until multi-task policy throughput exists, report a deadline envelope,
    # not an extrapolated throughput claim from one engineering task.
    wave_sizes = [2] * 10 + [5] * 40 + [20] * 40
    evaluation_envelope = sum(EXECUTION["service_startup_seconds"] + size * EXECUTION["episode_seconds"]
                              + EXECUTION["shutdown_seconds"] for size in wave_sizes)
    evaluation_envelope += 2 * EXECUTION["preflight_seconds"]
    estimate.update(evaluation_deadline_envelope_wall_hours=evaluation_envelope / 3600,
                    evaluation_deadline_envelope_allocated_gpu_hours=3 * evaluation_envelope / 3600,
                    evaluation_timing_note="Conservative timeout envelope only, not an expected duration; update observed throughput after each fixed block without expanding episode counts or deadlines",
                    evaluation_schedule_waves=90, evaluation_simulator_lanes=2,
                    evaluation_physical_devices=3,
                    free_storage_bytes_at_freeze=shutil.disk_usage(artifacts).free)
    if estimate["free_storage_bytes_at_freeze"] < candidate["training_estimate"]["full_checkpoint_logical_bytes_estimate"] * 1.2:
        raise ValueError("Insufficient free storage for registered checkpoints and working reserve")
    return {"goal_id": GOAL_ID, "protocol_id": PROTOCOL, "status": "frozen", "groups": list(GROUPS),
        "tasks": list(TASKS), "training": copy.deepcopy(TRAINING), "inference": copy.deepcopy(INFERENCE),
        "evaluation": copy.deepcopy(EVALUATION), "evaluation_execution": copy.deepcopy(EXECUTION),
        "matrix": reference(artifacts / "checkpoints/foundation-matrix-v3/initialization.json"),
        "environment": reference(artifacts / "reports/simulator_environment_001.json"),
        "assets": reference(artifacts / "reports/scene_assets_002.json"),
        "evidence": {key: reference(artifacts / "reports" / name) for key, name in GATES.items()},
        "source_sha256": sources, "budget_candidate": reference(candidate_path), "estimates": estimate,
        "limits": "From-Foundation clean-only, step/sample matched, single training seed, non-compute-matched pilot; no success guarantee or scope expansion"}


def freeze(output, artifacts=ARTIFACTS):
    output = Path(output).resolve()
    if output.name != "protocol.yaml":
        raise ValueError("Use the goal-local protocol.yaml filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    marker = output.with_name("protocol.COMPLETE.json")
    with output.with_name("protocol.freeze.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists() or marker.exists():
            raise FileExistsError("Preserve existing protocol, including an incomplete publication")
        plan = make_plan(artifacts)
        with tempfile.TemporaryDirectory(prefix=".r2-freeze-", dir=output.parent) as temporary:
            staged = Path(temporary) / "protocol.yaml"
            with staged.open("x") as stream:
                yaml.safe_dump(plan, stream, sort_keys=False, allow_unicode=False)
                stream.flush()
                os.fsync(stream.fileno())
            completion = {"goal_id": GOAL_ID, "protocol_sha256": sha256(staged)}
            atomic_json(staged.with_name("protocol.COMPLETE.json"), completion)
            load_protocol(staged)
            os.link(staged, output)
            atomic_json(marker, completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "manifests/protocol.yaml")
    args = parser.parse_args()
    print(json.dumps(freeze(args.output)))


if __name__ == "__main__":
    main()
