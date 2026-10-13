"""Deterministic, bounded scheduling for the final-checkpoint R2 matrix."""

import json
from pathlib import Path

from benchmarks.robotwin.r2_episode_ledger import validate_identity
from scripts.statewam.r2.common import GOAL_ID, TASKS, sha256, stable_hash
from scripts.statewam.r2.evaluation_scenes import validate_rows

GROUPS = ("E0", "E1", "E2", "E3")
CONDITIONS = ("demo_clean", "demo_randomized")
EXECUTION = {
    "phase_order": ["repeatability", "development", "final"],
    "simulator_lanes": 2, "shared_simulator_physical_gpu": 0, "policy_devices": 2,
    "physical_device_count": 3, "group_order": "rotating_pairs_by_task_condition_v1",
    "fresh_simulator_per_episode": True, "automatic_retries": 0,
    "preflight_seconds": 1200, "provenance_seconds": 900,
    "service_startup_seconds": 900, "episode_seconds": 3600,
    "service_idle_seconds": 600, "shutdown_seconds": 60,
    "statistics": "paired_percentile_bootstrap_task_scene_v1",
    "significance_tests": False,
}


def verify_export(directory, plan, group):
    directory = Path(directory).resolve()
    report = json.loads((directory / "export.json").read_text())
    if json.loads((directory / "COMPLETE.json").read_text()) != {
            "goal_id": GOAL_ID, "export_sha256": sha256(directory / "export.json")}:
        raise ValueError("Deployment export completion differs")
    identity = report["identity"]
    recipe = plan["training"]
    if (report["goal_id"] != GOAL_ID or report["passed"] is not True
            or report["all_tensor_values_exact"] is not True
            or identity["group"] != group or identity.get("purpose") != "training"
            or identity["protocol_id"] != plan["protocol_id"]
            or identity["plan_identity"]["plan"] != plan
            or identity["matrix_sha256"] != plan["matrix"]["sha256"]
            or any(identity[key] != recipe[key] for key in
                   ("updates", "seed", "micro_batch", "accumulation", "world_size", "global_batch", "workers", "zero_stage"))
            or report["position"] != {"optimizer_update": recipe["updates"], "next_occurrence": recipe["exposures"]}
            or report["weight_name"] != f"checkpoint_step_{recipe['updates']}.safetensors"):
        raise ValueError("Require this protocol's formal final checkpoint; engineering exports are excluded")
    protocol_path = Path(identity["plan_identity"]["protocol_path"])
    if not protocol_path.is_absolute() or sha256(protocol_path) != identity["plan_identity"]["protocol_sha256"]:
        raise ValueError("Exported training protocol file identity changed")
    required = {report["weight_name"], "config.yaml", "normalization_stats.npy"}
    if not required.issubset(report["files"]):
        raise ValueError("Incomplete deployment file inventory")
    for name, digest in report["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha256(path) != digest:
            raise ValueError("Exported deployment files changed")
    if report["files"]["normalization_stats.npy"] != identity["statistics_sha256"]:
        raise ValueError("Exported statistics differ from training")
    return {"directory": str(directory), "export_sha256": sha256(directory / "export.json"),
            "weight_name": report["weight_name"], "weight_sha256": report["files"][report["weight_name"]]}


def episode_identity(row, *, group, checkpoint_sha256, protocol_sha256, split, replicate=0):
    return validate_identity({"goal_id": GOAL_ID, "group": group, "checkpoint_sha256": checkpoint_sha256,
        "protocol_sha256": protocol_sha256, "scene_manifest_sha256": stable_hash(row),
        "instruction_sha256": row["instruction"]["text_sha256"], "task": row["task"],
        "condition": row["condition"], "scene_id": row["scene_id"], "policy_seed": row["policy_seed"],
        "split": split, "replicate": replicate})


def build_schedule(scenes, checkpoints, protocol_sha256):
    """Two equally provisioned lanes; rotate both group order and device slot.

    Checkpoints are verified by the caller before this pure scheduling step.
    A wave is a barrier: neither lane starts the next wave early.
    """
    validate_rows(scenes["rows"])
    if set(checkpoints) != set(GROUPS):
        raise ValueError("Require all four final checkpoint identities")
    waves = []

    def block(split, task, condition, group, rows, replicates=(0,)):
        episodes = []
        for row in rows:
            for replicate in replicates:
                identity = episode_identity(row, group=group, checkpoint_sha256=checkpoints[group]["weight_sha256"],
                                            protocol_sha256=protocol_sha256, split=split, replicate=replicate)
                episodes.append({"identity": identity, "episode_key": stable_hash(identity), "scene": row})
        return {"split": split, "task": task, "condition": condition, "group": group, "episodes": episodes}

    pairs = [(task, condition) for task in TASKS for condition in CONDITIONS]
    diagnostics = []
    for task, condition in pairs:
        first = min((row for row in scenes["rows"]["development"]
                     if (row["task"], row["condition"]) == (task, condition)), key=lambda row: row["seed"])
        diagnostics.append(block("repeatability", task, condition, "E0", [first], (0, 1)))
    for index in range(0, len(diagnostics), 2):
        waves.append(diagnostics[index:index + 2])
    for split in ("development", "final"):
        for pair_index, (task, condition) in enumerate(pairs):
            rows = sorted((row for row in scenes["rows"][split]
                           if (row["task"], row["condition"]) == (task, condition)), key=lambda row: row["seed"])
            offset = pair_index % len(GROUPS)
            order = GROUPS[offset:] + GROUPS[:offset]
            for index in (0, 2):
                waves.append([block(split, task, condition, group, rows) for group in order[index:index + 2]])
    result = []
    for index, wave in enumerate(waves):
        result.append({"wave": index, "blocks": [{**value, "lane": lane, "block_id": f"wave_{index:03d}_lane_{lane}"}
                                                  for lane, value in enumerate(wave)]})
    episodes = [episode for wave in result for value in wave["blocks"] for episode in value["episodes"]]
    counts = {split: sum(row["identity"]["split"] == split for row in episodes) for split in EXECUTION["phase_order"]}
    if counts != {"repeatability": 40, "development": 400, "final": 1600} or len({r["episode_key"] for r in episodes}) != 2040:
        raise ValueError("Schedule changed the finite episode matrix")
    return {"goal_id": GOAL_ID, "protocol_sha256": protocol_sha256, "execution": EXECUTION,
            "checkpoint_bindings": checkpoints, "scene_rows_sha256": scenes["rows_sha256"],
            "counts": counts, "waves": result}


def scheduled_episodes(schedule):
    return [episode for wave in schedule["waves"] for block in wave["blocks"] for episode in block["episodes"]]
