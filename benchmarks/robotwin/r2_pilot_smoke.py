"""Eight preselected development episodes exercise the real pilot lifecycle."""

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.r2_audit_pilot import audit_result, request_inventory
from benchmarks.robotwin.r2_audit_episode import require_conversion_environment
from benchmarks.robotwin.r2_pilot_evaluation import execute_schedule
from benchmarks.robotwin.r2_scene_manifest import inspect_probe
from benchmarks.robotwin.r2_scene_queue import verify_shard
from scripts.statewam.r2.common import ARTIFACTS, GOAL_ID, atomic_json, sha256, stable_hash
from scripts.statewam.r2.evaluation_plan import EXECUTION, episode_identity, scheduled_episodes
from scripts.statewam.r2.formal_training import executable_paths

PROTOCOL = "r2-pilot-lifecycle-engineering-v1"


def make_plan(pool, matrix, environment, assets):
    pool, matrix, environment, assets = [Path(path).resolve() for path in (pool, matrix, environment, assets)]
    scenes = []
    for lane, condition in ((0, "demo_clean"), (1, "demo_randomized")):
        directory = pool / f"lane_{lane}/shards/development/adjust_bottle" / condition
        verify_shard(directory, {"split": "development", "task": "adjust_bottle", "condition": condition})
        accepted = json.loads((directory / "manifest.json").read_text())["accepted"]
        for row in sorted(accepted, key=lambda row: row["seed"])[:2]:
            scenes.append({"task": "adjust_bottle", "condition": condition, **row})
    files = [environment, assets, matrix / "initialization.json", matrix / "COMPLETE.json"]
    for group in ("E0", "E3"):
        files.extend(matrix / group / name for name in
                     ("checkpoint_step_0.safetensors", "config.yaml", "normalization_stats.npy"))
    return {"goal_id": GOAL_ID, "protocol_id": PROTOCOL, "purpose": "engineering", "max_episodes": 8,
            "automatic_retries": 0, "groups": ["E0", "E3"], "execution": copy.deepcopy(EXECUTION),
            "matrix": str(matrix), "scenes": scenes,
            "environment": {"path": str(environment), "sha256": sha256(environment)},
            "assets": {"path": str(assets), "sha256": sha256(assets)},
            "files": {str(path): sha256(path) for path in files},
            "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in executable_paths()},
            "limits": "Two waves, E0/E3 step-zero, first two accepted adjust_bottle development scenes per condition; original limits, no final scenes, no efficacy or recovery claims"}


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if (plan["goal_id"] != GOAL_ID or plan["protocol_id"] != PROTOCOL or plan["purpose"] != "engineering"
            or plan["max_episodes"] != 8 or plan["automatic_retries"] != 0
            or plan["groups"] != ["E0", "E3"] or plan["execution"] != EXECUTION):
        raise ValueError("Require the finite eight-episode lifecycle engineering plan")
    matrix = Path(plan["matrix"])
    required = {plan["environment"]["path"], plan["assets"]["path"], str(matrix / "initialization.json"), str(matrix / "COMPLETE.json")}
    required.update(str(matrix / group / name) for group in plan["groups"] for name in
                    ("checkpoint_step_0.safetensors", "config.yaml", "normalization_stats.npy"))
    if set(plan["files"]) != required or any(sha256(name) != digest for name, digest in plan["files"].items()):
        raise ValueError("Engineering input file identity changed")
    for name in ("environment", "assets"):
        if plan[name]["sha256"] != plan["files"][plan[name]["path"]]:
            raise ValueError("Engineering provenance reference differs from inventory")
    expected_sources = {str(path.relative_to(ROOT)): sha256(path) for path in executable_paths()}
    if plan["source_sha256"] != expected_sources:
        raise ValueError("Engineering executable sources changed after registration")
    if len(plan["scenes"]) != 4:
        raise ValueError("Engineering must retain four preselected scenes")
    for row in plan["scenes"]:
        if (row["task"] != "adjust_bottle" or row["condition"] not in ("demo_clean", "demo_randomized")
                or not 200000 <= row["seed"] < 201000
                or row["scene_id"] != f"development/adjust_bottle/{row['condition']}/{row['seed']}"):
            raise ValueError("Engineering cannot use final or substitute scenes")
        inspected = inspect_probe(Path(row["probe"]), task=row["task"], condition=row["condition"], seed=row["seed"])
        if {**inspected, "scene_id": row["scene_id"], "task": row["task"], "condition": row["condition"]} != row:
            raise ValueError("Engineering prepared scene changed")
    return plan


def schedule_plan(plan, protocol_hash):
    bindings = {group: {"directory": str(Path(plan["matrix"]) / group), "weight_name": "checkpoint_step_0.safetensors",
                        "weight_sha256": plan["files"][str(Path(plan["matrix"]) / group / "checkpoint_step_0.safetensors")]}
                for group in plan["groups"]}
    waves = []
    for index, condition in enumerate(("demo_clean", "demo_randomized")):
        rows = sorted((row for row in plan["scenes"] if row["condition"] == condition), key=lambda row: row["seed"])
        if len(rows) != 2 or len({row["scene_id"] for row in rows}) != 2:
            raise ValueError("Require two distinct fixed development scenes per condition")
        blocks = []
        for lane, group in enumerate(plan["groups"][::1 if index == 0 else -1]):
            episodes = []
            for row in rows:
                identity = episode_identity(row, group=group, checkpoint_sha256=bindings[group]["weight_sha256"],
                    protocol_sha256=protocol_hash, split="development")
                episodes.append({"identity": identity, "episode_key": stable_hash(identity), "scene": row})
            blocks.append({"block_id": f"wave_{index:03d}_lane_{lane}", "lane": lane, "split": "development",
                           "task": "adjust_bottle", "condition": condition, "group": group, "episodes": episodes})
        waves.append({"wave": index, "blocks": blocks})
    return {"goal_id": GOAL_ID, "protocol_sha256": protocol_hash, "execution": copy.deepcopy(EXECUTION),
            "checkpoint_bindings": bindings, "counts": {"engineering": 8}, "waves": waves}


def audit(plan_path, run, output):
    plan = load_plan(plan_path)
    require_conversion_environment(json.loads(Path(plan["environment"]["path"]).read_text()))
    schedule = schedule_plan(plan, sha256(plan_path))
    execution = json.loads((run / "evaluation/execution.json").read_text())
    ledger = Path(execution["ledger_root"])
    if json.loads((ledger / "schedule.json").read_text()) != schedule:
        raise ValueError("Engineering execution changed the registered schedule")
    requests, provenance = request_inventory([run], schedule, plan)
    records = [audit_result(ledger, row, requests) for row in scheduled_episodes(schedule)]
    if len(records) != 8 or any(row["attempt"] != 1 for row in records):
        raise ValueError("Engineering count or attempt budget changed")
    for wave in schedule["waves"]:
        for block in wave["blocks"]:
            state = json.loads((run / "evaluation" / block["block_id"] / "state.json").read_text())
            if (state["state"] != "completed" or state["skipped"]
                    or state["completed"] != [row["episode_key"] for row in block["episodes"]]
                    or set(state["children"]) != {"service", "episode_000", "episode_001"}):
                raise ValueError("Engineering lacks service reuse and fresh simulator evidence")
    atomic_json(output, {"goal_id": GOAL_ID, "passed": True, "protocol_id": PROTOCOL,
                        "plan_sha256": sha256(plan_path), "run_evidence": provenance, "records": records,
                        "matrix_sha256": plan["files"][str(Path(plan["matrix"]) / "initialization.json")],
                        "environment_sha256": plan["environment"]["sha256"],
                        "assets_sha256": plan["assets"]["sha256"], "source_sha256": plan["source_sha256"],
                        "service_reuse_and_fresh_simulators": True, "completed_waves": 2,
                        "limits": plan["limits"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("prepare", "run", "audit"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--ledger-root", type=Path)
    args = parser.parse_args()
    if args.role == "prepare":
        if args.plan.exists():
            raise FileExistsError("Preserve registered engineering plan")
        plan = make_plan(ARTIFACTS / "runs/p1-scenes-pool-003/scenes", ARTIFACTS / "checkpoints/foundation-matrix-v3",
                         ARTIFACTS / "reports/simulator_environment_001.json", ARTIFACTS / "reports/scene_assets_002.json")
        atomic_json(args.plan, plan)
    elif args.role == "run":
        plan = load_plan(args.plan)
        execute_schedule(plan, schedule_plan(plan, sha256(args.plan)), args.plan, args.ledger_root, args.output)
    else:
        if args.output.exists():
            raise FileExistsError("Preserve engineering audit")
        audit(args.plan, args.run.resolve(), args.output)


if __name__ == "__main__":
    main()
