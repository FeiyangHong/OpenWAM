"""Bounded E0/E3 profiling from a recorded plan, separate from engineering/formal training."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.common import GOAL_ID, sha256


def validate_plan(plan):
    if plan["goal_id"] != GOAL_ID or plan["purpose"] != "profiling" or plan["groups"] != ["E0", "E3"]:
        raise ValueError("Not an E0/E3 R2 profiling plan")
    if not 200 <= plan["updates"] <= 500 or plan["world_size"] != 4:
        raise ValueError("Profiling requires 200..500 updates on four ranks")
    if min(plan["micro_batch"], plan["accumulation"], plan["save_every"]) < 1 or plan["workers"] < 0:
        raise ValueError("Invalid batch/worker/save plan")
    if plan["global_batch"] != 4 * plan["micro_batch"] * plan["accumulation"]:
        raise ValueError("Global batch is inconsistent")
    if plan["seed"] == 42 or plan["warmup_updates"] != 10:
        raise ValueError("Use a separate diagnostic seed and ten warmup updates")
    if plan["save_every"] > plan["updates"] or not plan["protocol_id"].startswith("r2-profile-"):
        raise ValueError("Missing profiling checkpoint/protocol budget")
    return plan


def load_plan(path):
    plan = validate_plan(json.loads(path.read_text()))
    capacity_path = Path(plan["capacity_report"])
    if sha256(capacity_path) != plan["capacity_report_sha256"]:
        raise ValueError("Capacity report changed")
    capacity = json.loads(capacity_path.read_text())
    if capacity["goal_id"] != GOAL_ID or not capacity["workload_audit_passed"] or not capacity["state_effective_parameter_counts_equal"]:
        raise ValueError("Workload/capacity gate has not passed")
    if set(capacity["groups"]) != {"E0", "E1", "E2", "E3"}:
        raise ValueError("Capacity evidence must cover all four groups")
    matrix = Path(plan["matrix"])
    if sha256(matrix / "initialization.json") != plan["matrix_sha256"]:
        raise ValueError("Initialization identity changed")
    for group, record in capacity["groups"].items():
        if record["identity"]["matrix_sha256"] != plan["matrix_sha256"]:
            raise ValueError(f"Capacity evidence has another initialization: {group}")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--group", choices=("E0", "E3"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--end-update", type=int)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    from scripts.statewam.r2.train import run_experiment
    settings = SimpleNamespace(matrix=Path(plan["matrix"]), group=args.group, output=args.output,
                               updates=plan["updates"], micro_batch=plan["micro_batch"],
                               accumulation=plan["accumulation"], workers=plan["workers"],
                               save_every=plan["save_every"], end_update=args.end_update, resume=args.resume,
                               audit_workload=False)
    run_experiment(settings, protocol_id=plan["protocol_id"], seed=plan["seed"], purpose="profiling",
                   plan_identity={"plan_sha256": sha256(args.plan), "plan": plan})


if __name__ == "__main__":
    main()
