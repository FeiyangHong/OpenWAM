"""Replay one preselected development scene per task/condition in fresh processes."""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.r2_episode_engineering import snapshot_sources, verify_simulator_provenance
from benchmarks.robotwin.r2_instructions import replay_instruction
from benchmarks.robotwin.r2_rollout import verify_initial
from benchmarks.robotwin.r2_scene_manifest import inspect_probe
from scripts.statewam.r2.common import GOAL_ID, TASKS, atomic_json, sha256, stable_hash
from scripts.statewam.r2.cpu_job import alive, identity as process_identity
from scripts.statewam.r2.cuda_identity import verify_cuda_devices
from scripts.statewam.r2.evaluation_scenes import validate_rows, verify
from scripts.statewam.r2.source_snapshot import assert_sources_unchanged

PROTOCOL = "r2-cross-process-reset-v1"


def selected_rows(scenes):
    validate_rows(scenes["rows"])
    return [min((row for row in scenes["rows"]["development"]
                 if row["task"] == task and row["condition"] == condition), key=lambda row: row["seed"])
            for task in TASKS for condition in ("demo_clean", "demo_randomized")]


def probe(row_path, output):
    from benchmarks.robotwin.r2_scene_probe import bind_renderer, configuration, initial_observation
    from benchmarks.robotwin.eval_policy_wrapper import bootstrap_robotwin_module, _robotwin_git, VERIFIED_ROBOTWIN_COMMIT

    row = json.loads(Path(row_path).read_text())
    original = inspect_probe(Path(row["probe"]), task=row["task"], condition=row["condition"], seed=row["seed"])
    if {**original, "scene_id": row["scene_id"], "task": row["task"], "condition": row["condition"]} != row:
        raise ValueError("Cross-process reset scene differs from prepared evidence")
    if not row["scene_id"].startswith("development/"):
        raise ValueError("Independent reset gate uses preselected development scenes only")
    uuids = verify_cuda_devices()
    if len(uuids) != 1:
        raise ValueError("Cross-process reset requires a single device")
    root = Path(os.environ["ROBOTWIN_PATH"]).resolve()
    if _robotwin_git(str(root), "rev-parse", "HEAD") != VERIFIED_ROBOTWIN_COMMIT:
        raise ValueError("Unverified simulator source")
    if os.environ.get("ROBOTWIN_ENABLE_PLANNER_FALLBACK") == "1":
        raise ValueError("Planner fallback forbidden")
    output.mkdir(parents=True, exist_ok=False)
    os.environ["ROBOTWIN_RUNTIME_ROOT"] = str(output / "runtime")
    module = bootstrap_robotwin_module()
    renderer = bind_renderer()
    cfg, limit = configuration(module, root, row["task"], row["condition"])
    if stable_hash(cfg) != row["config_sha256"]:
        raise ValueError("Reset environment configuration changed")
    instruction = replay_instruction(row["instruction"], robotwin_root=root, task=row["task"],
                                     episode_parameters=row["instruction"]["episode_parameters"])
    started = time.time()
    env = module.class_decorator(row["task"])
    try:
        random.seed(row["seed"])
        env.setup_demo(now_ep_num=0, seed=row["seed"], is_test=True, **copy.deepcopy(cfg))
        if env.step_lim != limit or env.scene.render_system.device.cuda_id != 0:
            raise ValueError("Reset task limit or render device differs")
        env.set_instruction(instruction)
        initial_observation(env, output / "initial")
        verify_initial(json.loads((Path(row["probe"]) / "reset_0/initial.json").read_text()),
                       json.loads((output / "initial/initial.json").read_text()))
    finally:
        env.close_env()
    atomic_json(output / "result.json", {"goal_id": GOAL_ID, "protocol_id": PROTOCOL, "passed": True,
        "row_sha256": sha256(row_path), "owner": process_identity(os.getpid()), "cuda_uuids": uuids,
        "renderer": renderer, "started_at": started, "ended_at": time.time(),
        "initial_sha256": sha256(output / "initial/initial.json"),
        "rgb_sha256": sha256(output / "initial/rgb.npz"), "policy_actions": 0})


def audit(directory):
    directory = Path(directory).resolve()
    plan = json.loads((directory / "plan.json").read_text())
    scenes = json.loads(Path(plan["scenes"]).read_text())
    if sha256(plan["scenes"]) != plan["scenes_sha256"]:
        raise ValueError("Cross-process scene manifest changed")
    before = json.loads((directory / "environment_before.json").read_text())
    if before != json.loads((directory / "environment_after.json").read_text()):
        raise ValueError("Cross-process reset environment changed")
    if (before["environment_report_sha256"] != plan["environment_sha256"]
            or before["asset_report_sha256"] != plan["asset_sha256"]):
        raise ValueError("Cross-process reset provenance identity differs")
    sources = json.loads((directory / "source_snapshot/manifest.json").read_text())
    if not sources or any(sha256(directory / "source_snapshot" / name) != digest for name, digest in sources.items()):
        raise ValueError("Cross-process executable snapshot changed")
    checks, owners = [], set()
    for index, row in enumerate(selected_rows(scenes)):
        target = directory / f"scene_{index:02d}"
        record = json.loads((target / "result.json").read_text())
        child = json.loads((directory / f"child_{index:02d}.json").read_text())
        row_path = directory / f"row_{index:02d}.json"
        if (json.loads(row_path.read_text()) != row or record["row_sha256"] != sha256(row_path)
                or child["owner"] != record["owner"] or child["exit_code"] != 0 or alive(child["owner"])
                or record["passed"] is not True or record["policy_actions"] != 0
                or record["initial_sha256"] != sha256(target / "initial/initial.json")
                or record["rgb_sha256"] != sha256(target / "initial/rgb.npz")):
            raise ValueError("Fresh-process reset result is missing, live or inconsistent")
        owner_key = (record["owner"]["pid"], record["owner"]["created_at"])
        if owner_key in owners:
            raise ValueError("Each task/condition requires a fresh reset process")
        owners.add(owner_key)
        verify_initial(json.loads((Path(row["probe"]) / "reset_0/initial.json").read_text()),
                       json.loads((target / "initial/initial.json").read_text()))
        checks.append({"task": row["task"], "condition": row["condition"], "scene_id": row["scene_id"],
                       "initial_exact": True, "result_sha256": sha256(target / "result.json"),
                       "owner": record["owner"]})
    return {"goal_id": GOAL_ID, "protocol_id": PROTOCOL, "passed": True, "checks": checks,
            "scene_rows_sha256": scenes["rows_sha256"], "environment_sha256": plan["environment_sha256"],
            "plan_sha256": sha256(directory / "plan.json"),
            "limits": "One model-independent fresh-process reset per task/condition; no policy rollout or mid-episode snapshot"}


def run(scenes_path, asset_report, environment_report, output):
    scenes = verify(scenes_path)
    environment = json.loads(environment_report.read_text())
    if (Path(environment["python"]).resolve() != Path(sys.executable).resolve()
            or environment["goal_id"] != GOAL_ID or environment["passed"] is not True):
        raise ValueError("Use the audited simulator interpreter")
    pool_identity = json.loads((Path(scenes["source_pool"]) / "identity.json").read_text())
    if (pool_identity["asset_report_sha256"] != sha256(asset_report)
            or pool_identity["environment_report_sha256"] != sha256(environment_report)):
        raise ValueError("Reset must use the original scene pool environment and assets")
    if len(verify_cuda_devices()) != 1:
        raise ValueError("One shared simulator device required")
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot_sources(output / "source_snapshot")
    plan = {"goal_id": GOAL_ID, "protocol_id": PROTOCOL, "scenes": str(scenes_path.resolve()),
            "scenes_sha256": sha256(scenes_path), "asset_report": str(asset_report.resolve()),
            "environment_report": str(environment_report.resolve()), "asset_sha256": sha256(asset_report),
            "environment_sha256": sha256(environment_report), "provenance_seconds": 900,
            "reset_timeout_seconds": 600, "max_resets": 20, "automatic_retries": 0}
    atomic_json(output / "plan.json", plan)
    verify_simulator_provenance(plan, output / "environment_before.json")
    for index, row in enumerate(selected_rows(scenes)):
        if Path(os.environ["R2_STOP_FILE"]).exists():
            raise RuntimeError("Cross-process reset stopped; no completion claim")
        row_path = output / f"row_{index:02d}.json"
        atomic_json(row_path, row)
        with (output / f"scene_{index:02d}.log").open("x") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "probe", "--row", str(row_path),
                                      "--output", str(output / f"scene_{index:02d}")], cwd=ROOT,
                                     stdout=log, stderr=subprocess.STDOUT)
            owner = process_identity(child.pid)
            deadline = time.monotonic() + plan["reset_timeout_seconds"]
            try:
                while child.poll() is None:
                    if Path(os.environ["R2_STOP_FILE"]).exists() or time.monotonic() >= deadline:
                        raise RuntimeError("Reset stopped or exceeded finite deadline")
                    time.sleep(0.5)
                if child.returncode != 0:
                    raise RuntimeError(f"Independent reset {index} failed; preserve evidence")
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
                atomic_json(output / f"child_{index:02d}.json", {"owner": owner, "exit_code": child.returncode})
    verify_simulator_provenance(plan, output / "environment_after.json")
    assert_sources_unchanged(sources)
    atomic_json(output / "report.json", audit(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("run", "probe", "audit"))
    parser.add_argument("--row", type=Path)
    parser.add_argument("--scenes", type=Path)
    parser.add_argument("--asset-report", type=Path)
    parser.add_argument("--environment-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.role == "probe":
        probe(args.row, args.output.resolve())
    elif args.role == "audit":
        print(json.dumps(audit(args.output)))
    else:
        run(args.scenes, args.asset_report, args.environment_report, args.output.resolve())


if __name__ == "__main__":
    main()
