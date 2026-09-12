"""Revalidate the R1 base, Foundation weights and pinned simulator identities."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.common import GOAL_ID, ROOT, TASKS, atomic_json, sha256

R1 = "98089d1fa612b7d1f0349ae68b3b8b0ef690aef7"
PLAN = "368654c874cbbadc19fa18ec955e17982c7a0081"
SIMULATOR = Path("/gaozt-test1/fyhong/RoboTwin-statewam-r1")
SIM_COMMIT = "0aeea2d669c0f8516f4d5785f0aa33ba812c14b4"
FOUNDATION = ROOT / "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model"
WEIGHT_HASH = "180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d"


def command(argv, cwd=ROOT):
    return subprocess.check_output(argv, cwd=cwd, text=True).strip()


def file_record(path):
    return {"path": str(path), "resolved_path": str(path.resolve()),
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    for ancestor in (R1, PLAN):
        subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, "HEAD"], cwd=ROOT, check=True)
    if command(["git", "rev-parse", "HEAD"], SIMULATOR) != SIM_COMMIT:
        raise ValueError("Simulator HEAD changed")
    if command(["git", "status", "--porcelain", "--untracked-files=no"], SIMULATOR):
        raise ValueError("Unexpected tracked simulator changes")
    weight = file_record(FOUNDATION / "checkpoint_step_154000.safetensors")
    if weight["sha256"] != WEIGHT_HASH:
        raise ValueError("Foundation checkpoint hash mismatch")
    print("Foundation SHA256 verified", flush=True)
    sidecars = [file_record(p) for p in sorted(FOUNDATION.rglob("*")) if p.is_file() and p.suffix != ".safetensors"]
    simulator_paths = [SIMULATOR / "envs" / f"{task}.py" for task in TASKS]
    simulator_paths += [SIMULATOR / p for p in (
        "script/eval_policy.py", "envs/_base_task.py", "envs/robot/planner.py",
        "task_config/demo_clean.yml", "task_config/demo_randomized.yml",
        "assets/embodiments/aloha-agilex/curobo_left.yml", "assets/embodiments/aloha-agilex/curobo_right.yml",
    )]
    simulator_python = "/gaozt-test1/fyhong/envs/robotwin/bin/python"
    packages_code = "import importlib.metadata,json,platform,torch; print(json.dumps({'python':platform.python_version(),'torch_runtime':torch.__version__,'cuda_runtime':torch.version.cuda,'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}))"
    simulator_environment = json.loads(command([simulator_python, "-c", packages_code]))
    simulator_environment["executable"] = simulator_python
    for name, version in {"torch": "2.4.1", "sapien": "3.0.0b1", "mplib": "0.2.1"}.items():
        if simulator_environment["packages"].get(name) != version:
            raise ValueError(f"Simulator dependency changed: {name}")
    if simulator_environment["torch_runtime"] != "2.4.1+cu121":
        raise ValueError("Simulator torch runtime changed")
    import torch

    model_packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    if torch.__version__ != "2.7.1+cu128":
        raise ValueError("Model torch version changed")
    asset_archives = [file_record(Path("/gaozt-test1/datasets/robotwin/upstream-785feb15") / name)
                      for name in ("objects.zip", "background_texture.zip", "embodiments.zip")]
    previous = json.loads((ROOT / "statewam_artifacts/robotwin/manifests/downloads.json").read_text())
    for record in asset_archives:
        if record["sha256"] != previous["archives"][record["path"]]["sha256"]:
            raise ValueError("Simulator source asset archive changed")
    result = {"goal_id": GOAL_ID, "protocol_id": "r2-data-v1", "run_id": args.output.parent.name,
              "r1_base": R1, "plan_base": PLAN, "head": command(["git", "rev-parse", "HEAD"]),
              "branch": command(["git", "branch", "--show-current"]),
              "working_tree": command(["git", "status", "--short"]),
              "foundation": weight, "foundation_sidecars": sidecars,
              "pretraining_coverage": "Historical contamination absence not established by the checkpoint name/config",
              "model_environment": {"executable": sys.executable, "python": platform.python_version(), "packages": model_packages,
                                    "torch_runtime": torch.__version__, "cuda_runtime": torch.version.cuda},
              "simulator_environment": simulator_environment, "simulator_commit": SIM_COMMIT,
              "simulator_git_common_dir": command(["git", "rev-parse", "--git-common-dir"], SIMULATOR),
              "simulator_files": [file_record(p) for p in simulator_paths], "asset_archives": asset_archives,
              "asset_scope": "Source archives verified; full extracted asset identity and behavior still require P1 scene audit",
              "gpu_snapshot": command(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv"]),
              "started_at": started, "ended_at": time.time(), "status": "source_identity_passed"}
    atomic_json(args.output, result)
    print(f"Source identities recorded: {args.output}", flush=True)


if __name__ == "__main__":
    main()
