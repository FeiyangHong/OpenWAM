"""Bounded, restartable model-independent RoboTwin scene preparation."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.statewam.r2.common import GOAL_ID, TASKS, atomic_json, sha256, stable_hash

CONDITIONS = ("demo_clean", "demo_randomized")
COUNTS = {"development": 5, "final": 20}


def completed_attempt(directory):
    attempt, probe = directory / "attempt.json", directory / "probe/scene_probe.json"
    if not attempt.exists() or not probe.exists():
        return False
    status, result = json.loads(attempt.read_text()), json.loads(probe.read_text())
    return (status.get("exit_code") == 0 and result.get("completed") is True
            and not result.get("stopped") and not result.get("error"))


def seed_start(task, split):
    if task not in TASKS or split not in COUNTS:
        raise ValueError("Unknown task or scene split")
    return (200000 if split == "development" else 400000) + 10000 * TASKS.index(task)


def inspect_probe(directory, *, task, condition, seed):
    result_path = directory / "scene_probe.json"
    result = json.loads(result_path.read_text())
    if result["goal_id"] != GOAL_ID or (result["task"], result["condition"], result["seed"]) != (task, condition, seed):
        raise ValueError("Scene probe identity mismatch")
    if not result["completed"] or result.get("error") or result.get("stopped"):
        raise ValueError("Incomplete scene probe")
    evidence = {"scene_probe.json": sha256(result_path), "config.json": sha256(directory / "config.json")}
    config = json.loads((directory / "config.json").read_text())
    if stable_hash(config) != result["config_sha256"]:
        raise ValueError("Probe configuration integrity failed")
    row = {"seed": seed, "probe": str(directory.resolve()), "expert_eligible": result["expert_eligible"],
           "evidence_sha256": evidence}
    if not row["expert_eligible"]:
        if result["rejection"]["kind"] not in ("unstable_initial_scene", "expert_unsuccessful"):
            raise ValueError("Infrastructure failure cannot be a scene rejection")
        row["rejection"] = result["rejection"]
        return row
    # Do not silently skip an expert-valid scene because replay is inconsistent.
    if result["numeric_differences"] or any(
        value[field] != 0 for value in result["rgb_differences"].values() for field in ("max", "mean", "fraction_changed")
    ):
        raise ValueError("Expert-valid scene reset differs; diagnose before freezing a tolerance")
    views = {"head_camera", "left_camera", "right_camera"}
    if set(result["rgb_differences"]) != views:
        raise ValueError("Incomplete three-view reset evidence")
    from benchmarks.robotwin.r2_scene_probe import numeric_difference
    import hashlib
    import numpy as np

    states = []
    for index in range(2):
        reset_path = directory / f"reset_{index}"
        metadata = json.loads((reset_path / "initial.json").read_text())
        if set(metadata["rgb"]) != views:
            raise ValueError("Missing reset view")
        with np.load(reset_path / "rgb.npz", allow_pickle=False) as images:
            if set(images.files) != views:
                raise ValueError("Missing saved RGB view")
            for view in views:
                image = images[view]
                if (image.dtype != np.uint8 or list(image.shape) != metadata["rgb"][view]["shape"]
                        or hashlib.sha256(image.tobytes()).hexdigest() != metadata["rgb"][view]["sha256"]):
                    raise ValueError("Saved RGB integrity mismatch")
        states.append(metadata)
        for name in ("initial.json", "rgb.npz"):
            evidence[f"reset_{index}/{name}"] = sha256(reset_path / name)
    if numeric_difference(states[0]["metadata"], states[1]["metadata"]) or states[0]["rgb"] != states[1]["rgb"]:
        raise ValueError("Saved reset evidence is not exactly repeated")
    instruction = json.loads((directory / "instruction.json").read_text())
    payload = {key: value for key, value in instruction.items() if key != "identity_sha256"}
    if (instruction["task"] != task or stable_hash(payload) != instruction["identity_sha256"]
            or stable_hash(instruction["text"]) != instruction["text_sha256"]):
        raise ValueError("Instruction identity mismatch")
    evidence["instruction.json"] = sha256(directory / "instruction.json")
    row.update(instruction=instruction, initial_metadata_sha256=stable_hash(states[0]["metadata"]),
               initial_rgb=states[0]["rgb"], config_sha256=result["config_sha256"],
               policy_seed=int(stable_hash(["r2-policy-seed-v1", task, condition, seed])[:8], 16))
    return row


def run_shard(output, *, task, condition, split, stop_file, resume=False):
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    first, target = seed_start(task, split), COUNTS[split]
    root = Path(os.environ["ROBOTWIN_PATH"]).resolve()
    from benchmarks.robotwin.eval_policy_wrapper import _robotwin_git, VERIFIED_ROBOTWIN_COMMIT
    if (_robotwin_git(str(root), "rev-parse", "HEAD") != VERIFIED_ROBOTWIN_COMMIT
            or _robotwin_git(str(root), "status", "--porcelain", "--untracked-files=no", "--",
                             "envs", "script", "task_config", "description", "policy")):
        raise ValueError("Scene preparation requires unchanged pinned RoboTwin sources")
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in (
        Path(__file__).resolve(), ROOT / "benchmarks/robotwin/r2_scene_probe.py",
        ROOT / "benchmarks/robotwin/r2_instructions.py", ROOT / "benchmarks/robotwin/eval_policy_wrapper.py",
    )}
    identity = {"goal_id": GOAL_ID, "protocol_id": "r2-scene-preparation-v2", "task": task,
                "condition": condition, "split": split, "first_seed": first, "target": target,
                "max_candidates": 1000, "max_infrastructure_attempts_per_seed": 2,
                "candidate_timeout_seconds": 900, "reset_comparison": "exact_or_stop_for_diagnosis",
                "robotwin_root": str(root), "python": sys.executable, "sources": sources}
    output = output.resolve()
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "identity.json", identity)
    elif json.loads((output / "identity.json").read_text()) != identity:
        raise ValueError("Scene shard source/configuration changed on resume")
    with (output / "shard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "COMPLETE.json").exists():
            raise ValueError("Shard already complete; use its manifest, do not rerun")
        accepted, rejected = [], []
        status = {"identity": identity, "run_id": os.environ.get("R2_RUN_DIR"), "state": "running",
                  "accepted": accepted, "rejected": rejected, "updated_at": time.time()}
        try:
            for seed in range(first, first + 1000):
                if stop_file.exists():
                    status["state"] = "stopped"
                    break
                if any(sha256(ROOT / name) != digest for name, digest in sources.items()):
                    raise ValueError("Scene preparation source drift")
                candidate = output / "candidates" / str(seed)
                attempts = sorted(candidate.glob("attempt_*"))
                complete = [path for path in attempts if completed_attempt(path)]
                if len(complete) > 1:
                    raise ValueError("Duplicate successful candidate attempts")
                if complete:
                    directory = complete[0]
                else:
                    if len(attempts) >= 2:
                        raise ValueError(f"Candidate {seed} exhausted infrastructure attempts")
                    directory = candidate / f"attempt_{len(attempts) + 1:03d}"
                    directory.mkdir(parents=True, exist_ok=False)
                    attempt = {"seed": seed, "started_at": time.time(), "exit_code": None}
                    atomic_json(directory / "attempt.json", attempt)
                    command = [sys.executable, str(ROOT / "benchmarks/robotwin/r2_scene_probe.py"),
                               "--task", task, "--condition", condition, "--seed", str(seed),
                               "--output", str(directory / "probe")]
                    print(f"candidate task={task} condition={condition} split={split} seed={seed}", flush=True)
                    with (directory / "console.log").open("x") as console:
                        try:
                            result = subprocess.run(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT,
                                                    timeout=identity["candidate_timeout_seconds"], check=False)
                            attempt["exit_code"] = result.returncode
                        except subprocess.TimeoutExpired:
                            attempt["error"] = "candidate_timeout"
                            raise
                        finally:
                            attempt["ended_at"] = time.time()
                            atomic_json(directory / "attempt.json", attempt)
                    if stop_file.exists():
                        status["state"] = "stopped"
                        break
                    if result.returncode != 0:
                        raise RuntimeError(f"Unclassified candidate error at {directory}; inspect before resume")
                row = inspect_probe(directory / "probe", task=task, condition=condition, seed=seed)
                if row["expert_eligible"]:
                    row["scene_id"] = f"{split}/{task}/{condition}/{seed}"
                    accepted.append(row)
                else:
                    rejected.append(row)
                status["next_seed"] = seed + 1
                status["updated_at"] = time.time()
                atomic_json(output / "progress.json", status)
                if len(accepted) == target:
                    status["state"] = "completed"
                    break
            else:
                status["state"] = "candidate_limit_exhausted"
            if status["state"] == "completed":
                manifest = {"identity": identity, "accepted": accepted, "rejected": rejected,
                            "count": len(accepted), "reset_tolerance_frozen": False,
                            "limits": "Model-independent candidates only. Exact reset evidence does not replace cross-task tolerance review or policy reproducibility tests."}
                atomic_json(output / "manifest.json", manifest)
                atomic_json(output / "COMPLETE.json", {"manifest_sha256": sha256(output / "manifest.json")})
            return status["state"]
        except BaseException as error:
            status.update(state="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            status["updated_at"] = time.time()
            atomic_json(output / "progress.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--split", choices=COUNTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    state = run_shard(args.output, task=args.task, condition=args.condition, split=args.split,
                      stop_file=Path(os.environ["R2_STOP_FILE"]), resume=args.resume)
    return 0 if state in ("completed", "stopped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
