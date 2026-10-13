"""Bounded, identity-checked fault injection into a dedicated engineering run."""

import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.common import ARTIFACTS, GOAL_ID, atomic_json, sha256
from scripts.statewam.r2.cpu_job import alive, identity


def check_target(spec, run_id):
    if (spec["goal_id"] != GOAL_ID or spec["run_id"] != run_id
            or spec["phase"] != "engineering" or "fault" not in run_id):
        raise ValueError("Only explicitly named Goal engineering fault runs are eligible")
    command = spec["command"]
    if "scripts/statewam/r2/train.py" not in command or "--updates" not in command:
        raise ValueError("Expected bounded real-model engineering entry")
    if not 1 <= int(command[command.index("--updates") + 1]) <= 8:
        raise ValueError("Unbounded training target")


def open_writers(state, partial):
    if not alive(state.get("supervisor")) or not alive(state.get("child")):
        return []
    supervisor = psutil.Process(state["supervisor"]["pid"])
    leader = psutil.Process(state["child"]["pid"])
    if leader.ppid() != supervisor.pid:
        raise ValueError("Workload leader is not owned by its supervisor")
    writers = []
    for process in leader.children(recursive=True):
        try:
            if not any(value.endswith("scripts/statewam/r2/train.py") for value in process.cmdline()):
                continue
            for opened in process.open_files():
                path = Path(opened.path)
                if path.is_relative_to(partial) and path.suffix == ".pt" and "w" in opened.mode:
                    writers.append({"process": identity(process.pid), "file": str(path),
                                    "bytes": path.stat().st_size, "fd": opened.fd,
                                    "mode": opened.mode})
        except (psutil.NoSuchProcess, FileNotFoundError):
            continue
    return writers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("kill-write", "stop"), required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.run_id):
        raise ValueError("Invalid run id")
    if not 1 <= args.update <= 8 or not 0 < args.timeout <= 3600 or args.output.exists():
        raise ValueError("Invalid bounded probe or existing evidence")
    run = ARTIFACTS / "runs" / args.run_id
    partial = run / "training/checkpoints" / f"update_{args.update:08d}.partial"
    started = time.time()
    result = {"goal_id": GOAL_ID, "target_run": args.run_id, "mode": args.mode,
              "update": args.update, "started_at": started, "action_taken": False,
              "limits": "Fault delivery only; checkpoint validity, resume and release require separate audit"}
    atomic_json(args.output, result)
    previous = {}
    try:
        while time.time() - started < args.timeout:
            if os.environ.get("R2_RUN_DIR") and (Path(os.environ["R2_RUN_DIR"]) / "STOP").exists():
                raise RuntimeError("Fault observer stopped before delivery")
            if not (run / "state.json").exists():
                time.sleep(0.1)
                continue
            spec = json.loads((run / "job.json").read_text())
            check_target(spec, args.run_id)
            state = json.loads((run / "state.json").read_text())
            if state["state"] in ("completed", "failed", "stopped"):
                raise RuntimeError("Target exited before fault trigger")
            if args.mode == "stop":
                progress = run / "training/progress_rank0.json"
                if (progress.exists() and alive(state.get("child"))
                        and json.loads(progress.read_text())["position"]["optimizer_update"] >= args.update):
                    atomic_json(run / "STOP", {"requested_at": time.time(), "reason": "engineering fault probe"})
                    result.update(action_taken=True, action="cooperative_STOP", child=state["child"])
                    break
            elif partial.exists() and not (partial / "COMPLETE.json").exists():
                for row in open_writers(state, partial):
                    key = (row["process"]["pid"], row["process"]["created_at"], row["file"])
                    before = previous.get(key)
                    previous[key] = row["bytes"]
                    if before is None or row["bytes"] <= before:
                        continue
                    # Record intent before delivering the signal; recheck the exact writer identity.
                    result.update(writer=row, prior_bytes=before, job_sha256=sha256(run / "job.json"))
                    atomic_json(args.output, result)
                    if not alive(row["process"]) or (partial / "COMPLETE.json").exists():
                        raise RuntimeError("Writer identity or checkpoint phase changed")
                    psutil.Process(row["process"]["pid"]).send_signal(signal.SIGKILL)
                    result.update(action_taken=True, action="SIGKILL_writer")
                    break
                if result["action_taken"]:
                    break
            time.sleep(0.05)
        if not result["action_taken"]:
            raise TimeoutError("No eligible trigger observed within the registered timeout")
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["ended_at"] = time.time()
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
