"""Run four fixed scene lanes under one GPU-0 lease and shared STOP boundary."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.r2_scene_queue import lane_entries, verify_shard
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256
from scripts.statewam.r2.cpu_job import identity as process_identity


def verify_lane(directory, lane):
    marker = json.loads((directory / "COMPLETE.json").read_text())
    if marker != {"manifest_sha256": sha256(directory / "manifest.json")}:
        raise ValueError("Lane manifest hash differs")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["identity"]["lane"] != lane or manifest["identity"]["entries"] != lane_entries(lane):
        raise ValueError("Completed lane belongs to another fixed scene list")
    if [row["entry"] for row in manifest["shards"]] != lane_entries(lane):
        raise ValueError("Completed lane has missing or reordered shards")
    for row in manifest["shards"]:
        entry = row["entry"]
        expected_path = directory / "shards" / entry["split"] / entry["task"] / entry["condition"]
        if Path(row["path"]).resolve() != expected_path.resolve() or verify_shard(expected_path, entry) != row:
            raise ValueError("Lane does not match its audited shard evidence")
    return {"lane": lane, "path": str(directory.resolve()), "manifest_sha256": marker["manifest_sha256"]}


def run_pool(output, *, asset_report, environment_report, stop_file, resume=False):
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in (
        Path(__file__).resolve(), ROOT / "benchmarks/robotwin/r2_scene_queue.py",
        ROOT / "benchmarks/robotwin/r2_scene_manifest.py", ROOT / "benchmarks/robotwin/r2_scene_probe.py",
        ROOT / "benchmarks/robotwin/r2_scene_provenance.py", ROOT / "benchmarks/robotwin/r2_instructions.py",
        ROOT / "benchmarks/robotwin/eval_policy_wrapper.py",
    )}
    identity = {"goal_id": GOAL_ID, "protocol_id": "r2-scene-pool-v3", "lanes": 4,
                "gpu_layout": "Four independent simulator processes share one physical GPU-0 lease",
                "asset_report_sha256": sha256(asset_report), "environment_report_sha256": sha256(environment_report),
                "sources": sources}
    output = output.resolve()
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "identity.json", identity)
        for name, digest in sources.items():
            target = output / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            if sha256(target) != digest:
                raise ValueError("Scene source changed while taking snapshot")
    elif json.loads((output / "identity.json").read_text()) != identity:
        raise ValueError("Scene pool identity changed on resume")
    with (output / "pool.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "COMPLETE.json").exists():
            raise ValueError("Scene pool already completed")
        status = {"identity": identity, "state": "running", "workers": {}, "completed_lanes": [],
                  "run_directory": os.environ.get("R2_RUN_DIR"), "updated_at": time.time()}
        processes, logs = {}, []
        try:
            for lane in range(4):
                if stop_file.exists():
                    break
                directory = output / f"lane_{lane}"
                if (directory / "COMPLETE.json").exists():
                    status["completed_lanes"].append(verify_lane(directory, lane))
                    continue
                command = [sys.executable, str(ROOT / "benchmarks/robotwin/r2_scene_queue.py"),
                           "--lane", str(lane), "--output", str(directory),
                           "--asset-report", str(asset_report.resolve()),
                           "--environment-report", str(environment_report.resolve())]
                if directory.exists():
                    command.append("--resume")
                log = (output / f"lane_{lane}.log").open("a")
                logs.append(log)
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                processes[lane] = child
                status["workers"][str(lane)] = {"process": process_identity(child.pid), "command": command, "exit_code": None}
            while processes:
                if any(sha256(ROOT / name) != digest for name, digest in sources.items()):
                    raise ValueError("Scene pool source drift")
                for lane, child in list(processes.items()):
                    result = child.poll()
                    if result is None:
                        continue
                    status["workers"][str(lane)]["exit_code"] = result
                    del processes[lane]
                    directory = output / f"lane_{lane}"
                    if result == 0 and (directory / "COMPLETE.json").exists():
                        status["completed_lanes"].append(verify_lane(directory, lane))
                    elif result != 0 or not stop_file.exists():
                        status.update(state="failed", failure=f"Lane {lane} exited {result} without complete evidence")
                        atomic_json(stop_file, {"requested_at": time.time(), "reason": status["failure"]})
                status["updated_at"] = time.time()
                atomic_json(output / "progress.json", status)
                if processes:
                    time.sleep(2)
            if status["state"] == "failed":
                return "failed"
            if len(status["completed_lanes"]) != 4:
                status["state"] = "stopped"
                return "stopped"
            status["state"] = "completed"
            atomic_json(output / "manifest.json", {"identity": identity,
                        "lanes": sorted(status["completed_lanes"], key=lambda row: row["lane"]),
                        "reset_tolerance_frozen": False,
                        "limits": "Model-independent scene preparation, not policy results or frozen evaluation protocol."})
            atomic_json(output / "COMPLETE.json", {"manifest_sha256": sha256(output / "manifest.json")})
            return "completed"
        except BaseException as error:
            status.update(state="failed", failure=f"{type(error).__name__}: {error}")
            atomic_json(stop_file, {"requested_at": time.time(), "reason": status["failure"]})
            raise
        finally:
            for child in processes.values():
                if child.poll() is None:
                    child.terminate()
            for child in processes.values():
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            for log in logs:
                log.close()
            status["updated_at"] = time.time()
            atomic_json(output / "progress.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-report", type=Path, required=True)
    parser.add_argument("--environment-report", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from scripts.statewam.r2.cuda_identity import verify_cuda_devices
    job = json.loads((Path(os.environ["R2_RUN_DIR"]) / "job.json").read_text())
    if job["gpu_indices"] != [0] or len(verify_cuda_devices()) != 1:
        raise ValueError("Pinned simulator pool requires exactly one supervised physical GPU-0 lease")
    result = run_pool(args.output, asset_report=args.asset_report, environment_report=args.environment_report,
                      stop_file=Path(os.environ["R2_STOP_FILE"]), resume=args.resume)
    return 0 if result in ("completed", "stopped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
