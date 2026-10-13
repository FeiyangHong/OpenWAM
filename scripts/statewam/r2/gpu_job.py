"""Detached GPU workloads with project-wide leases, monitoring and cooperative stop."""

import argparse
import ctypes
import fcntl
import importlib.metadata
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.common import ARTIFACTS, GOAL_ID, ROOT, atomic_json, sha256, stable_hash
from scripts.statewam.r2.cpu_job import alive, identity, tree_sample
from scripts.statewam.r2.gpu_resources import DeviceMonitor, lease_process_spill, performance_summary, validate_lease

LEASES = ARTIFACTS / "manifests/gpu_leases.json"
ENV_KEYS = {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONUTF8", "ROBOTWIN_PATH", "ROBOTWIN_PYTHON",
            "__EGL_VENDOR_LIBRARY_FILENAMES", "CUBLAS_WORKSPACE_CONFIG", "TOKENIZERS_PARALLELISM", "NCCL_DEBUG"}


def update_lease(run_id, row, *, devices=None):
    LEASES.parent.mkdir(parents=True, exist_ok=True)
    with LEASES.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        leases = json.loads(LEASES.read_text()) if LEASES.exists() else {}
        if devices is not None:
            validate_lease(row["gpu_uuids"], leases, devices)
        if row is None:
            leases.pop(run_id, None)
        else:
            leases[run_id] = row
        atomic_json(LEASES, leases)


def descendants():
    result = []
    for process in psutil.Process().children(recursive=True):
        record = identity(process.pid)
        if record and alive(record):
            result.append(record)
    return result


def signal_owned(records, sig):
    for record in records:
        if alive(record):
            try:
                process = psutil.Process(record["pid"])
                if process.create_time() == record["created_at"]:
                    process.send_signal(sig)
            except psutil.NoSuchProcess:
                pass


def process_gpu_ids(processes):
    ids = {row["pid"] for row in processes}
    for row in processes:
        try:
            for line in Path(f"/proc/{row['pid']}/status").read_text().splitlines():
                if line.startswith("NSpid:"):
                    ids.update(int(value) for value in line.split()[1:])
        except FileNotFoundError:
            pass
    return ids


def run(directory):
    spec = json.loads((directory / "job.json").read_text())
    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "state.json").exists():
            raise ValueError("Existing run state; inspect it and use a new attempt id")
        if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Unable to enable child subreaper")
        started = time.time()
        state = {"goal_id": GOAL_ID, "protocol_id": spec["protocol_id"], "run_id": directory.name,
                 "phase": spec["phase"], "state": "pending", "session": spec["session"],
                 "supervisor": identity(os.getpid()), "started_at": started, "exit_code": None,
                 "gpu_uuids": spec["gpu_uuids"], "processes": []}
        atomic_json(directory / "state.json", state)
        stop_requested = False

        def stop(*_):
            nonlocal stop_requested
            stop_requested = True

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        monitor, child, lease_acquired = None, None, False
        samples, failure = [], None
        try:
            monitor = DeviceMonitor()
            update_lease(directory.name, state, devices=monitor.snapshot())
            lease_acquired = True
            env = dict(os.environ)
            for key in list(env):
                if any(word in key.upper() for word in ("TOKEN", "SECRET", "API_KEY")):
                    env.pop(key)
            # DeepSpeed 0.18's memory query parses these as integer NVML indices.
            # Workloads must verify the CUDA UUID mapping before allocating models.
            env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, spec["gpu_indices"])),
                       CUDA_DEVICE_ORDER="PCI_BUS_ID", R2_GPU_UUIDS=",".join(spec["gpu_uuids"]),
                       PYTHON_EXEC=sys.executable,
                       PYTHONUNBUFFERED="1", PYTHONUTF8="1",
                       OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
                       WANDB_MODE="disabled", R2_STOP_FILE=str(directory / "STOP"), R2_RUN_DIR=str(directory))
            env.update(spec["environment"])
            state["state"] = "running"
            with (directory / "console.log").open("x") as console, (directory / "resource_samples.jsonl").open("x") as stream:
                child = subprocess.Popen(spec["command"], cwd=spec["cwd"], env=env,
                                         stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
                state["child"] = identity(child.pid)
                state["process_group"] = child.pid
                stop_at = None
                while True:
                    tick = time.monotonic()
                    owned = descendants()
                    state["processes"] = owned
                    gpu_pids = process_gpu_ids(owned)
                    devices = monitor.snapshot()
                    used = {row["uuid"] for row in devices if any(p["pid"] in gpu_pids for p in row["processes"])}
                    unmatched = [{"uuid": row["uuid"], **process} for row in devices
                                 if row["uuid"] in spec["gpu_uuids"] for process in row["processes"]
                                 if process["pid"] not in gpu_pids]
                    outside = used - set(spec["gpu_uuids"])
                    spill = lease_process_spill(devices, spec["gpu_uuids"])
                    outside.update(row["uuid"] for row in spill)
                    if outside:
                        failure = f"Workload or lease-associated NVML PID used GPUs outside its lease: {sorted(outside)}"
                        stop_requested = True
                    sample = tree_sample(os.getpid())
                    sample.update(goal_id=GOAL_ID, protocol_id=spec["protocol_id"], run_id=directory.name,
                                  phase=spec["phase"], used_uuids=sorted(used),
                                  unmatched_gpu_processes=unmatched,
                                  lease_associated_outside_processes=spill,
                                  process_attribution_complete=not unmatched,
                                  devices=[row for row in devices if row["uuid"] in set(spec["gpu_uuids"]) | used | outside],
                                  sampling_seconds=time.monotonic() - tick)
                    stream.write(json.dumps(sample) + "\n")
                    stream.flush()
                    samples.append({key: value for key, value in sample.items() if key != "processes"})
                    state["heartbeat"] = sample["time"]
                    returncode = child.poll()
                    if returncode is not None:
                        # poll() can reap the leader after the earlier sample.
                        # Only fresh, creation-time-checked survivors imply a leak.
                        owned = descendants()
                        state["leader_exit_observation"] = {"time": time.time(), "exit_code": returncode,
                                                            "live_descendants": owned}
                    if returncode is not None and owned:
                        failure = failure or "Workload leader exited with live descendants"
                        stop_requested = True
                    if stop_requested or (directory / "STOP").exists():
                        if stop_at is None:
                            stop_at = time.monotonic()
                            state["state"] = "stopping"
                            if not (directory / "STOP").exists():
                                atomic_json(directory / "STOP", {"requested_at": time.time(), "reason": failure or "signal"})
                        elapsed = time.monotonic() - stop_at
                        if elapsed > spec["stop_grace_seconds"] + 10:
                            signal_owned(owned, signal.SIGKILL)
                        elif elapsed > spec["stop_grace_seconds"]:
                            signal_owned(owned, signal.SIGTERM)
                    atomic_json(directory / "state.json", state)
                    update_lease(directory.name, state)
                    if returncode is not None and not descendants():
                        state["exit_code"] = returncode
                        state["state"] = "failed" if failure or returncode != 0 else "completed"
                        if stop_at is not None and failure is None and returncode == 0:
                            state["state"] = "stopped"
                        break
                    if stop_at is not None and time.monotonic() - stop_at > spec["stop_grace_seconds"] + 30:
                        raise TimeoutError("Owned processes did not exit after stop escalation")
                    time.sleep(max(0.1, spec["sample_interval_seconds"] - (time.monotonic() - tick)))
        except BaseException as error:
            failure = failure or f"{type(error).__name__}: {error}"
            state["state"] = "failed"
            signal_owned(descendants(), signal.SIGTERM)
            try:
                if child:
                    child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                signal_owned(descendants(), signal.SIGKILL)
                child.wait(timeout=10)
        finally:
            survivors = descendants()
            state["processes"] = survivors
            state["ended_at"] = time.time()
            state["failure"] = failure
            if child and child.returncode is not None:
                state["exit_code"] = child.returncode
            if survivors:
                state["state"] = "failed"
                state["failure"] = f"{failure or ''}; owned processes remain alive"
            state["gpu_release_snapshot"] = monitor.snapshot() if monitor else None
            atomic_json(directory / "state.json", state)
            summary = performance_summary(samples, gpu_uuids=spec["gpu_uuids"], started=started, ended=state["ended_at"])
            if not lease_acquired:
                summary["allocated_gpu_hours"] = 0
            atomic_json(directory / "performance_summary.json", {**state, **summary})
            if lease_acquired:
                update_lease(directory.name, state if survivors else None)
            if monitor:
                monitor.close()
        return 0 if state["state"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("launch", "run", "status", "stop"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--phase", default="engineering", choices=("engineering", "profiling", "calibration", "training", "evaluation"))
    parser.add_argument("--protocol-id", default="r2-engineering-v1")
    parser.add_argument("--stop-grace-seconds", type=float, default=120)
    parser.add_argument("--env", action="append", default=[])
    arguments = sys.argv[1:]
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    command = arguments[separator + 1:]
    args = parser.parse_args(arguments[:separator])
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.run_id):
        raise ValueError("Invalid run id")
    directory = ARTIFACTS / "runs" / args.run_id
    if args.operation == "launch":
        if not command or args.stop_grace_seconds < 0:
            raise ValueError("Require a command and nonnegative stop grace")
        if any(re.search(r"hf_[A-Za-z0-9]+|(?:token|password|secret|api[_-]key)=|^--(?:hf-)?token$", value, re.I) for value in command):
            raise ValueError("Credentials cannot appear in the command")
        environment = dict(value.split("=", 1) for value in args.env)
        if set(environment) - ENV_KEYS:
            raise ValueError("Environment override outside the allowlist")
        monitor = DeviceMonitor()
        try:
            devices = monitor.snapshot()
        finally:
            monitor.close()
        by_key = {str(row["index"]): row for row in devices}
        by_key.update({row["uuid"]: row for row in devices})
        selected = [by_key[token.strip()] for token in args.gpus.split(",")]
        uuids = [row["uuid"] for row in selected]
        indices = [row["index"] for row in selected]
        validate_lease(uuids, json.loads(LEASES.read_text()) if LEASES.exists() else {}, devices)
        directory.mkdir(parents=True, exist_ok=False)
        session = f"sw-r2-{args.run_id}"
        spec = {"goal_id": GOAL_ID, "protocol_id": args.protocol_id, "run_id": args.run_id,
                "phase": args.phase, "session": session, "command": command, "environment": environment,
                "cwd": str(ROOT), "gpu_uuids": uuids, "gpu_indices": indices, "sample_interval_seconds": 2,
                "stop_grace_seconds": args.stop_grace_seconds, "dependencies": [],
                "nvml_binding_version": importlib.metadata.version("nvidia-ml-py"),
                "code_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
        spec["spec_sha256"] = stable_hash(spec)
        atomic_json(directory / "job.json", spec)
        worker = [sys.executable, str(Path(__file__).resolve()), "run", "--run-id", args.run_id]
        subprocess.run(["tmux", "new-session", "-d", "-s", session, shlex.join(worker)], check=True)
        print(json.dumps({"session": session, "directory": str(directory), "gpu_uuids": uuids}))
    elif args.operation == "run":
        return run(directory)
    else:
        state = json.loads((directory / "state.json").read_text())
        if args.operation == "stop":
            atomic_json(directory / "STOP", {"requested_at": time.time()})
        state["supervisor_live_now"] = alive(state.get("supervisor"))
        state["child_live_now"] = alive(state.get("child"))
        state["live_processes_now"] = [row for row in state.get("processes", []) if alive(row)]
        print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
