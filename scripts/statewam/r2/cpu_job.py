"""Detached, locked and monitored CPU preparation jobs. Not a GPU job launcher."""

import argparse
import fcntl
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
from scripts.statewam.r2.common import ARTIFACTS, GOAL_ID, ROOT, atomic_json, sha256


def identity(pid):
    try:
        process = psutil.Process(pid)
        return {"pid": pid, "created_at": process.create_time()}
    except psutil.NoSuchProcess:
        return None


def alive(record):
    if not record:
        return False
    try:
        process = psutil.Process(record["pid"])
        return process.create_time() == record["created_at"] and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def tree_sample(pid):
    processes = []
    try:
        root = psutil.Process(pid)
        for process in [root] + root.children(recursive=True):
            try:
                io = process.io_counters()
                cpu = process.cpu_times()
                processes.append({**identity(process.pid), "rss_bytes": process.memory_info().rss,
                                  "cpu_seconds": cpu.user + cpu.system,
                                  "read_bytes": io.read_bytes, "write_bytes": io.write_bytes})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except psutil.NoSuchProcess:
        pass
    return {"time": time.time(), "processes": processes,
            "rss_bytes": sum(p["rss_bytes"] for p in processes)}


def run(directory):
    spec = json.loads((directory / "job.json").read_text())
    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "state.json").exists():
            raise ValueError("Existing run evidence; resume with a new run id")
        state = {"goal_id": GOAL_ID, "protocol_id": spec["protocol_id"], "run_id": directory.name,
                 "supervisor": identity(os.getpid()), "session": spec["session"],
                 "state": "running", "started_at": time.time(), "exit_code": None}
        atomic_json(directory / "state.json", state)
        requested_stop = False

        def request_stop(*_):
            nonlocal requested_stop
            requested_stop = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        # CPU-only preparation never needs credentials; don't expose inherited tokens.
        for key in list(env):
            if any(word in key.upper() for word in ("TOKEN", "SECRET", "API_KEY")):
                env.pop(key)
        samples = []
        child = None
        try:
            with (directory / "console.log").open("x") as console, (directory / "resource_samples.jsonl").open("x") as stream:
                child = subprocess.Popen(spec["command"], cwd=spec["cwd"], env=env,
                                         stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
                state["child"] = identity(child.pid)
                state["process_group"] = child.pid
                stop_at = None
                while True:
                    started = time.monotonic()
                    sample = tree_sample(os.getpid())
                    sample.update(goal_id=GOAL_ID, protocol_id=spec["protocol_id"], run_id=directory.name,
                                  gpu_uuids=[], cuda_visible_devices="", sampling_seconds=time.monotonic() - started)
                    samples.append(sample)
                    stream.write(json.dumps(sample) + "\n")
                    stream.flush()
                    state["heartbeat"] = sample["time"]
                    if requested_stop or (directory / "STOP").exists():
                        if stop_at is None:
                            stop_at = time.monotonic()
                            state["state"] = "stopping"
                            if alive(state["child"]):
                                os.killpg(child.pid, signal.SIGTERM)
                        elif time.monotonic() - stop_at > spec["stop_grace_seconds"] and alive(state["child"]):
                            os.killpg(child.pid, signal.SIGKILL)
                    atomic_json(directory / "state.json", state)
                    if child.poll() is not None:
                        break
                    time.sleep(spec["sample_interval_seconds"])
                state["exit_code"] = child.returncode
                state["state"] = "stopped" if stop_at is not None else ("completed" if child.returncode == 0 else "failed")
        except BaseException as error:
            state["state"] = "failed"
            state["error_type"] = type(error).__name__
            if child and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=spec["stop_grace_seconds"])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
        finally:
            state["ended_at"] = time.time()
            atomic_json(directory / "state.json", state)
            intervals = [b["time"] - a["time"] for a, b in zip(samples, samples[1:])]
            atomic_json(directory / "performance_summary.json", {
                **state, "wall_seconds": state["ended_at"] - state["started_at"],
                "sample_count": len(samples), "peak_process_tree_rss_bytes": max((s["rss_bytes"] for s in samples), default=None),
                "sampling_seconds_total": sum(s["sampling_seconds"] for s in samples),
                "max_sample_gap_seconds": max(intervals, default=None),
                "gpu_hours": 0, "gpu_accounting": "CPU preparation, CUDA hidden; no GPU workloads launched",
                "memory_note": "Sum of process RSS, shared pages may be counted more than once",
            })
        return 0 if state["state"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("launch", "run", "status", "stop"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, default=ARTIFACTS / "runs")
    parser.add_argument("--protocol-id", default="r2-data-v1")
    arguments = sys.argv[1:]
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    command = arguments[separator + 1:]
    args = parser.parse_args(arguments[:separator])
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.run_id):
        raise ValueError("Invalid run id")
    directory = args.root.resolve() / args.run_id
    if args.operation == "launch":
        if not command:
            raise ValueError("Missing command after --")
        if any(re.search(r"hf_[A-Za-z0-9]+|(?:token|password|secret|api[_-]key)=", x, re.I) for x in command):
            raise ValueError("Credentials must not appear in commands")
        directory.mkdir(parents=True, exist_ok=False)
        session = f"sw-r2-{args.run_id}"
        atomic_json(directory / "job.json", {
            "goal_id": GOAL_ID, "protocol_id": args.protocol_id, "run_id": args.run_id,
            "session": session, "command": command, "cwd": str(ROOT),
            "code_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "sample_interval_seconds": 2, "stop_grace_seconds": 15,
            "dependencies": [], "gpu_uuids": [],
        })
        worker = [sys.executable, str(Path(__file__).resolve()), "run", "--run-id", args.run_id, "--root", str(args.root.resolve())]
        subprocess.run(["tmux", "new-session", "-d", "-s", session, shlex.join(worker)], check=True)
        print(json.dumps({"session": session, "directory": str(directory)}))
    elif args.operation == "run":
        return run(directory)
    else:
        state = json.loads((directory / "state.json").read_text())
        state["supervisor_live_now"] = alive(state.get("supervisor"))
        state["child_live_now"] = alive(state.get("child"))
        if args.operation == "stop":
            atomic_json(directory / "STOP", {"requested_at": time.time()})
        print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
