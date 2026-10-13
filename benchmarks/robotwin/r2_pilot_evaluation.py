"""Frozen final-checkpoint pilot: two simulator lanes share one physical GPU.

Each block reuses a model service, each episode has a fresh simulator process.
An invalid or unclassified result stops the finite schedule, without retries.
"""

import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.r2_episode_engineering import snapshot_sources, verify_simulator_provenance
from benchmarks.robotwin.r2_episode_ledger import EpisodeLedger
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256, stable_hash
from scripts.statewam.r2.cpu_job import alive, identity as process_identity
from scripts.statewam.r2.cuda_identity import verify_cuda_devices
from scripts.statewam.r2.evaluation_plan import EXECUTION, build_schedule, verify_export
from scripts.statewam.r2.formal_training import load_protocol
from scripts.statewam.r2.source_snapshot import assert_sources_unchanged


def read_execution(path):
    path = Path(path).resolve()
    execution = json.loads(path.read_text())
    schedule_path = Path(execution["schedule"])
    if (execution["goal_id"] != GOAL_ID or sha256(schedule_path) != execution["schedule_sha256"]
            or sha256(execution["protocol"]) != execution["protocol_sha256"]):
        raise ValueError("Evaluation execution identity changed")
    schedule = json.loads(schedule_path.read_text())
    if schedule["execution"] != EXECUTION or schedule["protocol_sha256"] != execution["protocol_sha256"]:
        raise ValueError("Evaluation settings differ from frozen schedule")
    return execution, schedule


def select_block(schedule, block_id):
    blocks = [block for wave in schedule["waves"] for block in wave["blocks"] if block["block_id"] == block_id]
    if len(blocks) != 1:
        raise ValueError("Unknown or duplicate scheduled block")
    return blocks[0]


def completed_episode(ledger_root, episode):
    with EpisodeLedger(ledger_root, episode["identity"]) as ledger:
        attempts = ledger._attempts()
        if not attempts:
            return False
        result = ledger._result(attempts[-1])
        if result is None:
            raise ValueError("Unclassified partial episode; diagnose before resuming")
        if result["outcome"] == "invalid":
            if len(attempts) == 2:
                raise ValueError("Paired result incomplete after its only retry")
            return False
        if ledger.begin() is not None:
            raise ValueError("Completed episode was not deduplicated")
        return True


def wait_child(child, *, deadline, stop_file, progress=None, heartbeat=None):
    while child.poll() is None:
        if stop_file.exists() or time.monotonic() >= deadline:
            raise RuntimeError("Evaluation stopped or exceeded its registered deadline")
        if progress is not None:
            atomic_json(progress, {"heartbeat": time.time(), **(heartbeat or {})})
        time.sleep(0.5)
    if child.returncode != 0:
        raise RuntimeError(f"Evaluation child exited {child.returncode}; preserve evidence")


def finish_children(children, state, *, stop_file):
    deadline = time.monotonic() + EXECUTION["shutdown_seconds"]
    for name, child in children.items():
        try:
            child.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            owner = state["children"][name]["owner"]
            if alive(owner):
                child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if alive(owner):
                    child.kill()
                child.wait(timeout=10)
        state["children"][name]["exit_code"] = child.returncode
    if any(child.returncode != 0 for child in children.values()):
        state["state"] = "failed"
    if state["state"] == "failed" and not stop_file.exists():
        atomic_json(stop_file, {"reason": "evaluation_child_failure", "requested_at": time.time()})


def serve_block(execution_path, block_id, output):
    import torch
    from benchmarks.robotwin.r2_policy_service import build_session, serve

    execution, schedule = read_execution(execution_path)
    block = select_block(schedule, block_id)
    if len(verify_cuda_devices()) != 1:
        raise ValueError("Policy service must see only its assigned device")
    binding = schedule["checkpoint_bindings"][block["group"]]
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    session = build_session(binding["directory"], checkpoint_name=binding["weight_name"],
                            group=block["group"], protocol_sha256=execution["protocol_sha256"])
    if session.binding["checkpoint_sha256"] != binding["weight_sha256"]:
        raise ValueError("Loaded model differs from the frozen final checkpoint")
    atomic_json(output / "service_provenance.json", {**session.provenance, "load_seconds": time.monotonic() - started})
    limit = sum(json.loads((Path(row["scene"]["probe"]) / "reset_0/initial.json").read_text())["metadata"]["step_lim"] + 1
                for row in block["episodes"])
    try:
        asyncio.run(serve(session, port=0, output=output / "service", stop_file=output / "SERVICE_STOP",
                          max_messages=limit, max_seconds=len(block["episodes"]) * EXECUTION["episode_seconds"],
                          idle_seconds=EXECUTION["service_idle_seconds"]))
    finally:
        atomic_json(output / "service_memory.json", {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "total_seconds": time.monotonic() - started})


def simulate(execution_path, block_id, index, output):
    from benchmarks.robotwin.r2_scene_probe import bind_renderer
    from benchmarks.robotwin.r2_rollout import run_episode
    from benchmarks.robotwin.eval_policy_wrapper import bootstrap_robotwin_module

    execution, schedule = read_execution(execution_path)
    block = select_block(schedule, block_id)
    episode = block["episodes"][index]
    if len(verify_cuda_devices()) != 1:
        raise ValueError("Simulator must see only its shared assigned device")
    root = Path(os.environ["ROBOTWIN_PATH"])
    output.mkdir(parents=True, exist_ok=False)
    os.environ["ROBOTWIN_RUNTIME_ROOT"] = str(output / "runtime")
    module = bootstrap_robotwin_module()
    renderer = bind_renderer()
    atomic_json(output / "simulator.json", {"owner": process_identity(os.getpid()), "renderer": renderer,
                                            "identity": episode["identity"]})
    ready = json.loads((output.parent / "service/ready.json").read_text())
    expected = {key: episode["identity"][key] for key in ("group", "checkpoint_sha256", "protocol_sha256")}
    if ready["binding"] != expected:
        raise ValueError("Policy endpoint belongs to another block")
    row = {key: value for key, value in episode["scene"].items() if key not in ("task", "condition")}
    outcome = run_episode(module, robotwin_root=root, scene_row=row, identity=episode["identity"],
        ledger_root=execution["ledger_root"], service_url=f"ws://127.0.0.1:{ready['port']}",
        stop_file=Path(os.environ["R2_STOP_FILE"]))
    atomic_json(output / "result.json", {"identity": episode["identity"], "outcome": outcome,
                                         "ended_at": time.time(), "owner": process_identity(os.getpid())})
    if outcome not in ("success", "failure", "already_complete"):
        raise RuntimeError("Incomplete episode stops the schedule; no automatic retry")


def run_block(execution_path, block_id, output):
    execution, schedule = read_execution(execution_path)
    block = select_block(schedule, block_id)
    uuids = verify_cuda_devices()
    indices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(uuids) != 2:
        raise ValueError("Block requires the simulator and one policy device")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "block.json", block)
    stop_file = Path(os.environ["R2_STOP_FILE"])
    state = {"block_id": block_id, "state": "running", "children": {}, "skipped": [], "completed": []}
    children, logs = {}, []

    def spawn(name, role, python, device, extra=()):
        log = (output / f"{name}.log").open("x")
        logs.append(log)
        child = subprocess.Popen([python, str(Path(__file__).resolve()), role, "--execution", str(execution_path),
            "--block-id", block_id, "--output", str(output if role == "service" else output / name), *extra],
            cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": indices[device], "R2_GPU_UUIDS": uuids[device]},
            stdout=log, stderr=subprocess.STDOUT)
        children[name] = child
        state["children"][name] = {"owner": process_identity(child.pid)}
        atomic_json(output / "state.json", state)
        return child

    try:
        pending = []
        for index, episode in enumerate(block["episodes"]):
            if completed_episode(execution["ledger_root"], episode):
                state["skipped"].append(episode["episode_key"])
            else:
                pending.append((index, episode))
        if pending:
            service = spawn("service", "service", sys.executable, 1)
            deadline = time.monotonic() + EXECUTION["service_startup_seconds"]
            while not (output / "service/ready.json").exists():
                if service.poll() is not None or stop_file.exists() or time.monotonic() >= deadline:
                    raise RuntimeError("Service startup failed, stopped or timed out")
                time.sleep(0.5)
            for index, episode in pending:
                if stop_file.exists() or service.poll() is not None:
                    raise RuntimeError("Stopped or service unavailable before next episode")
                name = f"episode_{index:03d}"
                child = spawn(name, "episode", execution["simulator_python"], 0, ("--index", str(index)))
                wait_child(child, deadline=time.monotonic() + EXECUTION["episode_seconds"], stop_file=stop_file,
                    progress=output / "progress.json", heartbeat={"episode_key": episode["episode_key"], "block_id": block_id})
                if not completed_episode(execution["ledger_root"], episode):
                    raise ValueError("Episode child exited without a complete result")
                state["completed"].append(episode["episode_key"])
                atomic_json(output / "state.json", state)
        state["state"] = "completed"
    except BaseException as error:
        state.update(state="failed", error=f"{type(error).__name__}: {error}")
        if not stop_file.exists():
            atomic_json(stop_file, {"reason": state["error"], "requested_at": time.time()})
        raise
    finally:
        atomic_json(output / "SERVICE_STOP", {"reason": "block_finished", "requested_at": time.time()})
        finish_children(children, state, stop_file=stop_file)
        state["ended_at"] = time.time()
        atomic_json(output / "state.json", state)
        for log in logs:
            log.close()
    if state["state"] != "completed":
        raise RuntimeError("Block lifecycle failed")


def run(protocol_path, export_paths, ledger_root, output):
    protocol_path, ledger_root, output = (Path(path).resolve() for path in (protocol_path, ledger_root, output))
    plan = load_protocol(protocol_path)
    if plan.get("evaluation_execution") != EXECUTION:
        raise ValueError("Execution recipe was not frozen before training")
    if len(export_paths) != 4:
        raise ValueError("Four group export directories required in E0..E3 order")
    bindings = {group: verify_export(path, plan, group) for group, path in zip(plan["groups"], export_paths)}
    scenes = json.loads(Path(plan["evidence"]["scenes"]["path"]).read_text())
    schedule = build_schedule(scenes, bindings, sha256(protocol_path))
    execute_schedule(plan, schedule, protocol_path, ledger_root, output)


def execute_schedule(plan, schedule, protocol_path, ledger_root, output):
    """Common physical lifecycle, also exercised by bounded engineering plans."""
    protocol_path, ledger_root, output = (Path(path).resolve() for path in (protocol_path, ledger_root, output))
    uuids = verify_cuda_devices()
    indices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    job = json.loads((Path(os.environ["R2_RUN_DIR"]) / "job.json").read_text())
    if len(uuids) != 3 or job["gpu_indices"][0] != 0 or len(set(uuids)) != 3:
        raise ValueError("Require one three-device lease with shared simulator on physical GPU 0")
    ledger_root.mkdir(parents=True, exist_ok=True)
    with (ledger_root / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        schedule_path = ledger_root / "schedule.json"
        if schedule_path.exists():
            if json.loads(schedule_path.read_text()) != schedule:
                raise ValueError("Resume changed the frozen evaluation schedule")
        else:
            atomic_json(schedule_path, schedule)
        output.mkdir(parents=True, exist_ok=False)
        sources = snapshot_sources(output / "source_snapshot")
        environment = json.loads(Path(plan["environment"]["path"]).read_text())
        execution = {"goal_id": GOAL_ID, "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path),
            "schedule": str(schedule_path), "schedule_sha256": sha256(schedule_path), "ledger_root": str(ledger_root),
            "simulator_python": environment["python"], "asset_report": plan["assets"]["path"],
            "environment_report": plan["environment"]["path"], "provenance_seconds": EXECUTION["provenance_seconds"]}
        execution_path = output / "execution.json"
        atomic_json(execution_path, execution)
        stop_file = Path(os.environ["R2_STOP_FILE"])
        state = {"goal_id": GOAL_ID, "state": "running", "children": {}, "completed_waves": []}
        children, logs = {}, []
        try:
            for role, target in (("before", "environment_before.json"),):
                log = (output / f"{role}.log").open("x")
                logs.append(log)
                child = subprocess.Popen([execution["simulator_python"], str(Path(__file__).resolve()), "provenance",
                    "--execution", str(execution_path), "--output", str(output / target)], cwd=ROOT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, stdout=log, stderr=subprocess.STDOUT)
                children[role] = child
                state["children"][role] = {"owner": process_identity(child.pid)}
                wait_child(child, deadline=time.monotonic() + EXECUTION["preflight_seconds"], stop_file=stop_file)
            for wave in schedule["waves"]:
                assert_sources_unchanged(sources)
                active = []
                if stop_file.exists():
                    raise RuntimeError("Evaluation stopped before next wave")
                for block in wave["blocks"]:
                    name, lane = block["block_id"], block["lane"]
                    log = (output / f"{name}.log").open("x")
                    logs.append(log)
                    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "block",
                        "--execution", str(execution_path), "--block-id", name, "--output", str(output / name)], cwd=ROOT,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": f"{indices[0]},{indices[lane + 1]}",
                             "R2_GPU_UUIDS": f"{uuids[0]},{uuids[lane + 1]}"}, stdout=log, stderr=subprocess.STDOUT)
                    children[name] = child
                    state["children"][name] = {"owner": process_identity(child.pid)}
                    active.append(child)
                atomic_json(output / "state.json", state)
                deadline = time.monotonic() + EXECUTION["service_startup_seconds"] + max(
                    len(block["episodes"]) for block in wave["blocks"]) * EXECUTION["episode_seconds"] + EXECUTION["shutdown_seconds"]
                while any(child.poll() is None for child in active):
                    if (stop_file.exists() or time.monotonic() >= deadline
                            or any(child.returncode not in (None, 0) for child in active)):
                        raise RuntimeError("Wave stopped, failed or timed out; no automatic retry")
                    atomic_json(output / "progress.json", {"wave": wave["wave"], "heartbeat": time.time()})
                    time.sleep(0.5)
                if any(child.returncode != 0 for child in active):
                    raise RuntimeError("Wave failed")
                state["completed_waves"].append(wave["wave"])
            log = (output / "after.log").open("x")
            logs.append(log)
            child = subprocess.Popen([execution["simulator_python"], str(Path(__file__).resolve()), "provenance",
                "--execution", str(execution_path), "--output", str(output / "environment_after.json")], cwd=ROOT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, stdout=log, stderr=subprocess.STDOUT)
            children["after"] = child
            state["children"]["after"] = {"owner": process_identity(child.pid)}
            wait_child(child, deadline=time.monotonic() + EXECUTION["preflight_seconds"], stop_file=stop_file)
            if json.loads((output / "environment_before.json").read_text()) != json.loads((output / "environment_after.json").read_text()):
                raise ValueError("Evaluation environment changed")
            assert_sources_unchanged(sources)
            state["state"] = "completed"
        except BaseException as error:
            state.update(state="failed", error=f"{type(error).__name__}: {error}")
            if not stop_file.exists():
                atomic_json(stop_file, {"reason": state["error"], "requested_at": time.time()})
            raise
        finally:
            finish_children(children, state, stop_file=stop_file)
            state["ended_at"] = time.time()
            atomic_json(output / "state.json", state)
            for log in logs:
                log.close()
        if state["state"] != "completed":
            raise RuntimeError("Evaluation lifecycle failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("run", "block", "service", "episode", "provenance"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--exports", nargs=4, type=Path)
    parser.add_argument("--ledger-root", type=Path)
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--block-id")
    parser.add_argument("--index", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.role == "run":
        run(args.protocol, args.exports, args.ledger_root, args.output)
    elif args.role == "block":
        run_block(args.execution, args.block_id, args.output.resolve())
    elif args.role == "service":
        serve_block(args.execution, args.block_id, args.output.resolve())
    elif args.role == "episode":
        simulate(args.execution, args.block_id, args.index, args.output.resolve())
    else:
        execution, _ = read_execution(args.execution)
        verify_simulator_provenance(execution, args.output.resolve())


if __name__ == "__main__":
    main()
