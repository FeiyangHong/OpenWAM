"""One bounded real-model/SAPIEN engineering episode, never a final evaluation.

The outer GPU supervisor owns exactly two devices and all descendants. The
coordinator partitions that lease between a policy process and a simulator.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.r2_episode_ledger import validate_identity
from benchmarks.robotwin.r2_scene_manifest import inspect_probe
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256, stable_hash
from scripts.statewam.r2.cpu_job import alive, identity as process_identity
from scripts.statewam.r2.cuda_identity import verify_cuda_devices
from scripts.statewam.r2.source_snapshot import capture_sources, assert_sources_unchanged

PROTOCOL = "r2-physical-episode-engineering-v3"
LIFECYCLE_PROTOCOL = "r2-physical-episode-engineering-v2"
LEGACY_PROTOCOL = "r2-physical-episode-engineering-v1"


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if (plan["goal_id"] != GOAL_ID or plan["protocol_id"] not in (PROTOCOL, LIFECYCLE_PROTOCOL, LEGACY_PROTOCOL)
            or plan["purpose"] != "engineering" or plan["group"] != "E0"
            or plan["checkpoint_name"] != "checkpoint_step_0.safetensors"
            or plan["automatic_retries"] != 0 or plan["max_episodes"] != 1):
        raise ValueError("Require the single step-zero E0 engineering plan, without retries")
    for key in ("startup_seconds", "episode_seconds", "service_seconds", "idle_seconds"):
        if type(plan[key]) is not int or not 1 <= plan[key] <= 3600:
            raise ValueError("Require finite bounded engineering deadlines")
    if plan["protocol_id"] != LEGACY_PROTOCOL:
        if (type(plan["preflight_seconds"]) is not int or not 900 <= plan["preflight_seconds"] <= 1800
                or type(plan["provenance_seconds"]) is not int or not 1 <= plan["provenance_seconds"] <= 900
                or plan["preflight_seconds"] <= plan["provenance_seconds"]
                or plan["episode_seconds"] <= plan["provenance_seconds"]):
            raise ValueError("Require coordinated preflight and postcheck deadlines")
    if plan["protocol_id"] == PROTOCOL:
        fault = plan["fault"]
        if (set(fault) != {"mode", "after_actions", "latest_action"}
                or fault["mode"] not in ("none", "cooperative_stop", "forced_interrupt")
                or type(fault["after_actions"]) is not int or type(fault["latest_action"]) is not int
                or (fault["mode"] == "none" and (fault["after_actions"], fault["latest_action"]) != (0, 0))
                or (fault["mode"] != "none" and (fault["after_actions"], fault["latest_action"]) != (8, 32))):
            raise ValueError("Require a bounded, predeclared engineering fault")
    probe = Path(plan["scene_probe"])
    record = json.loads((probe / "scene_probe.json").read_text())
    if not 200000 <= record["seed"] < 400000:
        raise ValueError("Engineering cannot use final scenes")
    row = inspect_probe(probe, task=record["task"], condition=record["condition"], seed=record["seed"])
    if not row["expert_eligible"] or stable_hash(row) != plan["scene_row_sha256"]:
        raise ValueError("Prepared engineering scene identity differs")
    row["scene_id"] = f"engineering/{record['task']}/{record['condition']}/{record['seed']}"
    for name, digest in plan["file_sha256"].items():
        if sha256(name) != digest:
            raise ValueError(f"Engineering input content changed: {name}")
    checkpoint = Path(plan["checkpoint_dir"]) / plan["checkpoint_name"]
    required = {str(checkpoint), str(checkpoint.parent / "config.yaml"),
                str(checkpoint.parent / "normalization_stats.npy"), plan["asset_report"], plan["environment_report"]}
    if set(plan["file_sha256"]) != required:
        raise ValueError("Incomplete engineering input inventory")
    initial = json.loads((probe / "reset_0/initial.json").read_text())
    limit = initial["metadata"]["step_lim"]
    if type(limit) is not int or plan["max_messages"] != limit + 1:
        raise ValueError("Service budget must preserve the original task limit plus reset")
    identity = validate_identity({"goal_id": GOAL_ID, "group": plan["group"],
        "checkpoint_sha256": plan["file_sha256"][str(checkpoint)], "protocol_sha256": sha256(path),
        "scene_manifest_sha256": plan["scene_row_sha256"], "instruction_sha256": row["instruction"]["text_sha256"],
        "task": record["task"], "condition": record["condition"], "scene_id": row["scene_id"],
        "policy_seed": row["policy_seed"], "split": "development", "replicate": 0})
    return plan, row, identity


def snapshot_sources(output):
    sources = capture_sources(output)
    for directory in (ROOT / "benchmarks/robotwin", ROOT / "benchmarks/utils"):
        for path in sorted(directory.glob("*.py")):
            name = str(path.relative_to(ROOT))
            sources[name] = sha256(path)
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if sha256(target) != sources[name]:
                raise ValueError("Source changed during engineering snapshot")
    atomic_json(output / "manifest.json", sources)
    return sources


def service(plan, identity, output):
    import torch
    from benchmarks.robotwin.r2_policy_service import build_session, serve

    if len(verify_cuda_devices()) != 1:
        raise ValueError("Service requires its single assigned GPU")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    session = build_session(plan["checkpoint_dir"], checkpoint_name=plan["checkpoint_name"],
                            group=plan["group"], protocol_sha256=identity["protocol_sha256"])
    if session.binding["checkpoint_sha256"] != identity["checkpoint_sha256"]:
        raise ValueError("Loaded service checkpoint differs")
    atomic_json(output / "service_provenance.json", {**session.provenance, "load_seconds": time.monotonic() - started})
    try:
        asyncio.run(serve(session, port=0, output=output / "service", stop_file=output / "SERVICE_STOP",
                          max_messages=plan["max_messages"], max_seconds=plan["service_seconds"],
                          idle_seconds=plan["idle_seconds"]))
    finally:
        atomic_json(output / "service_memory.json", {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "total_seconds": time.monotonic() - started})


def verify_simulator_provenance(plan, output):
    """Keep package discovery independent of SAPIEN's pkg_resources imports."""
    command = [sys.executable, "-c",
        "from pathlib import Path; import sys; "
        "from benchmarks.robotwin.r2_scene_queue import verify_provenance; "
        "from scripts.statewam.r2.common import atomic_json; "
        "atomic_json(Path(sys.argv[3]), verify_provenance(Path(sys.argv[1]), Path(sys.argv[2])))",
        plan["asset_report"], plan["environment_report"], str(output)]
    if output.exists():
        raise FileExistsError("Preserve existing simulator provenance evidence")
    subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=True,
                   timeout=plan.get("provenance_seconds", 900))
    return json.loads(output.read_text())


def episode(plan, row, identity, output):
    from benchmarks.robotwin.r2_scene_probe import bind_renderer
    from benchmarks.robotwin.r2_rollout import run_episode

    if len(verify_cuda_devices()) != 1:
        raise ValueError("Simulator requires its single assigned GPU")
    provenance = json.loads((output / "environment_before.json").read_text())
    if (provenance["asset_report_sha256"] != plan["file_sha256"][plan["asset_report"]]
            or provenance["environment_report_sha256"] != plan["file_sha256"][plan["environment_report"]]):
        raise ValueError("Preflight provenance differs from episode plan")
    os.environ["ROBOTWIN_RUNTIME_ROOT"] = str(output / "runtime")
    from benchmarks.robotwin.eval_policy_wrapper import bootstrap_robotwin_module, _robotwin_git, VERIFIED_ROBOTWIN_COMMIT
    root = Path(os.environ["ROBOTWIN_PATH"]).resolve()
    if _robotwin_git(str(root), "rev-parse", "HEAD") != VERIFIED_ROBOTWIN_COMMIT:
        raise ValueError("Unverified simulator source")
    module = bootstrap_robotwin_module()
    renderer = bind_renderer()
    atomic_json(output / "simulator_provenance.json", {**provenance, "renderer": renderer})
    ready = json.loads((output / "service/ready.json").read_text())
    if ready["binding"] != {key: identity[key] for key in ("group", "checkpoint_sha256", "protocol_sha256")}:
        raise ValueError("Service ready identity differs")
    outcome = run_episode(module, robotwin_root=root, scene_row=row, identity=identity,
                          ledger_root=plan["ledger_root"], service_url=f"ws://127.0.0.1:{ready['port']}",
                          stop_file=Path(os.environ["R2_STOP_FILE"]))
    atomic_json(output / "episode_control_complete.json", {"identity": identity, "outcome": outcome,
                                                           "ended_at": time.time()})
    if outcome in ("invalid", "stopped") and Path(os.environ["R2_STOP_FILE"]).exists():
        atomic_json(output / "environment_after_skipped.json", {"reason": "external_stop", "outcome": outcome,
                    "limits": "Interrupted attempt only; a completed retry still requires full provenance checks"})
    elif verify_simulator_provenance(plan, output / "environment_after.json") != provenance:
        raise ValueError("Simulator provenance changed during episode")
    atomic_json(output / "episode_result.json", {"identity": identity, "outcome": outcome,
                                                "formal_policy_result": False, "ended_at": time.time()})
    return 0 if outcome in ("success", "failure", "already_complete", "stopped", "invalid") else 1


def child_environment(index, *, indices, uuids):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=indices[index], R2_GPU_UUIDS=uuids[index])
    return env


def maybe_inject_fault(plan, identity, output, simulation):
    fault = plan["fault"]
    if fault["mode"] == "none":
        return False
    attempt = Path(plan["ledger_root"]) / stable_hash(identity) / "attempt_001"
    evidence = attempt / "fault_injection.json"
    if evidence.exists():
        record = json.loads(evidence.read_text())
        if record["identity"] != identity or record["fault"] != fault:
            raise ValueError("Prior fault evidence belongs to another episode or plan")
        return False
    if not (attempt / "progress.json").exists():
        return False
    progress = json.loads((attempt / "progress.json").read_text())
    if progress["actions"] < fault["after_actions"]:
        return False
    if progress["actions"] > fault["latest_action"] or (attempt / "result.json").exists():
        raise RuntimeError("Fault trigger window missed; do not interrupt a completed episode")
    begin = json.loads((attempt / "begin.json").read_text())
    owner = process_identity(simulation.pid)
    if (begin["owner"] != owner or owner is None or not alive(owner)
            or begin["episode_key"] != stable_hash(identity) or progress["attempt"] != 1):
        raise ValueError("Fault target is not the live owned simulator attempt")
    record = {"goal_id": GOAL_ID, "identity": identity, "fault": fault, "target": owner,
              "progress": progress, "requested_at": time.time(), "coordinator": process_identity(os.getpid())}
    atomic_json(evidence, record)
    if fault["mode"] == "cooperative_stop":
        atomic_json(Path(os.environ["R2_STOP_FILE"]), {"reason": "registered_engineering_stop",
                    "requested_at": time.time(), "evidence_sha256": sha256(evidence)})
    else:
        simulation.kill()
    atomic_json(output / "fault_action.json", {"identity": identity, "fault": fault, "target": owner,
                "evidence_sha256": sha256(evidence), "sent_at": time.time(), "executed": True})
    return True


def coordinate(plan_path, plan, identity, output):
    from scripts.statewam.r2.gpu_resources import DeviceMonitor

    if plan["protocol_id"] != PROTOCOL:
        raise ValueError("Legacy episode plans are audit-only; register the revised lifecycle")
    uuids = verify_cuda_devices()
    indices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(uuids) != 2 or len(indices) != 2:
        raise ValueError("Engineering requires one two-device lease")
    monitor = DeviceMonitor()
    try:
        physical_zero = next(row["uuid"] for row in monitor.snapshot() if row["index"] == 0)
    finally:
        monitor.close()
    if uuids[0] != physical_zero:
        raise ValueError("Simulator must use physical GPU 0 to avoid renderer spill")
    output.mkdir(parents=True, exist_ok=False)
    sources = snapshot_sources(output / "source_snapshot")
    atomic_json(output / "plan.json", plan)
    atomic_json(output / "identity.json", identity)
    stop_file = Path(os.environ["R2_STOP_FILE"])
    state = {"goal_id": GOAL_ID, "protocol_id": PROTOCOL, "identity": identity,
             "started_at": time.time(), "state": "starting", "children": {}}
    children, logs = {}, []

    def spawn(role, python, index):
        log = (output / f"{role}.log").open("x")
        logs.append(log)
        env = ({**os.environ, "CUDA_VISIBLE_DEVICES": "", "R2_GPU_UUIDS": ""} if index is None
               else child_environment(index, indices=indices, uuids=uuids))
        child = subprocess.Popen([python, str(Path(__file__).resolve()), role, "--plan", str(plan_path),
                                  "--output", str(output)], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
        children[role] = child
        state["children"][role] = {"owner": process_identity(child.pid), "gpu_uuid": None if index is None else uuids[index]}
        atomic_json(output / "state.json", state)
        return child

    try:
        preflight = spawn("preflight", plan["simulator_python"], None)
        deadline = time.monotonic() + plan["preflight_seconds"]
        while preflight.poll() is None:
            if stop_file.exists() or time.monotonic() >= deadline:
                raise RuntimeError("Preflight stopped or exceeded deadline")
            state["heartbeat"] = time.time()
            atomic_json(output / "state.json", state)
            time.sleep(0.5)
        if preflight.returncode != 0 or not (output / "environment_before.json").exists():
            raise RuntimeError("Preflight failed; do not start policy service")
        policy = spawn("service", sys.executable, 1)
        deadline = time.monotonic() + plan["startup_seconds"]
        while not (output / "service/ready.json").exists():
            if policy.poll() is not None or stop_file.exists() or time.monotonic() >= deadline:
                raise RuntimeError("Service failed, stopped or exceeded startup deadline")
            time.sleep(0.5)
        simulation = spawn("episode", plan["simulator_python"], 0)
        state["state"] = "running"
        deadline = time.monotonic() + plan["episode_seconds"]
        while simulation.poll() is None:
            maybe_inject_fault(plan, identity, output, simulation)
            if stop_file.exists():
                state["state"] = "stopping"
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Engineering episode exceeded wall deadline; not a task failure")
            control_complete = output / "episode_control_complete.json"
            if control_complete.exists():
                if json.loads(control_complete.read_text())["identity"] != identity:
                    raise ValueError("Control completion identity differs")
                if not (output / "SERVICE_STOP").exists():
                    atomic_json(output / "SERVICE_STOP", {"requested_at": time.time(), "reason": "control_complete"})
            if policy.poll() is not None:
                service_state = json.loads((output / "service/state.json").read_text())
                if policy.returncode != 0 or (not control_complete.exists() and service_state["messages"] != plan["max_messages"]):
                    raise RuntimeError("Service exited before the original episode budget")
            state["heartbeat"] = time.time()
            atomic_json(output / "state.json", state)
            time.sleep(0.5)
        if state["state"] != "stopping":
            if simulation.returncode != 0:
                raise RuntimeError("Simulator episode failed; preserve and diagnose partial evidence")
            result = json.loads((output / "episode_result.json").read_text())
            state.update(state="completed", outcome=result["outcome"])
        assert_sources_unchanged(sources)
    except BaseException as error:
        state.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        # The supervisor observes this request and bounds cleanup even if a
        # native simulator or model call cannot reach a cooperative boundary.
        atomic_json(output / "SERVICE_STOP", {"requested_at": time.time(), "reason": "bounded_episode_finished"})
        if state["state"] != "completed" and not stop_file.exists():
            atomic_json(stop_file, {"requested_at": time.time(), "reason": "bounded_episode_finished",
                                    "engineering_state": state["state"]})
        for role, child in children.items():
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            state["children"][role]["exit_code"] = child.returncode
        for log in logs:
            log.close()
        if state["state"] in ("completed", "stopping") and any(child.returncode != 0 for child in children.values()):
            state.update(state="failed", error="A child did not exit cleanly after the episode")
        elif state["state"] == "stopping":
            state["state"] = "stopped"
        state["ended_at"] = time.time()
        atomic_json(output / "state.json", state)
    if state["state"] == "failed":
        raise RuntimeError(state["error"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("run", "preflight", "service", "episode"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan, row, identity = load_plan(args.plan.resolve())
    output = args.output.resolve()
    if args.role == "run":
        coordinate(args.plan.resolve(), plan, identity, output)
    elif args.role == "preflight":
        verify_simulator_provenance(plan, output / "environment_before.json")
    elif args.role == "service":
        service(plan, identity, output)
    else:
        return episode(plan, row, identity, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
