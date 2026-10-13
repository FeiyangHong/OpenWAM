"""Audit the registered real-model write-kill, STOP and recovery chain."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.audit_model_resume import require_exact
from scripts.statewam.r2.checkpoints import validate_checkpoint
from scripts.statewam.r2.common import ARTIFACTS, GOAL_ID, atomic_json, sha256
from scripts.statewam.r2.cpu_job import alive


def read(path):
    return json.loads(path.read_text())


def audit():
    names = ("p2-model-fault-write-001", "p2-model-fault-stop-001", "p2-model-fault-finish-001")
    runs = [ARTIFACTS / "runs" / name for name in names]
    positions = (2, 3, 4)
    states = ("failed", "stopped", "completed")
    identities, markers, manifests, costs = [], [], [], []
    for run, update, expected_state in zip(runs, positions, states):
        state = read(run / "state.json")
        if state["state"] != expected_state or any(alive(r) for r in (
                state.get("supervisor"), state.get("child"), *state.get("processes", []))):
            raise ValueError(f"Unexpected live/terminal state: {run}")
        if (expected_state == "failed") != (state["exit_code"] != 0):
            raise ValueError("Unexpected workload exit code")
        identity = read(run / "training/identity.json")
        if identity["updates"] != 4 or identity["global_batch"] != 8 or identity["world_size"] != 4:
            raise ValueError("Wrong registered fault budget")
        checkpoint = run / "training/checkpoints" / f"update_{update:08d}"
        position = validate_checkpoint(checkpoint, identity=identity)
        require_exact(position, {"optimizer_update": update, "next_occurrence": update * 8})
        require_exact(read(checkpoint.parent / "latest-complete.json"), {
            "checkpoint": checkpoint.name, "complete_sha256": sha256(checkpoint / "COMPLETE.json")})
        if expected_state != "failed":
            for rank in range(4):
                summary = read(run / f"training/summary_rank{rank}.json")
                require_exact(summary["identity"], identity)
                require_exact(summary["position"], position)
                if summary["stopped"] != (expected_state == "stopped"):
                    raise ValueError("STOP was not acknowledged by every rank")
        own_devices = [row for row in state["gpu_release_snapshot"] if row["uuid"] in state["gpu_uuids"]]
        if len(own_devices) != 4 or any(row["processes"] for row in own_devices):
            raise ValueError("Missing clean four-GPU release evidence")
        identities.append(identity)
        markers.append(read(checkpoint / "COMPLETE.json"))
        manifests.append(read(run / "training/source_snapshot/manifest.json"))
        performance = read(run / "performance_summary.json")
        costs.append({"run": run.name, "wall_seconds": performance["wall_seconds"],
                      "allocated_gpu_hours": performance["allocated_gpu_hours"]})
    require_exact(identities[0], identities[1])
    require_exact(identities[0], identities[2])
    require_exact(manifests[0], manifests[1])
    require_exact(manifests[0], manifests[2])
    partial = runs[0] / "training/checkpoints/update_00000004.partial"
    if not partial.is_dir() or (partial / "COMPLETE.json").exists():
        raise ValueError("Missing interrupted-write evidence")
    try:
        validate_checkpoint(partial, identity=identities[0])
    except ValueError as error:
        rejection = str(error)
    else:
        raise ValueError("Incomplete checkpoint accepted")
    reference = ARTIFACTS / "runs/p2-model-resume-reference-002"
    require_exact(read(reference / "training/identity.json"), identities[0])
    checkpoint = reference / "training/checkpoints/update_00000004"
    validate_checkpoint(checkpoint, identity=identities[0])
    # Hash inventories were independently recomputed above, not merely trusted.
    require_exact(read(checkpoint / "COMPLETE.json"), markers[2])
    fields = ("position", "samples", "noise_hashes", "metrics", "gradients")
    evidence = []
    for rank in range(4):
        paths = [run / f"training/steps_rank{rank}.jsonl" for run in [reference, *runs]]
        rows = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
        if [len(values) for values in rows] != [8, 8, 2, 2]:
            raise ValueError("Missing, repeated or unexpected microsteps")
        select = lambda values: [{field: row[field] for field in fields} for row in values]
        require_exact(select(rows[0]), select(rows[1]), f"rank{rank}.interrupted")
        require_exact(select(rows[0]), select(rows[1][:4] + rows[2] + rows[3]), f"rank{rank}.recovery")
        evidence.append({"rank": rank, "steps_sha256": [sha256(path) for path in paths],
                         "logical_microsteps": 8, "actual_microsteps": 12})
    actions = {}
    for name, expected in (("model_fault_write_001", "SIGKILL_writer"), ("model_fault_stop_001", "cooperative_STOP")):
        path = ARTIFACTS / "reports" / f"{name}.json"
        record = read(path)
        if record.get("action") != expected or not record["action_taken"]:
            raise ValueError("Missing actual fault delivery")
        actions[name] = sha256(path)
    takeover = ARTIFACTS / "reports/model_fault_takeover_001.json"
    if not read(takeover)["passed"]:
        raise ValueError("Live-run duplicate launch rejection not passed")
    actions["takeover"] = sha256(takeover)
    return {"goal_id": GOAL_ID, "passed": True, "runs": list(names), "identity": identities[0],
            "partial_rejection": rejection, "actions_sha256": actions, "ranks": evidence,
            "final_checkpoint_bytes_identical_to_reference": True,
            "logical_exposures": 32, "actual_exposures": 48, "recomputed_exposures": 16,
            "gpu_runs": costs, "allocated_gpu_hours": sum(row["allocated_gpu_hours"] for row in costs),
            "limits": "Real E3 microbatch1 accumulation2 four-rank ZeRO2; orderly STOP plus actual write-kill. Not client interrupt, other batch configurations, or policy episode recovery. CPU observers/audits have separate cost records."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior audit")
    try:
        result = audit()
    except Exception as error:
        atomic_json(args.output, {"goal_id": GOAL_ID, "passed": False, "error": f"{type(error).__name__}: {error}"})
        raise
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
