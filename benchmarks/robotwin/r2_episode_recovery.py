"""Reconcile registered physical faults and independently audit one bounded retry."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.r2_audit_episode import audit as audit_episode, audit_actions
from benchmarks.robotwin.r2_episode_engineering import PROTOCOL, load_plan
from benchmarks.robotwin.r2_episode_ledger import EpisodeLedger
from benchmarks.robotwin.r2_rollout import verify_initial
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256, stable_hash
from scripts.statewam.r2.cpu_job import alive


def released(run):
    state = json.loads((run / "state.json").read_text())
    if (state["state"] not in ("completed", "stopped", "failed") or alive(state.get("supervisor"))
            or alive(state.get("child")) or any(alive(row) for row in state.get("processes", []))):
        raise ValueError("Physical run is not terminal and fully released")
    devices = {row["uuid"]: row for row in state["gpu_release_snapshot"]}
    if any(uuid not in devices or devices[uuid]["processes"] for uuid in state["gpu_uuids"]):
        raise ValueError("Physical run lacks device release evidence")
    return state


def inspect_fault(run):
    run = Path(run).resolve()
    outer = released(run)
    output = run / "physical"
    plan, row, identity = load_plan(output / "plan.json")
    if plan["protocol_id"] != PROTOCOL or plan["fault"]["mode"] == "none":
        raise ValueError("Require a registered physical fault, not an ordinary policy failure")
    mode = plan["fault"]["mode"]
    state = json.loads((output / "state.json").read_text())
    if state["identity"] != identity or any(alive(row["owner"]) for row in state["children"].values()):
        raise ValueError("Fault coordinator identity or release differs")
    expected = ("stopped", 0) if mode == "cooperative_stop" else ("failed", 1)
    if (outer["state"], outer["exit_code"]) != expected or state["state"] != expected[0]:
        raise ValueError("Physical fault did not exit in its registered manner")
    attempt = Path(plan["ledger_root"]) / stable_hash(identity) / "attempt_001"
    begin = json.loads((attempt / "begin.json").read_text())
    evidence = json.loads((attempt / "fault_injection.json").read_text())
    action = json.loads((output / "fault_action.json").read_text())
    for record in (evidence, action):
        if record["identity"] != identity or record["fault"] != plan["fault"] or record["target"] != begin["owner"]:
            raise ValueError("Fault evidence identity differs from its actual target")
    if (action["executed"] is not True or action["evidence_sha256"] != sha256(attempt / "fault_injection.json")
            or evidence["requested_at"] > action["sent_at"] or action["sent_at"] > state["ended_at"]
            or begin["owner"] != state["children"]["episode"]["owner"]
            or begin["episode_key"] != stable_hash(identity)
            or not 8 <= evidence["progress"]["actions"] <= 32):
        raise ValueError("Fault action was not evidenced at the declared trigger")
    if mode == "cooperative_stop":
        stop = json.loads((run / "STOP").read_text())
        if (stop["evidence_sha256"] != action["evidence_sha256"]
                or any(row["exit_code"] != 0 for row in state["children"].values())
                or not (output / "environment_after_skipped.json").exists()):
            raise ValueError("STOP was not cooperative or lacks explicit interrupted provenance status")
    elif state["children"]["episode"]["exit_code"] != -9:
        raise ValueError("Forced-interruption case did not terminate the simulator with SIGKILL")
    return plan, row, identity, attempt


def reconcile(run):
    plan, _, identity, attempt = inspect_fault(run)
    with EpisodeLedger(plan["ledger_root"], identity) as ledger:
        if plan["fault"]["mode"] == "forced_interrupt" and not (attempt / "result.json").exists():
            ledger.record_interruption(reason="external_interrupt", evidence="fault_injection.json")
        result = ledger._result(attempt)
        if result is None or result["outcome"] != "invalid" or result["reason"] != "external_interrupt":
            raise ValueError("Interrupted attempt is not an evidenced external invalid")
        if len(ledger._attempts()) != 1 or (ledger.directory / "COMPLETE.json").exists():
            raise ValueError("Recovery must preserve exactly one interrupted attempt before retry")
    return {"goal_id": GOAL_ID, "passed": True, "identity": identity, "mode": plan["fault"]["mode"],
            "source_run": str(Path(run).resolve()), "invalid_result_sha256": sha256(attempt / "result.json"),
            "remaining_retries": 1, "limits": "Invalid classification only; no retry launched by reconciliation"}


def audit_pair(first, resumed):
    first, resumed = Path(first).resolve(), Path(resumed).resolve()
    plan, _, identity, attempt = inspect_fault(first)
    released(resumed)
    complete = audit_episode(resumed)
    if complete["identity"] != identity:
        raise ValueError("Retry changed the scene, checkpoint, instruction or policy seed")
    snapshots = []
    for run in (first, resumed):
        root = run / "physical/source_snapshot"
        sources = json.loads((root / "manifest.json").read_text())
        if not sources or any(sha256(root / name) != digest for name, digest in sources.items()):
            raise ValueError("Recovery executable snapshot changed")
        snapshots.append(sources)
    if snapshots[0] != snapshots[1]:
        raise ValueError("Recovery used different executable sources")
    with EpisodeLedger(plan["ledger_root"], identity) as ledger:
        paths = ledger._attempts()
        if len(paths) != 2:
            raise ValueError("Require exactly one interrupted attempt and one retry")
        invalid, result = (ledger._result(path) for path in paths)
        if invalid["outcome"] != "invalid" or invalid["reason"] != "external_interrupt" or result["outcome"] not in ("success", "failure"):
            raise ValueError("Recovery outcome or retry accounting differs")
        result_hash = sha256(paths[-1] / "result.json")
        if ledger.begin() is not None or sha256(paths[-1] / "result.json") != result_hash or len(ledger._attempts()) != 2:
            raise ValueError("Completed retry was not deduplicated")
    verify_initial(json.loads((attempt / "initial/initial.json").read_text()),
                   json.loads((paths[-1] / "initial/initial.json").read_text()))
    prefix = [json.loads(line) for line in (attempt / "actions.jsonl").read_text().splitlines()[:8]]
    repeated = [json.loads(line) for line in (paths[-1] / "actions.jsonl").read_text().splitlines()[:8]]
    if len(prefix) != 8 or len(repeated) != 8:
        raise ValueError("Interrupted run lacks the registered eight-action prefix")
    requests = [json.loads(line)["result"] for line in (first / "physical/service/requests.jsonl").read_text().splitlines()]
    audit_actions(prefix, requests[:9], identity, attempt=1, stats={"policy_actions": 8,
        "step_limit": plan["max_messages"] - 1, "physics_steps": sum(row["physics_steps"] for row in prefix)},
        outcome="invalid", reason="external_interrupt")
    comparisons = []
    for index, (left, right) in enumerate(zip(prefix, repeated)):
        equal = {}
        for key in ("action_index", "eef20", "ee16", "proprio", "rgb_sha256", "physics_steps"):
            equal[key] = left[key] == right[key]
            if index == 0 and key != "physics_steps" and not equal[key]:
                raise ValueError(f"Same-initial-state first action differs: {key}")
        if left["response"]["seeds"] != right["response"]["seeds"]:
            raise ValueError("Retry reset changed the policy noise sequence")
        comparisons.append({"action_index": index, "exact": equal,
            "eef20_max_abs_difference": float(np.abs(np.asarray(left["eef20"]) - np.asarray(right["eef20"])).max())})
    return {"goal_id": GOAL_ID, "protocol_id": PROTOCOL, "passed": True, "mode": plan["fault"]["mode"],
            "identity": identity, "environment_sha256": plan["file_sha256"][plan["environment_report"]],
            "first": str(first), "resumed": str(resumed), "attempts": 2, "initial_exact": True,
            "first_action_exact": True, "prefix_comparisons": comparisons,
            "first_eight_actions_exact": all(row["exact"]["eef20"] for row in comparisons),
            "completed_retry_deduplicated": True,
            "resource_release": True, "resumed_episode_audit": complete,
            "invalid_result_sha256": sha256(attempt / "result.json"), "retry_result_sha256": result_hash,
            "limits": "One registered real fault and one same-scene retry; exact initial input/first action and paired noise; subsequent physical residual differences are reported, not filtered; not formal policy efficacy"}


def bundle(stop_pair, kill_pair, episode_report, dedup_report):
    references = {"stop_pair": Path(stop_pair), "kill_pair": Path(kill_pair),
                  "episode": Path(episode_report), "dedup": Path(dedup_report)}
    reports = {key: json.loads(path.read_text()) for key, path in references.items()}
    for key, mode in (("stop_pair", "cooperative_stop"), ("kill_pair", "forced_interrupt")):
        report = reports[key]
        if report["mode"] != mode or report != audit_pair(report["first"], report["resumed"]):
            raise ValueError("Physical recovery pair report differs from raw evidence")
    episode = reports["episode"]
    if episode != audit_episode(episode["run"]):
        raise ValueError("Original-limit episode report differs from raw evidence")
    dedup = reports["dedup"]
    if (dedup["goal_id"] != GOAL_ID or dedup["passed"] is not True or dedup["attempt_count"] != 1
            or dedup["artifacts_unchanged"] is not True or dedup["begin_returns_none_twice"] is not True):
        raise ValueError("Completed policy failure dedup evidence incomplete")
    with EpisodeLedger(Path(dedup["source_run"]) / "episodes", dedup["identity"]) as ledger:
        attempts = ledger._attempts()
        if (len(attempts) != 1 or ledger.begin() is not None or ledger._result(attempts[0])["outcome"] != "failure"
                or sha256(attempts[0] / "result.json") != dedup["result_sha256"]):
            raise ValueError("Completed real failure was changed or retried")
    plan, _, _ = load_plan(Path(episode["run"]) / "physical/plan.json")
    environment = plan["file_sha256"][plan["environment_report"]]
    if any(reports[key]["environment_sha256"] != environment for key in ("stop_pair", "kill_pair")):
        raise ValueError("Physical acceptance cases used different environments")
    checks = {name: True for name in ("original_limit_episode", "cooperative_stop", "forced_interrupt",
              "same_scene_retry", "completed_failure_dedup", "cache_reset", "video_decode", "resource_release")}
    return {"goal_id": GOAL_ID, "protocol_id": PROTOCOL, "passed": True, "checks": checks,
            "environment_sha256": environment,
            "evidence": {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in references.items()},
            "limits": "Real step-zero engineering recovery on preselected development scenes; no client interrupt-button claim, formal policy result or multi-task repeatability claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("reconcile", "audit", "bundle"))
    parser.add_argument("--first", type=Path)
    parser.add_argument("--resumed", type=Path)
    parser.add_argument("--stop-pair", type=Path)
    parser.add_argument("--kill-pair", type=Path)
    parser.add_argument("--episode-report", type=Path)
    parser.add_argument("--dedup-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing physical recovery evidence")
    if args.role == "bundle":
        result = bundle(args.stop_pair, args.kill_pair, args.episode_report, args.dedup_report)
    else:
        result = reconcile(args.first) if args.role == "reconcile" else audit_pair(args.first, args.resumed)
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
