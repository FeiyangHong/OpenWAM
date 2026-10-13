"""Audit the complete pilot ledger and physical/service evidence before analysis."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.r2_audit_episode import audit_actions, require_conversion_environment
from benchmarks.robotwin.r2_episode_ledger import EpisodeLedger
from benchmarks.robotwin.r2_policy_protocol import reset_identity, validate_reset_ack
from benchmarks.robotwin.r2_rollout import verify_initial
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256, stable_hash
from scripts.statewam.r2.cpu_job import alive
from scripts.statewam.r2.evaluation_plan import scheduled_episodes, build_schedule, verify_export
from scripts.statewam.r2.formal_training import load_protocol
from scripts.statewam.r2.paired_statistics import summarize


def audit_trace(actions, requests, identity, *, attempt, result, rollout):
    outcome, reason, stats = result["outcome"], result["reason"], result["stats"]
    if outcome == "success" or reason == "action_limit":
        audit_actions(actions, requests, identity, attempt=attempt, stats=stats, outcome=outcome, reason=reason)
        return
    expected = reset_identity(identity, attempt=attempt)
    if outcome != "failure" or reason not in ("nonfinite_action", "model_error"):
        raise ValueError("Unrecognized policy failure; do not convert it to infrastructure invalid")
    if len(requests) != len(actions) + 2:
        raise ValueError("Policy failure lacks exactly one terminal service error")
    error = requests[-1]
    code = "policy_output_error" if reason == "nonfinite_action" else "internal_error"
    if (error.get("type") != "error" or error.get("code") != code
            or rollout["evidence"] != {"code": code, "message": error["message"]}):
        raise ValueError("Policy failure reason differs from recorded service error")
    validate_reset_ack(requests[0], expected)
    if actions:
        audit_actions(actions, requests[:-1], identity, attempt=attempt, stats=stats,
                      outcome="invalid", reason="trace_only")
    elif stats["policy_actions"] != 0:
        raise ValueError("Missing physical action trace")


def audit_result(ledger_root, episode, requests):
    identity = episode["identity"]
    with EpisodeLedger(ledger_root, identity) as ledger:
        paths = ledger._attempts()
        if not paths:
            raise ValueError("Missing pilot episode")
        results = [ledger._result(path) for path in paths]
        if any(result is None for result in results) or results[-1]["outcome"] not in ("success", "failure"):
            raise ValueError("Incomplete paired result; cannot silently drop it")
        if len(results) == 2 and results[0]["outcome"] != "invalid":
            raise ValueError("A policy result was retried")
        if any(alive(json.loads((path / "begin.json").read_text())["owner"]) for path in paths):
            raise ValueError("Episode owner remains live")
        result, path = results[-1], paths[-1]
        if ledger.begin() is not None:
            raise ValueError("Completed result did not deduplicate")
        stats = result["stats"]
        rollout = json.loads((path / "rollout.json").read_text())
        if (rollout["identity"] != identity or rollout["stats"] != stats
                or (rollout["outcome"], rollout["reason"]) != (result["outcome"], result["reason"])):
            raise ValueError("Rollout and ledger differ")
        initial = json.loads((path / "initial/initial.json").read_text())
        prepared = json.loads((Path(episode["scene"]["probe"]) / "reset_0/initial.json").read_text())
        verify_initial(prepared, initial)
        if stats["step_limit"] != prepared["metadata"]["step_lim"]:
            raise ValueError("Original task limit was changed")
        if sha256(path / "video.mp4") != stats["video"]["sha256"]:
            raise ValueError("Video digest differs")
        actions = [json.loads(line) for line in (path / "actions.jsonl").read_text().splitlines()] if (path / "actions.jsonl").exists() else []
        if actions != rollout["actions"]:
            raise ValueError("Physical action log differs from rollout")
        attempt = len(paths)
        key = (episode["episode_key"], attempt)
        if key not in requests:
            raise ValueError("Missing service ledger for the completed attempt")
        audit_trace(actions, requests[key], identity, attempt=attempt, result=result, rollout=rollout)
        stream = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,nb_read_frames,codec_name", "-of", "json", str(path / "video.mp4")],
            capture_output=True, text=True, check=True, timeout=120)
        if json.loads(stream.stdout)["streams"] != [stats["video"]["stream"]]:
            raise ValueError("Video stream or frame count differs")
        data = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path / "video.mp4"), "-vf", "scale=32:32",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True, timeout=120).stdout
        frames = np.frombuffer(data, dtype=np.uint8).reshape(-1, 32, 32, 3)
        if len(frames) != stats["video"]["frames"] or any(frame.std() <= 1 for frame in frames):
            raise ValueError("Blank or truncated video")
        return {"identity": identity, "outcome": result["outcome"], "reason": result["reason"], "audited": True,
                "result_sha256": sha256(path / "result.json"), "attempt": attempt, "stats": stats,
                "attempt_directory": str(path), "initial_exact": True}


def request_inventory(runs, schedule, plan):
    inventory, run_evidence = {}, []
    blocks = {block["block_id"]: block for wave in schedule["waves"] for block in wave["blocks"]}
    for run in runs:
        run = Path(run).resolve()
        outer = json.loads((run / "state.json").read_text())
        if (outer["state"] not in ("completed", "failed", "stopped") or alive(outer.get("supervisor"))
                or alive(outer.get("child")) or any(alive(row) for row in outer.get("processes", []))):
            raise ValueError("Evaluation run is live or lacks terminal evidence")
        released = {row["uuid"]: row["processes"] for row in outer["gpu_release_snapshot"]}
        if (len(set(outer["gpu_uuids"])) != 3 or not set(outer["gpu_uuids"]).issubset(released)
                or any(released[uuid] for uuid in outer["gpu_uuids"])):
            raise ValueError("Evaluation devices were not released")
        if outer["state"] == "completed" and outer["exit_code"] != 0:
            raise ValueError("Completed evaluation has a nonzero exit")
        output = run / "evaluation"
        execution = json.loads((output / "execution.json").read_text())
        if (execution["schedule_sha256"] != stable_file_schedule(schedule, execution["schedule"])
                or execution["protocol_sha256"] != schedule["protocol_sha256"]):
            raise ValueError("Evaluation run belongs to another schedule")
        sources = json.loads((output / "source_snapshot/manifest.json").read_text())
        if (sources != plan["source_sha256"]
                or any(sha256(output / "source_snapshot" / name) != digest for name, digest in sources.items())):
            raise ValueError("Evaluation source evidence changed")
        before = json.loads((output / "environment_before.json").read_text())
        if (before["environment_report_sha256"] != plan["environment"]["sha256"]
                or before["asset_report_sha256"] != plan["assets"]["sha256"]):
            raise ValueError("Evaluation provenance differs from frozen environment/assets")
        if outer["state"] == "completed":
            if (json.loads((output / "state.json").read_text())["state"] != "completed"
                    or before != json.loads((output / "environment_after.json").read_text())):
                raise ValueError("Completed evaluation lifecycle/provenance differs")
        for path in sorted(output.glob("wave_*_lane_*/service/requests.jsonl")):
            directory = path.parent.parent
            block = blocks.get(directory.name)
            if block is None or json.loads((directory / "block.json").read_text()) != block:
                raise ValueError("Service block differs from the frozen schedule")
            state = json.loads((directory / "state.json").read_text())
            if (state["block_id"] != block["block_id"] or state["state"] not in ("completed", "failed")
                    or any(alive(row["owner"]) for row in state["children"].values())):
                raise ValueError("Service block has no terminal released lifecycle")
            binding = {"group": block["group"], "protocol_sha256": schedule["protocol_sha256"],
                       "checkpoint_sha256": schedule["checkpoint_bindings"][block["group"]]["weight_sha256"]}
            service = json.loads((path.parent / "state.json").read_text())
            if service["binding"] != binding:
                raise ValueError("Service loaded a different group/checkpoint/protocol")
            records = [json.loads(line) for line in path.read_text().splitlines()]
            if [row["message_index"] for row in records] != list(range(1, len(records) + 1)):
                raise ValueError("Service message log has gaps or duplicates")
            if state["state"] == "completed" and (
                    service["state"] != "stopped" or service["messages"] != len(records)
                    or any(row["exit_code"] != 0 for row in state["children"].values())):
                raise ValueError("Completed service lifecycle or message accounting differs")
            episodes = {row["episode_key"]: row for row in block["episodes"]}
            current = None
            for row in records:
                message = row["result"]
                if message["type"] == "reset_ack":
                    current = (message["episode_key"], message["attempt"])
                    if current[0] not in episodes:
                        raise ValueError("Service evaluated an episode outside its registered block")
                    validate_reset_ack(message, reset_identity(episodes[current[0]]["identity"], attempt=current[1]))
                    if current in inventory:
                        raise ValueError("An episode attempt had duplicate service reset")
                    inventory[current] = []
                if current is None:
                    raise ValueError("Service message before reset")
                inventory[current].append(message)
        run_evidence.append({"run": str(run), "state_sha256": sha256(run / "state.json"),
                             "source_sha256": stable_hash(sources), "completed": outer["state"] == "completed"})
    if not run_evidence or not run_evidence[-1]["completed"] or len({row["source_sha256"] for row in run_evidence}) != 1:
        raise ValueError("Need final completed lifecycle and identical sources across resumes")
    return inventory, run_evidence


def stable_file_schedule(schedule, path):
    if json.loads(Path(path).read_text()) != schedule:
        raise ValueError("Schedule changed")
    return sha256(path)


def audit(ledger_root, runs, environment_path):
    require_conversion_environment(json.loads(Path(environment_path).read_text()))
    ledger_root = Path(ledger_root).resolve()
    schedule = json.loads((ledger_root / "schedule.json").read_text())
    execution = json.loads((Path(runs[-1]) / "evaluation/execution.json").read_text())
    plan = load_protocol(execution["protocol"])
    if sha256(environment_path) != plan["environment"]["sha256"]:
        raise ValueError("Audit interpreter evidence belongs to another environment")
    bindings = {group: verify_export(schedule["checkpoint_bindings"][group]["directory"], plan, group)
                for group in plan["groups"]}
    scenes = json.loads(Path(plan["evidence"]["scenes"]["path"]).read_text())
    if build_schedule(scenes, bindings, sha256(execution["protocol"])) != schedule:
        raise ValueError("Pilot ledger does not match the frozen complete schedule")
    requests, provenance = request_inventory(runs, schedule, plan)
    records = {row["episode_key"]: audit_result(ledger_root, row, requests) for row in scheduled_episodes(schedule)}
    report = summarize(schedule, records)
    report.update(passed=True, schedule_sha256=sha256(ledger_root / "schedule.json"),
                  run_evidence=provenance, records=records)
    for pair in report["repeatability"]:
        left, right = [Path(records[key]["attempt_directory"]) for key in pair["episode_keys"]]
        a = json.loads((left / "rollout.json").read_text())["actions"]
        b = json.loads((right / "rollout.json").read_text())["actions"]
        comparisons = [{"action_index": index,
                        "eef20_max_abs_difference": float(np.max(np.abs(np.asarray(x["eef20"]) - np.asarray(y["eef20"])))),
                        "proprio_exact": x["proprio"] == y["proprio"], "rgb_exact": x["rgb_sha256"] == y["rgb_sha256"],
                        "physics_steps_exact": x["physics_steps"] == y["physics_steps"]}
                       for index, (x, y) in enumerate(zip(a, b))]
        pair.update(initial_exact=True, actions_recorded=[len(a), len(b)], common_prefix=comparisons,
                    limits="Physical residual differences retained, not filtered; no full-trajectory determinism assumption")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing pilot audit")
    atomic_json(args.output, audit(args.ledger_root, args.runs, args.environment))


if __name__ == "__main__":
    main()
