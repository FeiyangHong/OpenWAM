"""Audit a closed physical engineering episode independently of its summary."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin.r2_episode_engineering import LEGACY_PROTOCOL, load_plan
from benchmarks.robotwin.r2_episode_ledger import artifact_inventory, validate_outcome
from benchmarks.robotwin.r2_policy_protocol import reset_identity, validate_action_response, validate_reset_ack
from benchmarks.robotwin.r2_rollout import verify_initial
from benchmarks.utils.action_conversion import eef20d_to_ee16d
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256, stable_hash
from scripts.statewam.r2.cpu_job import alive


def require_conversion_environment(environment):
    if (environment["goal_id"] != GOAL_ID or environment["passed"] is not True
            or environment["packages"]["numpy"] != np.__version__
            or Path(environment["python"]).resolve() != Path(sys.executable).resolve()):
        raise ValueError("Exact EEF conversion audit requires the recorded simulator Python/NumPy environment")


def audit_actions(actions, requests, identity, *, attempt, stats, outcome, reason):
    expected = reset_identity(identity, attempt=attempt)
    if len(requests) != len(actions) + 1:
        raise ValueError("Service ledger and physical actions differ")
    validate_reset_ack(requests[0], expected)
    if len(actions) != stats["policy_actions"] or not 0 < len(actions) <= stats["step_limit"]:
        raise ValueError("Policy action count differs from original limit")
    if stats["physics_steps"] < sum(row["physics_steps"] for row in actions):
        raise ValueError("Physics step count does not cover all controls")
    if outcome == "failure" and (reason != "action_limit" or len(actions) != stats["step_limit"]):
        raise ValueError("This audit requires a full original-limit failure or genuine success")
    for index, (row, response) in enumerate(zip(actions, requests[1:])):
        validate_action_response(response, expected, index)
        if row["action_index"] != index or row["response"] != response or row["eef20"] != response["action"]:
            raise ValueError("Physical action does not match ordered service response")
        value = np.asarray(row["eef20"], dtype=np.float32)
        if value.shape != (20,) or not np.isfinite(value).all():
            raise ValueError("Nonfinite or malformed physical action")
        if not np.array_equal(eef20d_to_ee16d(value), np.asarray(row["ee16"])):
            raise ValueError("EEF action conversion differs")
        if row["physics_steps"] < 0 or row["control_seconds"] < 0 or row["request_seconds"] < 0:
            raise ValueError("Invalid control timing or physics count")
        generation = response["generation"]
        if index % 32:
            if generation is not None:
                raise ValueError("Buffered action regenerated")
        else:
            if generation["chunk_index"] != index // 32 or generation["seeds"] != response["seeds"]:
                raise ValueError("Chunk identity differs")
            if index + 32 <= len(actions):
                chunk = [row["eef20"] for row in actions[index:index + 32]]
                if generation["actions_sha256"] != stable_hash(chunk):
                    raise ValueError("Full chunk content differs from service hash")


def audit(run):
    run = Path(run).resolve()
    state = json.loads((run / "state.json").read_text())
    if (state["state"] != "completed" or state["exit_code"] != 0 or alive(state.get("supervisor"))
            or alive(state.get("child")) or any(alive(row) for row in state.get("processes", []))):
        raise ValueError("Require completed and released engineering run")
    directory = run / "physical"
    plan, scene, identity = load_plan(directory / "plan.json")
    require_conversion_environment(json.loads(Path(plan["environment_report"]).read_text()))
    sources = json.loads((directory / "source_snapshot/manifest.json").read_text())
    if not sources or any(sha256(directory / "source_snapshot" / name) != digest for name, digest in sources.items()):
        raise ValueError("Engineering source snapshot changed")
    coordinator = json.loads((directory / "state.json").read_text())
    if (coordinator["state"] != "completed" or coordinator["identity"] != identity
            or any(row["exit_code"] != 0 or alive(row["owner"]) for row in coordinator["children"].values())):
        raise ValueError("Coordinator or child identities/exit status differ")
    before = json.loads((directory / "environment_before.json").read_text())
    after = json.loads((directory / "environment_after.json").read_text())
    if (before != after or before["asset_report_sha256"] != plan["file_sha256"][plan["asset_report"]]
            or before["environment_report_sha256"] != plan["file_sha256"][plan["environment_report"]]):
        raise ValueError("Pre/post simulator provenance differs from registered inputs")
    episode_result = json.loads((directory / "episode_result.json").read_text())
    if (episode_result["identity"] != identity or episode_result["outcome"] != coordinator["outcome"]
            or episode_result["formal_policy_result"] is not False):
        raise ValueError("Episode lifecycle completion differs")
    if plan["protocol_id"] != LEGACY_PROTOCOL:
        control = json.loads((directory / "episode_control_complete.json").read_text())
        if control["identity"] != identity or control["outcome"] != episode_result["outcome"]:
            raise ValueError("Physical control completion differs from checked lifecycle")
    key = stable_hash(identity)
    ledger = Path(plan["ledger_root"]) / key
    marker = json.loads((ledger / "COMPLETE.json").read_text())
    attempt = ledger / marker["attempt"]
    result = json.loads((attempt / "result.json").read_text())
    if (marker != {"episode_key": key, "attempt": attempt.name, "result_sha256": sha256(attempt / "result.json")}
            or result["episode_key"] != key or result["attempt"] != attempt.name
            or result["begin_sha256"] != sha256(attempt / "begin.json")):
        raise ValueError("Episode completion provenance differs")
    validate_outcome(result["outcome"], result["reason"], result["stats"], result["artifacts"])
    if artifact_inventory(attempt, {key: row["path"] for key, row in result["artifacts"].items()}) != result["artifacts"]:
        raise ValueError("Episode artifacts changed")
    rollout = json.loads((attempt / "rollout.json").read_text())
    if (rollout["identity"] != identity or rollout["stats"] != result["stats"]
            or rollout["outcome"] != result["outcome"] or rollout["reason"] != result["reason"]):
        raise ValueError("Rollout and ledger outcome differ")
    if result["outcome"] != episode_result["outcome"]:
        raise ValueError("Ledger policy result differs from completed episode")
    verify_initial(json.loads((Path(scene["probe"]) / "reset_0/initial.json").read_text()),
                   json.loads((attempt / "initial/initial.json").read_text()))
    actions = [json.loads(line) for line in (attempt / "actions.jsonl").read_text().splitlines()]
    requests = [json.loads(line)["result"] for line in (directory / "service/requests.jsonl").read_text().splitlines()]
    if actions != rollout["actions"] or result["stats"]["step_limit"] + 1 != plan["max_messages"]:
        raise ValueError("Action trace or original task limit differs")
    audit_actions(actions, requests, identity, attempt=int(attempt.name.split("_")[-1]), stats=result["stats"],
                  outcome=result["outcome"], reason=result["reason"])
    video = attempt / "video.mp4"
    stream = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_read_frames,codec_name", "-of", "json", str(video)], capture_output=True, text=True, check=True)
    if json.loads(stream.stdout)["streams"] != [result["stats"]["video"]["stream"]]:
        raise ValueError("Saved video geometry/frame count differs")
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", "scale=64:64", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-"], capture_output=True, check=True, timeout=120).stdout
    frames = np.frombuffer(decoded, dtype=np.uint8).reshape(-1, 64, 64, 3)
    if len(frames) != result["stats"]["video"]["frames"] or any(frame.std() <= 1 for frame in frames):
        raise ValueError("Video is blank, truncated or undecodable")
    service = json.loads((directory / "service/state.json").read_text())
    if service["state"] != "stopped" or service["messages"] != len(requests):
        raise ValueError("Service did not stop with complete message accounting")
    return {"goal_id": GOAL_ID, "passed": True, "run": str(run), "identity": identity,
            "outcome": result["outcome"], "stats": result["stats"], "complete_sha256": sha256(ledger / "COMPLETE.json"),
            "video_nonblank": True, "initial_exact": True, "service_actions_exact": True,
            "limits": "One step-zero non-final real physical episode only; not formal efficacy, interruption recovery or full task-set acceptance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing episode audit")
    try:
        report = audit(args.run)
    except Exception as error:
        atomic_json(args.output, {"goal_id": GOAL_ID, "passed": False, "error": f"{type(error).__name__}: {error}"})
        raise
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
