import copy
import json
import subprocess

import pytest

from benchmarks.robotwin import r2_audit_pilot as module
from benchmarks.robotwin.r2_audit_pilot import audit_result, audit_trace, request_inventory
from benchmarks.robotwin.r2_episode_ledger import EpisodeLedger
from benchmarks.robotwin.r2_policy_protocol import reset_identity
from scripts.statewam.r2.common import atomic_json, sha256, stable_hash
from tests.statewam.test_r2_audit_episode import trace
from tests.statewam.test_r2_evaluation_plan import schedule_fixture


@pytest.mark.parametrize("reason,code", [("nonfinite_action", "policy_output_error"), ("model_error", "internal_error")])
def test_zero_action_policy_error_is_counted_failure_not_invalid(reason, code):
    identity = schedule_fixture()["waves"][0]["blocks"][0]["episodes"][0]["identity"]
    ack = {"type": "reset_ack", **reset_identity(identity, attempt=1), "next_action": 0, "next_chunk": 0,
           "cleared": {"executor": True, "prompt": True, "vace": True, "velocity": True, "temporary_state": True}}
    error = {"type": "error", "code": code, "message": "recorded error"}
    result = {"outcome": "failure", "reason": reason, "stats": {"policy_actions": 0}}
    rollout = {"evidence": {"code": code, "message": "recorded error"}}
    audit_trace([], [ack, error], identity, attempt=1, result=result, rollout=rollout)
    error["code"] = "protocol_error"
    with pytest.raises(ValueError, match="reason differs"):
        audit_trace([], [ack, error], identity, attempt=1, result=result, rollout=rollout)


def test_policy_error_requires_service_evidence():
    identity = schedule_fixture()["waves"][0]["blocks"][0]["episodes"][0]["identity"]
    with pytest.raises(ValueError, match="terminal service error"):
        audit_trace([], [], identity, attempt=1,
                    result={"outcome": "failure", "reason": "model_error", "stats": {}}, rollout={})


def test_real_ledger_action_conversion_and_video_decode(tmp_path, monkeypatch, trace):
    actions, requests, identity, stats = trace
    stats = {**stats, "wall_seconds": 1.0}
    probe = tmp_path / "probe"
    initial = {"metadata": {"step_lim": 33}, "rgb": {"head": "fixture-rgb"}}
    atomic_json(probe / "reset_0/initial.json", initial)
    episode = {"identity": identity, "episode_key": stable_hash(identity), "scene": {"probe": str(probe)}}
    ledger_root = tmp_path / "ledger"
    with EpisodeLedger(ledger_root, identity) as ledger:
        attempt = ledger.begin()
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=33",
            "-frames:v", "33", "-c:v", "libx264", "-threads", "1", str(attempt / "video.mp4")],
            check=True, timeout=30)
        stream = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_read_frames,codec_name", "-of", "json", str(attempt / "video.mp4")],
            check=True, text=True, capture_output=True, timeout=30)
        stats["video"] = {"sha256": sha256(attempt / "video.mp4"), "frames": 33,
                          "stream": json.loads(stream.stdout)["streams"][0]}
        atomic_json(attempt / "initial/initial.json", initial)
        (attempt / "actions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in actions))
        atomic_json(attempt / "rollout.json", {"identity": identity, "stats": stats,
                    "outcome": "failure", "reason": "action_limit", "actions": actions})
        ledger.finish(attempt, outcome="failure", reason="action_limit", stats=stats,
                      artifacts={"video": "video.mp4", "rollout": "rollout.json", "initial": "initial/initial.json"})
    # The fixture was published in this test process; lifecycle itself has separate subprocess tests.
    monkeypatch.setattr(module, "alive", lambda owner: False)
    evidence = {(episode["episode_key"], 1): requests}
    result = audit_result(ledger_root, episode, evidence)
    assert result["audited"] and result["outcome"] == "failure" and result["attempt"] == 1
    assert audit_result(ledger_root, episode, evidence) == result
    assert len(list(attempt.parent.glob("attempt_*"))) == 1
    with pytest.raises(ValueError, match="Missing service"):
        audit_result(ledger_root, episode, {})
    atomic_json(probe / "reset_0/initial.json", {**initial, "metadata": {"step_lim": 34}})
    with pytest.raises(ValueError, match="Initial scene differs"):
        audit_result(ledger_root, episode, evidence)


def inventory_fixture(root):
    schedule = schedule_fixture()
    schedule_path = root / "schedule.json"
    atomic_json(schedule_path, schedule)
    plan = {"source_sha256": {}, "environment": {"sha256": "environment"}, "assets": {"sha256": "assets"}}
    output = root / "evaluation"
    atomic_json(root / "state.json", {"state": "completed", "exit_code": 0, "gpu_uuids": ["sim", "a", "b"],
                "gpu_release_snapshot": [{"uuid": uuid, "processes": []} for uuid in ("sim", "a", "b")]})
    atomic_json(output / "execution.json", {"schedule": str(schedule_path), "schedule_sha256": sha256(schedule_path),
                                            "protocol_sha256": schedule["protocol_sha256"]})
    atomic_json(output / "source_snapshot/manifest.json", {})
    before = {"environment_report_sha256": "environment", "asset_report_sha256": "assets"}
    atomic_json(output / "environment_before.json", before)
    atomic_json(output / "environment_after.json", before)
    atomic_json(output / "state.json", {"state": "completed"})
    block = schedule["waves"][0]["blocks"][0]
    directory = output / block["block_id"]
    atomic_json(directory / "block.json", block)
    atomic_json(directory / "state.json", {"block_id": block["block_id"], "state": "completed", "children": {}})
    binding = {"group": "E0", "checkpoint_sha256": schedule["checkpoint_bindings"]["E0"]["weight_sha256"],
               "protocol_sha256": schedule["protocol_sha256"]}
    episode = block["episodes"][0]
    ack = {"type": "reset_ack", **reset_identity(episode["identity"], attempt=1), "next_action": 0, "next_chunk": 0,
           "cleared": {"executor": True, "prompt": True, "vace": True, "velocity": True, "temporary_state": True}}
    atomic_json(directory / "service/state.json", {"binding": binding, "state": "stopped", "messages": 1})
    (directory / "service/requests.jsonl").write_text(json.dumps({"message_index": 1, "result": ack}) + "\n")
    return schedule, plan, directory, episode


def change_json(path, **changes):
    atomic_json(path, {**json.loads(path.read_text()), **changes})


def test_service_inventory_binds_block_reset_and_terminal_provenance(tmp_path):
    schedule, plan, directory, episode = inventory_fixture(tmp_path)
    requests, evidence = request_inventory([tmp_path], schedule, plan)
    assert list(requests) == [(episode["episode_key"], 1)]
    assert evidence[0]["completed"]
    with pytest.raises(ValueError, match="duplicate service reset"):
        request_inventory([tmp_path, tmp_path], schedule, plan)


@pytest.mark.parametrize("case", ["release", "exit", "block", "binding", "messages", "outside", "order", "after", "live"])
def test_service_inventory_rejects_mismatched_evidence(tmp_path, monkeypatch, case):
    schedule, plan, directory, episode = inventory_fixture(tmp_path)
    if case == "release":
        change_json(tmp_path / "state.json", gpu_release_snapshot=[])
    elif case == "exit":
        change_json(tmp_path / "state.json", exit_code=1)
    elif case == "block":
        change_json(directory / "block.json", group="E3")
    elif case == "binding":
        change_json(directory / "service/state.json", binding={"group": "E3"})
    elif case == "messages":
        change_json(directory / "service/state.json", messages=2)
    elif case == "outside":
        row = copy.deepcopy(schedule["waves"][1]["blocks"][0]["episodes"][0])
        path = directory / "service/requests.jsonl"
        message = json.loads(path.read_text())
        message["result"].update(reset_identity(row["identity"], attempt=1))
        path.write_text(json.dumps(message) + "\n")
    elif case == "order":
        path = directory / "service/requests.jsonl"
        path.write_text(json.dumps({**json.loads(path.read_text()), "message_index": 2}) + "\n")
    elif case == "after":
        change_json(tmp_path / "evaluation/environment_after.json", asset_report_sha256="changed")
    elif case == "live":
        change_json(directory / "state.json", children={"service": {"owner": {"pid": 1}}})
        monkeypatch.setattr(module, "alive", lambda owner: bool(owner))
    with pytest.raises(ValueError):
        request_inventory([tmp_path], schedule, plan)
