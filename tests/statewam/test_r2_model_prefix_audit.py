import json

import pytest

from scripts.statewam.r2 import audit_model_prefix as module
from scripts.statewam.r2.checkpoints import inventory
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256, stable_hash


@pytest.fixture
def runs(tmp_path):
    identity = {"world_size": 4, "updates": 4, "accumulation": 1, "global_batch": 64}
    paths = [tmp_path / name for name in ("reference", "prefix")]
    for run, end in zip(paths, (4, 2)):
        training = run / "training"
        snapshot = training / "source_snapshot"
        snapshot.mkdir(parents=True)
        (snapshot / "source.py").write_text("value = 1\n")
        atomic_json(snapshot / "manifest.json", {"source.py": sha256(snapshot / "source.py")})
        atomic_json(run / "state.json", {"state": "completed", "exit_code": 0})
        atomic_json(training / "identity.json", identity)
        position = {"optimizer_update": end, "next_occurrence": end * 64}
        checkpoint = training / "checkpoints" / f"update_{end:08d}"
        (checkpoint / "accelerate").mkdir(parents=True)
        atomic_json(checkpoint / "position.json", {"goal_id": GOAL_ID, "identity": identity, "position": position})
        for rank in range(4):
            atomic_json(checkpoint / f"rank_{rank}_ready.json", {
                "rank": rank, "position": position, "identity_sha256": stable_hash(identity)})
            (checkpoint / f"accelerate/random_states_{rank}.pkl").write_bytes(b"fixture, not loaded")
            atomic_json(training / f"summary_rank{rank}.json", {
                "position": position, "identity": identity, "stopped": False, "checkpoint": str(checkpoint)})
            rows = [{"position": {"optimizer_update": step, "next_occurrence": step * 64},
                     "samples": [{"rank": rank, "step": step}], "noise_hashes": [str(step)],
                     "metrics": {"loss": 1 / step}, "gradients": {"norm": step / 10}}
                    for step in range(1, end + 1)]
            (training / f"steps_rank{rank}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        atomic_json(checkpoint / "COMPLETE.json", {"identity_sha256": stable_hash(identity), "files": inventory(checkpoint)})
        atomic_json(checkpoint.parent / "latest-complete.json", {
            "checkpoint": checkpoint.name, "complete_sha256": sha256(checkpoint / "COMPLETE.json")})
    return paths


def test_exact_prefix_requires_all_four_ranks_and_complete_checkpoint(runs):
    report = module.audit(*runs)
    assert report["passed"]
    assert len(report["ranks"]) == 4
    assert all(row["prefix_microsteps"] == 2 and row["exact"] for row in report["ranks"])


@pytest.mark.parametrize("field", ["samples", "noise_hashes", "metrics", "gradients", "position"])
def test_prefix_rejects_each_trace_difference(runs, field):
    path = runs[1] / "training/steps_rank3.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = None
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="rank3.prefix"):
        module.audit(*runs)


@pytest.mark.parametrize("case", ["live", "failed", "snapshot", "identity", "summary", "missing_step", "checkpoint", "pointer"])
def test_prefix_rejects_unproven_evidence(runs, monkeypatch, case):
    if case == "live":
        monkeypatch.setattr(module, "alive", lambda record: True)
    elif case == "failed":
        atomic_json(runs[0] / "state.json", {"state": "completed", "exit_code": 1})
    elif case == "snapshot":
        (runs[1] / "training/source_snapshot/source.py").write_text("value = 2\n")
    elif case == "identity":
        path = runs[1] / "training/identity.json"
        data = json.loads(path.read_text())
        atomic_json(path, {**data, "global_batch": 16})
    elif case == "summary":
        path = runs[0] / "training/summary_rank3.json"
        data = json.loads(path.read_text())
        atomic_json(path, {**data, "stopped": True})
    elif case == "missing_step":
        (runs[0] / "training/steps_rank2.jsonl").write_text("")
    elif case == "checkpoint":
        (runs[1] / "training/checkpoints/update_00000002/accelerate/random_states_3.pkl").write_bytes(b"changed")
    elif case == "pointer":
        atomic_json(runs[1] / "training/checkpoints/latest-complete.json", {})
    with pytest.raises(ValueError):
        module.audit(*runs)
