import copy
import json

import pytest

from scripts.statewam.r2 import audit_formal_training as module
from scripts.statewam.r2.audit_formal_training import audit_group, pairing_digest, validate_row
from scripts.statewam.r2.checkpoints import inventory
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256, stable_hash
from scripts.statewam.r2.exposure_sampler import ExposureBatchSampler


def row_fixture():
    return {"position": {"optimizer_update": 1, "next_occurrence": 64},
            "samples": [{"occurrence": index} for index in range(16)], "noise_hashes": ["noise"] * 16,
            "metrics": {"loss": 0.1}, "gradients": {"action.weight": [1.0, 1]},
            "phase_timings": {name: {"calls": 1} for name in ("forward", "engine_backward", "engine_step")}}


def test_formal_row_requires_exact_exposure_cursor_and_finite_gradients():
    row = row_fixture()
    validate_row(row, update=1, occurrences=list(range(16)), global_batch=64)
    for field, value in (("position", {"optimizer_update": 2, "next_occurrence": 128}),
                         ("noise_hashes", []), ("gradients", {}), ("metrics", {"loss": float("nan")}),
                         ("metrics", {}), ("gradients", {"video_backbone.weight": [1.0, 1]}),
                         ("phase_timings", {})):
        changed = {**row, field: value}
        with pytest.raises(ValueError):
            validate_row(changed, update=1, occurrences=list(range(16)), global_batch=64)


def test_pairing_digest_includes_augmentation_noise_and_sequence_not_loss():
    row = row_fixture()
    digest = pairing_digest([row])
    other = copy.deepcopy(row)
    other["metrics"]["loss"] = 123.0
    assert pairing_digest([other]) == digest
    other["samples"][0]["augmentation"] = "changed"
    assert pairing_digest([other]) != digest
    other = copy.deepcopy(row)
    other["noise_hashes"][0] = "other"
    assert pairing_digest([other]) != digest
    assert pairing_digest([row, other]) != pairing_digest([other, row])


def chain_plan():
    return {"protocol_id": "test-only", "source_sha256": {}, "matrix": {"sha256": "matrix"},
            "training": {"updates": 4, "seed": 42, "micro_batch": 2, "accumulation": 1,
                         "global_batch": 8, "world_size": 4, "workers": 0,
                         "exposures": 32, "save_every": 2}}


def checkpoint_fixture(root, identity, update):
    # Tiny non-loadable payloads exercise the actual inventory/identity validator.
    path = root / f"update_{update:08d}"
    path.mkdir(parents=True)
    position = {"optimizer_update": update, "next_occurrence": update * identity["global_batch"]}
    atomic_json(path / "position.json", {"goal_id": GOAL_ID, "identity": identity, "position": position})
    (path / "accelerate").mkdir()
    for rank in range(4):
        atomic_json(path / f"rank_{rank}_ready.json", {
            "rank": rank, "position": position, "identity_sha256": stable_hash(identity)})
        (path / f"accelerate/random_states_{rank}.pkl").write_bytes(b"fixture RNG")
        (path / f"accelerate/rank{rank}_optim_states.pt").write_bytes(b"fixture optimizer")
    (path / "accelerate/rank0_model_states.pt").write_bytes(b"fixture model")
    atomic_json(path / "COMPLETE.json", {"identity_sha256": stable_hash(identity), "files": inventory(path)})
    atomic_json(root / "latest-complete.json", {
        "checkpoint": path.name, "complete_sha256": sha256(path / "COMPLETE.json")})
    return path


def invocation_fixture(run, plan, *, group="E0", start=0, end=4, resume=None,
                       state="completed", saves=(2, 4), loss=0.1):
    training = run / "training"
    training.mkdir(parents=True)
    recipe = plan["training"]
    identity = {key: recipe[key] for key in (
        "updates", "seed", "micro_batch", "accumulation", "global_batch", "world_size", "workers")}
    identity.update(purpose="training", group=group, zero_stage=2, protocol_id=plan["protocol_id"],
                    matrix_sha256=plan["matrix"]["sha256"],
                    plan_identity={"plan": plan, "protocol_sha256": "protocol"})
    atomic_json(run / "state.json", {"state": state, "exit_code": 0 if state == "completed" else 1})
    atomic_json(training / "identity.json", identity)
    atomic_json(training / "source_snapshot/manifest.json", plan["source_sha256"])
    for rank in range(4):
        atomic_json(training / f"invocation_rank{rank}.json", {
            "start_update": start, "end_update": recipe["updates"], "resume": str(resume) if resume else None,
            "purpose": "training"})
        sampler = iter(ExposureBatchSampler(updates=recipe["updates"], micro_batch=recipe["micro_batch"],
            accumulation=1, world_size=4, rank=rank, start_update=start))
        rows = []
        for update in range(start + 1, end + 1):
            row = row_fixture()
            row.update(position={"optimizer_update": update, "next_occurrence": update * recipe["global_batch"]},
                       samples=[{"occurrence": value, "task_index": value % 10} for value in next(sampler)],
                       noise_hashes=["paired-noise"] * recipe["micro_batch"], metrics={"loss": loss})
            rows.append(row)
        (training / f"steps_rank{rank}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        atomic_json(training / f"summary_rank{rank}.json", {
            "identity": identity, "position": {"optimizer_update": end, "next_occurrence": end * recipe["global_batch"]},
            "stopped": state == "stopped"})
    for update in saves:
        checkpoint_fixture(training / "checkpoints", identity, update)
    atomic_json(run / "performance_summary.json", {"wall_seconds": 10, "allocated_gpu_hours": 4 / 360})
    return run


def change_json(path, **changes):
    atomic_json(path, {**json.loads(path.read_text()), **changes})


def test_whole_chain_uses_real_checkpoint_integrity_validation(tmp_path):
    plan = chain_plan()
    run = invocation_fixture(tmp_path / "single", plan)
    report = audit_group([run], plan, "protocol", "E0")
    assert report["logical_exposures"] == report["actual_recorded_exposures"] == 32
    assert report["saved_updates"] == [2, 4]
    assert report["loss_windows"][0]["mean_training_losses"]["loss"] == pytest.approx(0.1)
    shard = run / "training/checkpoints/update_00000004/accelerate/rank2_optim_states.pt"
    shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="inventory/hash"):
        audit_group([run], plan, "protocol", "E0")


def test_resume_discards_unsaved_suffix_but_keeps_actual_cost(tmp_path):
    plan = chain_plan()
    first = invocation_fixture(tmp_path / "first", plan, end=3, state="failed", saves=(2,))
    resume = first / "training/checkpoints/update_00000002"
    last = invocation_fixture(tmp_path / "last", plan, start=2, resume=resume, saves=(4,))
    report = audit_group([first, last], plan, "protocol", "E0")
    assert report["actual_recorded_exposures"] == 40
    assert report["recorded_recomputed_exposures"] == 8
    continuous = invocation_fixture(tmp_path / "continuous", plan)
    assert report["paired_trace_sha256_by_rank"] == audit_group(
        [continuous], plan, "protocol", "E0")["paired_trace_sha256_by_rank"]
    change_json(last / "training/invocation_rank3.json", resume=None)
    with pytest.raises(ValueError, match="Ranks resumed"):
        audit_group([first, last], plan, "protocol", "E0")


@pytest.mark.parametrize("corruption", ["identity", "budget", "missing_step", "latest", "missing_save",
                                       "live", "final_failed", "source", "checkpoint_ahead", "duplicate"])
def test_full_chain_rejects_incomplete_or_inconsistent_evidence(tmp_path, monkeypatch, corruption):
    plan = chain_plan()
    run = invocation_fixture(tmp_path / "run", plan, saves=(4,) if corruption == "missing_save" else (2, 4))
    training = run / "training"
    if corruption == "identity":
        change_json(training / "identity.json", seed=43)
    elif corruption == "budget":
        change_json(training / "invocation_rank0.json", end_update=5)
    elif corruption == "missing_step":
        path = training / "steps_rank0.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[1:]) + "\n")
    elif corruption == "latest":
        change_json(training / "checkpoints/latest-complete.json", checkpoint="update_00000002")
    elif corruption == "live":
        monkeypatch.setattr(module, "alive", lambda row: bool(row))
        change_json(run / "state.json", child={"pid": 123, "created_at": 1})
    elif corruption == "final_failed":
        change_json(run / "state.json", state="failed", exit_code=1)
    elif corruption == "source":
        atomic_json(training / "source_snapshot/manifest.json", {"scripts/new.py": "changed"})
    elif corruption == "checkpoint_ahead":
        first = invocation_fixture(tmp_path / "first", plan, end=1, state="failed", saves=(2,))
        run = invocation_fixture(tmp_path / "last", plan, start=2, saves=(4,),
                                 resume=first / "training/checkpoints/update_00000002")
        with pytest.raises(ValueError, match="outside the recorded"):
            audit_group([first, run], plan, "protocol", "E0")
        return
    with pytest.raises(ValueError):
        audit_group([run, run] if corruption == "duplicate" else [run], plan, "protocol", "E0")


def test_four_group_audit_checks_pairing_and_zero_gate_losses(tmp_path, monkeypatch):
    plan = chain_plan()
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("test protocol")
    monkeypatch.setattr(module, "load_protocol", lambda path: plan)
    monkeypatch.setattr(module, "sha256", lambda path: "protocol" if path == protocol else sha256(path))
    chains = {group: [invocation_fixture(tmp_path / group, plan, group=group)] for group in module.GROUPS}
    assert module.audit(protocol, chains)["all_samples_noise_paired"]
    path = chains["E3"][0] / "training/steps_rank0.jsonl"
    original = path.read_text()
    rows = [json.loads(line) for line in original.splitlines()]
    rows[0]["noise_hashes"][0] = "unpaired"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="not paired"):
        module.audit(protocol, chains)
    rows = [json.loads(line) for line in original.splitlines()]
    rows[0]["metrics"]["loss"] = 0.2
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="first losses"):
        module.audit(protocol, chains)
