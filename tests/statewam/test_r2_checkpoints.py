import json
from types import SimpleNamespace

import pytest

from scripts.statewam.r2.checkpoints import inventory, save_checkpoint, validate_checkpoint
from scripts.statewam.r2.common import atomic_json, sha256, stable_hash
from scripts.statewam.r2.exposure_sampler import ExposureBatchSampler


IDENTITY = {"world_size": 1, "updates": 4, "global_batch": 8, "data": "fixed", "zero_stage": 2}


def fake_accelerator():
    def save(path):
        from pathlib import Path
        path = Path(path)
        path.mkdir()
        for name in ("random_states_0.pkl", "rank0_optim_states.pt", "rank0_model_states.pt"):
            (path / name).write_bytes(b"test-only-not-a-real-checkpoint")
    return SimpleNamespace(is_main_process=True, num_processes=1, process_index=0,
                           sync_gradients=True, save_state=save, wait_for_everyone=lambda: None)


def test_atomic_checkpoint_and_identity(tmp_path):
    path = save_checkpoint(fake_accelerator(), tmp_path, identity=IDENTITY,
                           position={"optimizer_update": 2, "next_occurrence": 16})
    assert validate_checkpoint(path, identity=IDENTITY)["optimizer_update"] == 2
    latest = json.loads((tmp_path / "latest-complete.json").read_text())
    assert latest["complete_sha256"] == sha256(path / "COMPLETE.json")
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_checkpoint(path, identity=dict(IDENTITY, data="different"))
    with pytest.raises(RuntimeError, match="already exists"):
        save_checkpoint(fake_accelerator(), tmp_path, identity=IDENTITY,
                        position={"optimizer_update": 2, "next_occurrence": 16})


def test_reject_incomplete_and_corrupt_checkpoint(tmp_path):
    partial = tmp_path / "update_2.partial"
    partial.mkdir()
    with pytest.raises(ValueError, match="Incomplete"):
        validate_checkpoint(partial, identity=IDENTITY)
    path = save_checkpoint(fake_accelerator(), tmp_path, identity=IDENTITY,
                           position={"optimizer_update": 2, "next_occurrence": 16})
    (path / "accelerate/rank0_optim_states.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="inventory/hash"):
        validate_checkpoint(path, identity=IDENTITY)


def test_accumulation_boundary_rejected(tmp_path):
    accelerator = fake_accelerator()
    accelerator.sync_gradients = False
    with pytest.raises(ValueError, match="accumulation"):
        save_checkpoint(accelerator, tmp_path, identity=IDENTITY,
                        position={"optimizer_update": 2, "next_occurrence": 16})
    with pytest.raises(ValueError, match="exposure boundary"):
        save_checkpoint(fake_accelerator(), tmp_path, identity=IDENTITY,
                        position={"optimizer_update": 2, "next_occurrence": 15})


def test_sampler_covers_paired_exposures_and_resumes():
    rows = [list(ExposureBatchSampler(updates=4, micro_batch=2, accumulation=3, world_size=4, rank=r))
            for r in range(4)]
    assert sorted(x for rank in rows for batch in rank for x in batch) == list(range(96))
    for rank in range(4):
        resumed = list(ExposureBatchSampler(updates=4, micro_batch=2, accumulation=3,
                                           world_size=4, rank=rank, start_update=2))
        assert resumed == rows[rank][6:]


@pytest.mark.parametrize("missing", ["accelerate/random_states_0.pkl", "accelerate/rank0_optim_states.pt"])
def test_missing_shards_rejected_even_with_matching_inventory(tmp_path, missing):
    path = save_checkpoint(fake_accelerator(), tmp_path, identity=IDENTITY,
                           position={"optimizer_update": 2, "next_occurrence": 16})
    (path / missing).unlink()
    atomic_json(path / "COMPLETE.json", {"identity_sha256": stable_hash(IDENTITY), "files": inventory(path)})
    with pytest.raises(ValueError, match="Missing"):
        validate_checkpoint(path, identity=IDENTITY)


def test_interrupted_save_preserves_previous_complete(tmp_path):
    old = save_checkpoint(fake_accelerator(), tmp_path, identity=IDENTITY,
                          position={"optimizer_update": 1, "next_occurrence": 8})
    before = (tmp_path / "latest-complete.json").read_bytes()
    broken = fake_accelerator()
    def interrupted(_):
        raise RuntimeError("simulated write interruption")
    broken.save_state = interrupted
    with pytest.raises(RuntimeError, match="write interruption"):
        save_checkpoint(broken, tmp_path, identity=IDENTITY,
                        position={"optimizer_update": 2, "next_occurrence": 16})
    assert (tmp_path / "latest-complete.json").read_bytes() == before
    assert validate_checkpoint(old, identity=IDENTITY)["optimizer_update"] == 1
    with pytest.raises(ValueError, match="Incomplete"):
        validate_checkpoint(tmp_path / "update_00000002.partial", identity=IDENTITY)
