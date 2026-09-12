import copy
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from openwam.dataloader.utils.episode_manifest import file_sha256, load_episode_selection
from scripts.statewam.r2.common import atomic_json, stable_hash
from scripts.statewam.r2.prepare_data import assign_split, extract_checked, fit_train_stats, session_ids
from tests.test_robotwin_dataloader import _create_mock_episode


def rows(count=50):
    return [{"task": "adjust_bottle", "episode_id": f"episode{i}",
             "trajectory_sha256": stable_hash(i), "session_ids": []} for i in range(count)]


def test_split_deterministic_and_disjoint():
    original = rows()
    first = assign_split(copy.deepcopy(original))
    second = assign_split(list(reversed(copy.deepcopy(original))))
    assert {r["episode_id"]: r["split"] for r in first} == {r["episode_id"]: r["split"] for r in second}
    assert sum(r["split"] == "val" for r in first) == 5
    assert sum(r["split"] == "train" for r in first) == 45


def test_duplicate_and_session_groups_not_split():
    original = rows()
    original[0]["trajectory_sha256"] = original[1]["trajectory_sha256"]
    original[1]["session_ids"] = [["/session_id", "shared"]]
    original[2]["session_ids"] = [["/session_id", "shared"]]
    result = assign_split(original)
    assert len({r["split"] for r in result[:3]}) == 1
    assert len({r["duplicate_group"] for r in result[:3]}) == 1


def test_session_ids_recognized_without_asset_grouping():
    assert session_ids({"info": {"session_id": "x", "{A}": "bottle"}}) == [["/info/session_id", "x"]]


@pytest.mark.parametrize("member", ["../outside", "/absolute", "other/data/test", "aloha-agilex_clean_50/../../outside"])
def test_extract_rejects_unsafe_members(tmp_path, member):
    archive = tmp_path / "test.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(member, "test")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_checked(archive, tmp_path / "data")


def test_atomic_json_rejects_nonfinite(tmp_path):
    path = tmp_path / "result.json"
    atomic_json(path, {"x": 1})
    with pytest.raises(ValueError):
        atomic_json(path, {"x": np.nan})
    assert json.loads(path.read_text()) == {"x": 1}


@pytest.fixture
def isolated_dataset(tmp_path):
    root = tmp_path / "dataset"
    variant = "aloha-agilex_clean_50"
    records = []
    for i in range(3):
        path = root / "adjust_bottle" / variant / "data" / f"episode{i}.hdf5"
        path.parent.mkdir(parents=True, exist_ok=True)
        _create_mock_episode(path, T=40, seed=i)
        instruction = path.parent.parent / "instructions" / f"episode{i}.json"
        atomic_json(instruction, {"seen": [f"episode {i}"], "unseen": [f"other {i}"]})
        if i == 2:
            with h5py.File(path, "r+") as file:
                file["endpose/left_endpose"][:, :3] = 1000
        records.append({"task": "adjust_bottle", "episode_id": f"episode{i}",
                        "path": str(path.relative_to(root)), "sha256": file_sha256(path),
                        "split": "val" if i == 2 else "train", "trajectory_sha256": str(i),
                        "duplicate_group": str(i), "instruction_path": str(instruction.relative_to(root)),
                        "instruction_sha256": file_sha256(instruction),
                        "scene": {"info": {"active_arm": "right" if i == 2 else "both"}}})
    atomic_json(path.parent.parent / "scene_info.json", {f"episode_{i}": r["scene"] for i, r in enumerate(records)})
    manifest_path = tmp_path / "episode_manifest.json"
    manifest = {"schema_version": 1, "dataset_root": str(root), "variant": variant,
                "tasks": ["adjust_bottle"], "episodes": records}
    atomic_json(manifest_path, manifest)
    stats = tmp_path / "normalization_stats.npy"
    provenance = fit_train_stats(root, records, stats)
    provenance["episode_manifest_sha256"] = file_sha256(manifest_path)
    atomic_json(tmp_path / "statistics_provenance.json", provenance)
    return root, manifest_path, stats


def test_stats_excludes_validation_extremes(isolated_dataset):
    _, _, path = isolated_dataset
    stats = np.load(path, allow_pickle=True).item()
    assert np.max(stats["eef"]["max"][:3]) < 2
    assert stats["num_timesteps"] == 80
    np.testing.assert_array_equal(stats["eef"]["min"][3:9], -1)
    np.testing.assert_array_equal(stats["eef"]["max"][3:9], 1)


def test_reader_enforces_physical_split_and_scene_identity(isolated_dataset):
    from openwam.dataloader.robotwin import MultiTaskRoboTwinDataset

    root, manifest, stats = isolated_dataset
    kwargs = dict(dataset_dir=str(root), embodiment="aloha-agilex", variant="clean_50",
                  episode_manifest=str(manifest), normalization_stats_path=str(stats),
                  action_mode="eef", multiview=False, num_frames=33, height=32, width=32)
    train = MultiTaskRoboTwinDataset(**kwargs, split="train")
    val = MultiTaskRoboTwinDataset(**kwargs, split="val")
    train_paths = {str(Path(p).resolve()) for ds in train._sub_datasets for p in ds._episode_files}
    val_paths = {str(Path(p).resolve()) for ds in val._sub_datasets for p in ds._episode_files}
    assert not train_paths & val_paths
    assert len(train_paths) == 2 and len(val_paths) == 1
    assert train[0]["episode_path"] in train_paths
    assert val[0]["active_arm"] == "right"
    assert val[0]["episode_path"] in val_paths


@pytest.mark.parametrize("mutation", ["duplicate", "group", "file", "stats", "missing_stats", "instruction", "scene"])
def test_manifest_rejects_leakage_or_changed_inputs(isolated_dataset, mutation):
    root, manifest, stats = isolated_dataset
    payload = json.loads(manifest.read_text())
    if mutation == "duplicate":
        payload["episodes"].append(copy.deepcopy(payload["episodes"][0]))
    elif mutation == "group":
        payload["episodes"][-1]["duplicate_group"] = payload["episodes"][0]["duplicate_group"]
    elif mutation in ("file", "instruction"):
        key = "sha256" if mutation == "file" else "instruction_sha256"
        payload["episodes"][0][key] = "bad"
    elif mutation == "stats":
        stats.write_bytes(b"invalid")
    elif mutation == "scene":
        payload["episodes"][0]["scene"] = {}
    else:
        stats = stats.with_name("missing.npy")
    atomic_json(manifest, payload)
    with pytest.raises((ValueError, FileNotFoundError)):
        load_episode_selection(manifest, root, "aloha-agilex", "clean_50", "train", stats)


def test_allowlist_missing_file_and_loading_error_do_not_fallback(isolated_dataset, monkeypatch):
    from openwam.dataloader.robotwin import RoboTwinDataset

    root, _, stats = isolated_dataset
    data_root = root / "adjust_bottle/aloha-agilex_clean_50/data"
    kwargs = dict(data_root=str(data_root), normalization_stats_path=str(stats), height=32, width=32,
                  multiview=False, strict_data=True, action_mode="eef")
    with pytest.raises(FileNotFoundError):
        RoboTwinDataset(**kwargs, episode_allowlist=["episode999.hdf5"])
    dataset = RoboTwinDataset(**kwargs, episode_allowlist=["episode0.hdf5"])
    calls = []

    def fail(ep_idx, start):
        calls.append(ep_idx)
        raise OSError("read failure")

    monkeypatch.setattr(dataset, "_build_sample", fail)
    with pytest.raises(OSError, match="read failure"):
        dataset[0]
    assert calls == [0]
