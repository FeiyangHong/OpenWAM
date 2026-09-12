"""Validate explicit episode isolation and train-only normalization provenance."""

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_episode_selection(manifest_path, dataset_dir, embodiment, variant, split, stats_path):
    if split not in ("train", "val"):
        raise ValueError("Explicit episode manifests require split=train or val")
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    root = Path(dataset_dir).resolve()
    if manifest["schema_version"] != 1 or Path(manifest["dataset_root"]).resolve() != root:
        raise ValueError("Episode manifest schema or dataset root mismatch")
    if manifest["variant"] != f"{embodiment}_{variant}":
        raise ValueError("Episode manifest variant mismatch")
    tasks = manifest["tasks"]
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Empty or duplicate manifest tasks")
    selections = {task: [] for task in tasks}
    physical, ids, trajectory_splits, group_splits = set(), set(), {}, {}
    scenes = {}
    train_files = []
    for row in manifest["episodes"]:
        task, episode_id, row_split = row["task"], row["episode_id"], row["split"]
        if task not in selections or row_split not in ("train", "val"):
            raise ValueError("Unexpected task or split in manifest")
        path = root / row["path"]
        expected = root / task / manifest["variant"] / "data" / f"{episode_id}.hdf5"
        if path != expected or not path.resolve().is_relative_to(root):
            raise ValueError("Episode path does not match declared identity")
        inode = (path.stat().st_dev, path.stat().st_ino)
        if inode in physical or (task, episode_id) in ids:
            raise ValueError("Duplicate physical episode or identity in manifest")
        physical.add(inode)
        ids.add((task, episode_id))
        for mapping, key in ((trajectory_splits, row["trajectory_sha256"]),
                             (group_splits, (task, row["duplicate_group"]))):
            if key in mapping and mapping[key] != row_split:
                raise ValueError("Duplicate group crosses splits")
            mapping[key] = row_split
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"Episode checksum mismatch: {path}")
        instruction = root / row["instruction_path"]
        expected_instruction = expected.parent.parent / "instructions" / f"{episode_id}.json"
        if instruction != expected_instruction or not instruction.resolve().is_relative_to(root):
            raise ValueError("Instruction path mismatch")
        if file_sha256(instruction) != row["instruction_sha256"]:
            raise ValueError("Instruction checksum mismatch")
        if task not in scenes:
            scenes[task] = json.loads((expected.parent.parent / "scene_info.json").read_text())
        scene_key = f"episode_{int(episode_id.removeprefix('episode'))}"
        if scenes[task].get(scene_key) != row["scene"]:
            raise ValueError("Scene metadata changed since episode audit")
        if row_split == split:
            selections[task].append(path.name)
        if row_split == "train":
            train_files.append({"path": row["path"], "sha256": row["sha256"]})
    if any(not selected for selected in selections.values()):
        raise ValueError("Manifest selection leaves an empty task")
    if not stats_path or not Path(stats_path).is_file():
        raise FileNotFoundError("Explicit episode manifests require existing train-only statistics")
    provenance = json.loads((manifest_path.parent / "statistics_provenance.json").read_text())
    if (provenance["fit_split"] != "train"
            or provenance["episode_manifest_sha256"] != file_sha256(manifest_path)
            or provenance["sha256"] != file_sha256(stats_path)
            or provenance["files"] != train_files):
        raise ValueError("Statistics provenance does not match this manifest's training episodes")
    return selections
