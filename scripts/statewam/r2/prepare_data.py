"""Fetch pinned clean archives, audit episodes, and fit train-only statistics."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

import h5py
import numpy as np
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.common import (
    GOAL_ID,
    REPO_ID,
    REVISION,
    TASKS,
    VARIANT,
    atomic_json,
    sha256,
    stable_hash,
)


def fetch_manifest(endpoint, output):
    entries = []
    for task in TASKS:
        url = f"{endpoint}/api/datasets/{REPO_ID}/tree/{REVISION}/dataset/{task}"
        matches = []
        while url:
            response = requests.get(url, params={"limit": 1000}, timeout=60)
            response.raise_for_status()
            matches.extend(x for x in response.json() if x["path"] == f"dataset/{task}/{VARIANT}.zip")
            url = response.links.get("next", {}).get("url")
        if len(matches) != 1 or "lfs" not in matches[0]:
            raise ValueError(f"Expected one LFS clean archive for {task}")
        entries.append(dict(matches[0], task=task, sha256=matches[0]["lfs"]["oid"]))
        print(f"metadata {task}: {matches[0]['size']} bytes", flush=True)
    response = requests.get(f"{endpoint}/datasets/{REPO_ID}/raw/{REVISION}/README.md", timeout=60)
    response.raise_for_status()
    sections = response.text.split("---", 2)
    card = yaml.safe_load(sections[1]) if len(sections) == 3 else {}
    if card.get("license") != "mit":
        raise ValueError("Dataset license changed or could not be established")
    record = {"goal_id": GOAL_ID, "protocol_id": "r2-data-v1", "revision": REVISION,
              "repo_id": REPO_ID, "endpoint": endpoint, "dataset_card": response.text,
              "license": card["license"], "archives": entries,
              "archive_bytes_total": sum(x["size"] for x in entries)}
    atomic_json(output, record)
    return record


def download(entry, endpoint, root, reuse_root):
    target = root / "archives" / entry["task"] / f"{VARIANT}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != entry["size"] or sha256(target) != entry["sha256"]:
            raise ValueError(f"Existing archive is inconsistent: {target}")
        return target
    reusable = reuse_root / entry["path"] if reuse_root else None
    if reusable and reusable.is_file() and reusable.stat().st_size == entry["size"] and sha256(reusable) == entry["sha256"]:
        target.symlink_to(reusable.resolve())
        print(f"verified reuse {entry['task']}", flush=True)
        return target
    partial = target.with_suffix(".zip.partial")
    url = f"{endpoint}/datasets/{REPO_ID}/resolve/{REVISION}/{entry['path']}"
    # A failed attempt remains on disk, but is never accepted without the LFS hash.
    for attempt in range(2):
        try:
            with requests.get(url, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(8 * 1024**2):
                        stream.write(chunk)
            if partial.stat().st_size != entry["size"] or sha256(partial) != entry["sha256"]:
                raise ValueError(f"Downloaded hash mismatch: {entry['task']}")
            os.replace(partial, target)
            print(f"download complete {entry['task']}", flush=True)
            return target
        except (requests.RequestException, ValueError) as error:
            print(f"download attempt {attempt + 1} failed for {entry['task']}: {type(error).__name__}", flush=True)
            if attempt:
                raise


def extract_checked(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            path = PurePosixPath(entry.filename)
            mode = entry.external_attr >> 16
            if (path.is_absolute() or ".." in path.parts or "\\" in entry.filename
                    or not path.parts or path.parts[0] != VARIANT or stat.S_ISLNK(mode)):
                raise ValueError(f"Unsafe archive member: {entry.filename}")
            destination = target.joinpath(*path.parts)
            if not destination.resolve().is_relative_to(target.resolve()):
                raise ValueError("Archive destination escapes extraction root")
            if entry.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".partial")
            with source.open(entry) as src, partial.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024**2)
            os.replace(partial, destination)


def session_ids(value, prefix=""):
    found = []
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            name = f"{prefix}/{key}"
            if key.lower() in ("session_id", "recording_id", "trajectory_id", "source_episode_id") and child is not None:
                found.append([name, str(child)])
            found.extend(session_ids(child, name))
    return found


def audit_task(dataset_root, task):
    root = dataset_root / task / VARIANT
    paths = sorted((root / "data").glob("episode*.hdf5"))
    if {p.stem for p in paths} != {f"episode{i}" for i in range(50)}:
        raise ValueError(f"Expected exactly episodes 0..49: {task}")
    scenes = json.loads((root / "scene_info.json").read_text())
    rows = []
    for path in paths:
        episode_id = path.stem
        instruction = root / "instructions" / f"{episode_id}.json"
        prompt = json.loads(instruction.read_text())
        if not prompt.get("seen") or not prompt.get("unseen"):
            raise ValueError(f"Missing instruction pools: {instruction}")
        if not all(isinstance(x, str) and x.strip() for k in ("seen", "unseen") for x in prompt[k]):
            raise ValueError(f"Invalid instruction text: {instruction}")
        scene = scenes[f"episode_{int(episode_id.removeprefix('episode'))}"]
        with h5py.File(path, "r") as file:
            cameras = ("head_camera", "left_camera", "right_camera")
            lengths = [len(file[f"observation/{cam}/rgb"]) for cam in cameras]
            trajectory = hashlib.sha256()
            for key in ("endpose/left_endpose", "endpose/right_endpose",
                        "endpose/left_gripper", "endpose/right_gripper", "joint_action/vector"):
                array = file[key][()]
                if len(array) != lengths[0] or not np.isfinite(array).all():
                    raise ValueError(f"Invalid actions: {path}:{key}")
                trajectory.update(key.encode())
                trajectory.update(str(array.shape).encode())
                trajectory.update(array.astype("<f8").tobytes())
            if len(set(lengths)) != 1 or lengths[0] < 33:
                raise ValueError(f"Invalid camera lengths: {path}")
            attrs = {str(k): str(v) for k, v in file.attrs.items()}
            sessions = session_ids(scene) + session_ids(attrs)
        rows.append({"task": task, "episode_id": episode_id,
                     "path": str(path.relative_to(dataset_root)), "sha256": sha256(path),
                     "bytes": path.stat().st_size, "length": lengths[0],
                     "train_windows": lengths[0] - 1, "val_windows": lengths[0] - 32,
                     "trajectory_sha256": trajectory.hexdigest(), "session_ids": sessions,
                     "hdf5_attrs": attrs, "scene": scene,
                     "instruction_path": str(instruction.relative_to(dataset_root)),
                     "instruction_sha256": sha256(instruction)})
    return rows


def assign_split(rows, validation_count=5, seed=20260912):
    """Order episodes by stable keys; keep linked duplicate/session groups together."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tokens = {}
    for i, row in enumerate(rows):
        keys = [("trajectory", row["trajectory_sha256"])]
        keys += [("session", *s) for s in row.get("session_ids", [])]
        for key in keys:
            if key in tokens:
                parent[find(i)] = find(tokens[key])
            tokens[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)
    def rank(row):
        return stable_hash([REVISION, row["task"], row["episode_id"], seed])

    ordered = sorted(groups.values(), key=lambda group: min(rank(r) for r in group))
    count = 0
    for group in ordered:
        split = "val" if count < validation_count else "train"
        if split == "val":
            count += len(group)
        for row in group:
            row["split"] = split
            row["split_key"] = rank(row)
            row["duplicate_group"] = stable_hash(sorted(r["episode_id"] for r in group))
    if not any(row["split"] == "train" for row in rows):
        raise ValueError("Grouped split leaves no training episodes")
    return rows


def fit_train_stats(dataset_root, rows, output):
    from openwam.dataloader.utils.stats_computation.robotwin_stats_computation import (
        _ModeAccumulator,
        _pin_eef_rot6d_identity,
        _read_eef_actions,
        _read_joint_actions,
        atomic_save_stats_npy,
    )
    joint, eef = _ModeAccumulator(14), _ModeAccumulator(20)
    selected = [row for row in rows if row["split"] == "train"]
    if not selected:
        raise ValueError("No training episodes")
    for row in selected:
        path = dataset_root / row["path"]
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Episode changed since audit: {path}")
        with h5py.File(path, "r") as file:
            joint_actions, eef_actions = _read_joint_actions(file), _read_eef_actions(file)
        for array in (joint_actions, eef_actions):
            if array is None or not np.isfinite(array).all():
                raise ValueError(f"Missing or nonfinite action data: {path}")
        joint.update(joint_actions)
        eef.update(eef_actions)
    stats = {"joint": joint.finalize(), "eef": _pin_eef_rot6d_identity(eef.finalize()),
             "num_timesteps": eef.total_count}
    atomic_save_stats_npy(str(output), stats)
    return {"path": str(output), "sha256": sha256(output), "fit_split": "train",
            "episode_count": len(selected), "num_timesteps": eef.total_count,
            "files": [{"path": row["path"], "sha256": row["sha256"]} for row in selected],
            "rot6d": "identity pinning from repository stats helper", "mode": "min-max"}


def prepare(args):
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "source_manifest.json"
    source = fetch_manifest(args.endpoint, manifest_path)
    print(f"Total archive bytes: {source['archive_bytes_total']}", flush=True)
    if args.metadata_only:
        return
    dataset = args.output / "dataset"
    rows = []
    for entry in source["archives"]:
        archive = download(entry, args.endpoint, args.output, args.reuse_root)
        extract_checked(archive, dataset / entry["task"])
        rows.extend(assign_split(audit_task(dataset, entry["task"])))
        atomic_json(args.output / "audit_progress.json", {"episodes": rows})
        print(f"audited {entry['task']}", flush=True)
    duplicate_splits = {}
    for row in rows:
        key = row["trajectory_sha256"]
        if key in duplicate_splits and duplicate_splits[key] != row["split"]:
            raise ValueError("Cross-task duplicate crosses splits; revise grouping before fitting stats")
        duplicate_splits[key] = row["split"]
    manifest = {"schema_version": 1, "goal_id": GOAL_ID, "protocol_id": "r2-data-v1",
                "revision": REVISION, "variant": VARIANT, "dataset_root": str(dataset.resolve()),
                "split_seed": 20260912, "split_hash_encoding": "canonical JSON [revision, task, episode_id, seed]",
                "tasks": list(TASKS), "episodes": rows,
                "grouping": "exact action-trajectory duplicates and explicit session identifiers; not perceptual near-duplicate detection",
                "source_manifest_sha256": sha256(manifest_path)}
    atomic_json(args.output / "episode_manifest.json", manifest)
    stats = fit_train_stats(dataset, rows, args.output / "normalization_stats.npy")
    stats["episode_manifest_sha256"] = sha256(args.output / "episode_manifest.json")
    atomic_json(args.output / "statistics_provenance.json", stats)
    counts = {task: {split: {"episodes": sum(r["task"] == task and r["split"] == split for r in rows),
                           "windows": sum(r[f"{split}_windows"] for r in rows if r["task"] == task and r["split"] == split)}
                     for split in ("train", "val")} for task in TASKS}
    atomic_json(args.output / "data_summary.json", {"goal_id": GOAL_ID, "protocol_id": "r2-data-v1",
                "counts": counts, "started_at": started, "ended_at": time.time(),
                "episode_manifest_sha256": stats["episode_manifest_sha256"],
                "statistics_sha256": stats["sha256"], "status": "data_prepared_reader_validation_pending"})
    print(json.dumps(counts, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "data_summary.json").exists():
            raise FileExistsError("Completed data version is immutable; verify it or use a new version directory")
        prepare(args)


if __name__ == "__main__":
    main()
