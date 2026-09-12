"""Exercise the actual R2 reader on both explicit splits and real EEF samples."""

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from openwam.dataloader.registry import build_dataset
from scripts.statewam.r2.common import GOAL_ID, ROOT, atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / "assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Clean2Random/config.yaml"
    cfg = OmegaConf.load(source).dataloader
    cfg.dataset_dir = str((args.data / "dataset").resolve())
    cfg.episode_manifest = str((args.data / "episode_manifest.json").resolve())
    cfg.normalization_stats_path = str((args.data / "normalization_stats.npy").resolve())
    cfg.variant = "clean_50"
    cfg.window_stride = 1
    assert (cfg.num_frames, cfg.video_stride, cfg.height, cfg.width) == (33, 4, 384, 320)
    assert cfg.action_mode == "eef" and cfg.unify_action and cfg.multiview
    manifest = json.loads(Path(cfg.episode_manifest).read_text())
    records, physical = {}, {}
    for split in ("train", "val"):
        dataset = build_dataset(cfg, split=split)
        allowed = {str(Path(cfg.dataset_dir) / row["path"]) for row in manifest["episodes"] if row["split"] == split}
        actual = {path for ds in dataset._sub_datasets for path in ds._episode_files}
        if actual != allowed:
            raise ValueError("Actual reader selection does not equal explicit split")
        physical[split] = {Path(path).resolve() for path in actual}
        samples = []
        for task_dataset in dataset._sub_datasets:
            if not task_dataset.strict_data or task_dataset._normalizer is None:
                raise ValueError("Strict statistics loading is not active")
            if split == "val" and task_dataset._color_jitter is not None:
                raise ValueError("Validation augmentation is enabled")
            for ep_idx, _ in task_dataset._window_index:
                if task_dataset._episode_files[ep_idx] not in allowed:
                    raise ValueError("Window points outside split")
            for index in (0, len(task_dataset) // 2, len(task_dataset) - 1):
                sample = task_dataset[index]
                assert len(sample["video"]) == 9 and all(frame.size == (320, 384) for frame in sample["video"])
                assert sample["action"].shape == (32, 80) and sample["proprio"].shape == (1, 80)
                assert sample["action_mask"].shape == (32, 80) and sample["proprio_mask"].shape == (1, 80)
                reconstructed = task_dataset.denormalize_action(sample["action"].numpy())
                with h5py.File(sample["episode_path"], "r") as file:
                    raw = task_dataset._read_raw_actions(file, sample["start_frame"] + 1, sample["end_frame"])
                np.testing.assert_allclose(reconstructed[:len(raw)], raw, rtol=1e-5, atol=1e-5)
                if split == "val":
                    random.seed(912)
                    repeated = task_dataset[index]
                    assert sample["prompt"] == repeated["prompt"]
                    for a, b in zip(sample["video"], repeated["video"]):
                        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
                samples.append({"task": sample["task_name"], "path": sample["episode_path"],
                                "start": sample["start_frame"], "prompt": sample["prompt"],
                                "valid_action_steps": int(sample["action_mask"].any(dim=-1).sum()),
                                "roundtrip_max_error": float(np.abs(reconstructed[:len(raw)] - raw).max())})
        records[split] = {"episodes": len(actual), "windows": len(dataset), "samples": samples}
    if physical["train"] & physical["val"]:
        raise ValueError("Train and validation physical paths overlap")
    atomic_json(args.output, {"goal_id": GOAL_ID, "protocol_id": "r2-data-v1", "run_id": args.output.parent.name,
                             "status": "passed", "splits": records,
                             "resolved_dataloader": OmegaConf.to_container(cfg, resolve=True),
                             "episode_manifest_sha256": sha256(cfg.episode_manifest),
                             "statistics_sha256": sha256(cfg.normalization_stats_path),
                             "configuration_source_note": "Published configuration used only for compatible reader geometry/mapping; weights and normalization not reused"})
    print(f"Actual reader audit passed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
