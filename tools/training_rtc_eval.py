"""Robot-free checkpoint and multi-chunk evaluation using absolute EEF targets.

Imports model inference and the controller's pure preparation function only.
No robot SDK object, transport, publisher, or actuator interface is created.
Rollouts replay recorded observations: they are not a physical closed-loop simulator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cosine(u, v):
    den = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.dot(u, v) / den) if den > 1e-12 else None


def seam_metrics(actions, prefix, gt, state, d=10):
    result = {}
    for name, k in (("left", 0), ("right", 29)):
        end = prefix[-1, k:k + 3]
        incoming = end - prefix[-2, k:k + 3]
        outgoing = actions[d, k:k + 3] - end
        gt_in = gt[d - 1, k:k + 3] - gt[d - 2, k:k + 3]
        gt_out = gt[d, k:k + 3] - gt[d - 1, k:k + 3]
        internal = np.diff(actions[d:d + 16, k:k + 3], axis=0)
        result[name] = dict(
            before_mm=float(np.linalg.norm(incoming) * 1000),
            after_mm=float(np.linalg.norm(outgoing) * 1000),
            cosine=cosine(incoming, outgoing),
            gt_cosine=cosine(gt_in, gt_out),
            gt_before_mm=float(np.linalg.norm(gt_in) * 1000),
            gt_after_mm=float(np.linalg.norm(gt_out) * 1000),
            toward_state_cosine=cosine(state[k:k + 3] - end, outgoing),
            seam_velocity_change_mm_per_tick=float(np.linalg.norm(outgoing - incoming) * 1000),
            internal_step_median_mm=float(np.median(np.linalg.norm(internal, axis=1)) * 1000),
            internal_velocity_change_median_mm_per_tick=float(
                np.median(np.linalg.norm(np.diff(internal, axis=0), axis=1)) * 1000),
            first_error_mm=float(np.linalg.norm(actions[d, k:k + 3] - gt[d, k:k + 3]) * 1000),
            suffix_error_mm=float(np.linalg.norm(actions[d:d + 16, k:k + 3] - gt[d:d + 16, k:k + 3], axis=1).mean() * 1000),
            prefix_error_mm=float(np.linalg.norm(prefix[:, k:k + 3] - gt[:d, k:k + 3], axis=1).mean() * 1000),
            state_to_first_mm=float(np.linalg.norm(state[k:k + 3] - prefix[0, k:k + 3]) * 1000),
            state_to_end_mm=float(np.linalg.norm(state[k:k + 3] - end) * 1000),
            returned_prefix_xyz_error_mm=float(np.linalg.norm(actions[:d, k:k + 3] - prefix[:, k:k + 3], axis=1).max() * 1000),
        )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[9000, 6000, 7000, 8000])
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--client-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rollout-requests", type=int, default=6)
    ap.add_argument("--rollout-per-group", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "results.jsonl").exists():
        raise RuntimeError("Refusing to append to an existing evaluation")
    sys.path[:0] = [str(args.repo), str(args.client_root)]
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.server import merge_deploy_cfg
    from openwam.deploy.engine import JointInferenceEngine
    from openwam.dataloader.wuji_real_task import WujiRealTaskDataset
    from openwam_robot_client.rtc_controller import RTCController
    from openwam_robot_client.geometry import eef9_column_to_pose7

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(4)
    cfg = OmegaConf.load(args.checkpoint_dir / "config.yaml")
    assert cfg.dataloader.action_mode == "eef" and cfg.dataloader.rot6d_convention == "column"
    assert cfg.training.rtc.enabled
    dataset = WujiRealTaskDataset.from_config(cfg.dataloader)
    selected = json.loads(args.selection.read_text())
    episode_cache = {}

    def frame_for(item):
        ep = item["episode"]
        if ep not in episode_cache:
            p = Path(cfg.dataloader.dataset_dir) / f"data/chunk-000/episode_{ep:06d}.parquet"
            frame = pd.read_parquet(p, columns=["action", "observation.state"])
            episode_cache[ep] = (np.stack(frame["action"]).astype(np.float32),
                                 np.stack(frame["observation.state"]).astype(np.float32))
        return episode_cache[ep]

    prepared = {}

    def prepare(item, offset):
        key = (item["episode"], offset)
        if key in prepared:
            return prepared[key]
        actions, states = frame_for(item)
        if offset + 32 > len(actions):
            raise ValueError("Evaluation window crosses episode end")
        index = int(item["dataset_index"]) + offset - int(item["offset"])
        ep_local = int(np.searchsorted(dataset._cum_n_starts, index, side="right") - 1)
        if ep_local != int(item["ep_local"]):
            raise ValueError("Dataset index would enter a different episode")
        sample = dataset[index]
        state = states[offset].copy()
        prepared[key] = dict(image=sample["first_frame_image"][0], prompt=sample["prompt"],
                             state=state, normalized_action=sample["action"].numpy(),
                             normalized_state=sample["proprio"].numpy().reshape(-1),
                             source_actions=actions[offset:offset + 32].copy())
        return prepared[key]

    for item in selected:
        prepare(item, item["offset"])
    trajectories = []
    for group in ("slow", "medium", "fast"):
        pool = [dict(x, case=i) for i, x in enumerate(selected) if x["group"] == group]
        chosen = []
        for item in pool:
            raw, _ = frame_for(item)
            last = item["offset"] + 6 + 16 * (args.rollout_requests - 1)
            if last + 32 > len(raw):
                continue
            try:
                for j in range(args.rollout_requests):
                    prepare(item, item["offset"] + 6 + 16 * j)
            except (IndexError, ValueError):
                continue
            chosen.append(item)
            if len(chosen) == args.rollout_per_group:
                break
        if len(chosen) != args.rollout_per_group:
            raise RuntimeError(f"Insufficient complete rollout trajectories for {group}")
        trajectories.extend(chosen)
    dump(args.out / "selection.json", selected)
    dump(args.out / "trajectories.json", trajectories)
    dump(args.out / "protocol.json", dict(
        checkpoint_steps=args.steps, clean_windows=len(selected), seeds=args.seeds,
        rollout_trajectories=len(trajectories), rollout_requests=args.rollout_requests,
        rollout_seed=42, bootstrap_steps=16, first_request_offset=6, prefix_steps=10,
        new_steps=16, horizon=32, compile=False, cache=False, denoise_steps=10,
        observation_mode="Recorded image and state at each request, independent of generated actions",
        projection="Exact RTCController.prepare_from; 30mm XYZ, 10deg rotation, 0.1rad hand step, +/-1.57rad hand range, 2m workspace",
        exclusions="No SDK filter, dynamics, hardware tracking guards, left hold, transport, or robot execution",
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), torch=torch.__version__,
        config_sha256=sha256(args.checkpoint_dir / "config.yaml"),
        stats_sha256=sha256(args.checkpoint_dir / "normalization_stats.npy"),
    ))
    print("PREPARED", len(prepared), "observation windows", flush=True)
    if args.prepare_only:
        dump(args.out / "prepare_complete.json", dict(windows=len(prepared)))
        return

    deploy = OmegaConf.load(args.repo / "configs/deploy.yaml")
    for k, v in {"optimization.compile.enabled": False, "optimization.dit_cache.enabled": False,
                 "optimization.save_video": False, "optimization.decode_video": False,
                 "inference.rtc.enabled": True, "inference.inference_mode": "sync",
                 "inference.denoise_steps": 10, "inference.inference_horizon": None,
                 "inference.inference_delay_steps": None}.items():
        OmegaConf.update(deploy, k, v)
    train, architecture = load_from_checkpoint_dir(str(args.checkpoint_dir), device="cuda:0",
                                                  ckpt_name=f"checkpoint_step_{args.steps[0]}.safetensors")
    engine = JointInferenceEngine(merge_deploy_cfg(train, deploy), architecture=architecture)
    for obs in prepared.values():
        assert np.allclose(architecture.normalizer.normalize(obs["state"]), obs["normalized_state"], atol=2e-6)
        assert np.allclose(np.clip(architecture.normalizer.normalize_action(obs["source_actions"]), -1, 1),
                           obs["normalized_action"], atol=2e-6)
        obs["gt"] = architecture.normalizer.unnormalize(obs["normalized_action"]).astype(np.float32)
    count = 0
    started = time.monotonic()

    def infer(obs, prefix, seed, state=None):
        nonlocal count
        t = time.monotonic()
        with torch.inference_mode():
            result = engine.generate(dict(first_frame_image=[obs["image"]], prompt=obs["prompt"],
                                          proprio=obs["state"] if state is None else state,
                                          rtc_prefix=prefix, seed=seed))
        actions = np.asarray(result["actions"], dtype=np.float32)
        assert actions.shape == (32, 58) and np.isfinite(actions).all()
        count += 1
        return actions, time.monotonic() - t

    def record(step, mode, item, obs, actions, prefix, seed, seconds, **extra):
        state = np.asarray(extra.pop("metric_state", obs["state"]), dtype=np.float32)
        row = dict(checkpoint=step, mode=mode, case=item["case"], episode=item["episode"],
                   group=item["group"], seed=seed, seconds=seconds, metrics=seam_metrics(actions, prefix, obs["gt"], state),
                   actions=actions.tolist(), prefix=prefix.tolist(), state=state.tolist(),
                   gt=obs["gt"].tolist(), **extra)
        with (args.out / "results.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        m = row["metrics"]["right"]
        print("RESULT", count, step, mode, item["case"], extra.get("request", "-"), seed,
              "cos", m["cosine"], "first_mm", m["first_error_mm"], flush=True)

    def project(block, anchor, origin):
        # Use the exact existing pure controller method with data-only attributes.
        holder = SimpleNamespace(
            left_init_pose=None, left_init_hand=None,
            max_eef_step=0.03, max_rotation_step_rad=float(np.deg2rad(10)),
            workspace_min=(origin[:3] - 2, origin[29:32] - 2),
            workspace_max=(origin[:3] + 2, origin[29:32] + 2),
            hands=SimpleNamespace(max_step=0.1))
        poses = (eef9_column_to_pose7(anchor[:9]), eef9_column_to_pose7(anchor[29:38]))
        hands = (anchor[9:29], anchor[38:58])
        result = RTCController.prepare_from(holder, block, poses, hands)
        applied = np.stack([x.raw for x in result])
        delta = applied - block
        stats = dict(right_xyz_max_mm=float(np.linalg.norm(delta[:, 29:32], axis=1).max() * 1000),
                     left_xyz_max_mm=float(np.linalg.norm(delta[:, :3], axis=1).max() * 1000),
                     rotation6d_max_abs=float(np.abs(delta[:, list(range(3, 9)) + list(range(32, 38))]).max()),
                     hand_max_rad=float(np.abs(delta[:, list(range(9, 29)) + list(range(38, 58))]).max()))
        return applied, stats

    try:
        for step_index, step in enumerate(args.steps):
            checkpoint = args.checkpoint_dir / f"checkpoint_step_{step}.safetensors"
            if step_index:
                architecture.load_checkpoint(str(checkpoint), strict=True)
                architecture.eval()
            print("CHECKPOINT", step, flush=True)
            dump(args.out / f"checkpoint_{step}.json", dict(path=str(checkpoint), size=checkpoint.stat().st_size,
                                                          sha256=sha256(checkpoint)))
            for i, source in enumerate(selected):
                item = dict(source, case=i)
                obs = prepare(item, item["offset"])
                for seed in args.seeds:
                    a, elapsed = infer(obs, obs["gt"][:10], seed)
                    record(step, "clean", item, obs, a, obs["gt"][:10], seed, elapsed)
                if item["group"] == "fast" and i in [24, 25, 26, 27, 28, 29]:
                    direction = obs["gt"][9, 29:32] - obs["gt"][0, 29:32]
                    direction /= np.linalg.norm(direction)
                    shifted = obs["state"].copy()
                    shifted[29:32] -= 0.015 * direction
                    for seed in args.seeds:
                        a, elapsed = infer(obs, obs["gt"][:10], seed, state=shifted)
                        record(step, "state_minus15mm", item, obs, a, obs["gt"][:10], seed, elapsed,
                               metric_state=shifted)
            for item in trajectories:
                obs0 = prepare(item, item["offset"])
                first, elapsed = infer(obs0, None, 42)
                raw_queue = first[:16].copy()
                projected_queue, bootstrap_projection = project(raw_queue, obs0["state"], obs0["state"])
                dump(args.out / f"bootstrap_{step}_{item['case']}.json",
                     dict(actions=first.tolist(), projected=projected_queue.tolist(), projection=bootstrap_projection,
                          seconds=elapsed))
                for j in range(args.rollout_requests):
                    offset = item["offset"] + 6 + 16 * j
                    obs = prepare(item, offset)
                    raw_prefix = raw_queue[-10:].copy()
                    projected_prefix = projected_queue[-10:].copy()
                    for mode, prefix in (("rollout_teacher", obs["gt"][:10]),
                                         ("rollout_raw", raw_prefix), ("rollout_projected", projected_prefix)):
                        a, elapsed = infer(obs, prefix, 42)
                        extra = dict(request=j + 1, offset=offset)
                        if mode == "rollout_raw":
                            raw_queue = a[10:26].copy()
                        elif mode == "rollout_projected":
                            projected_queue, stats = project(a[10:26], prefix[-1], obs0["state"])
                            applied = np.concatenate((prefix, projected_queue, a[26:]), axis=0)
                            extra.update(projection=stats, applied_actions=applied.tolist(),
                                         applied_metrics=seam_metrics(applied, prefix, obs["gt"], obs["state"]))
                        record(step, mode, item, obs, a, prefix, 42, elapsed, **extra)
            dump(args.out / f"complete_{step}.json", dict(checkpoint=step, cumulative_inferences=count))
        dump(args.out / "complete.json", dict(inferences=count, seconds=time.monotonic() - started, completed=True))
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
