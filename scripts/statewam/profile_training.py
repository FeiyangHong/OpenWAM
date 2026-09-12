"""Profile the repository's real trainer with a bounded optimizer-update budget."""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openwam.dataloader.registry import build_dataset  # noqa: E402
from openwam.train.openwam_trainer import OpenWAMTrainer  # noqa: E402
from scripts.train import _build_accelerator  # noqa: E402


class ProfileTrainer(OpenWAMTrainer):
    def __init__(self, cfg, *, accelerator, dataset, profile_dir):
        self.profile_dir = profile_dir
        self.events = []
        self.gradients = {}
        self.previous_end = time.monotonic()
        super().__init__(cfg, accelerator=accelerator, dataset=dataset)
        for name, p in self.architecture.named_parameters():
            if name.startswith("interaction_state.") and p.requires_grad and name.endswith(("gate", "video_proj.weight", "output.weight")):
                p.register_hook(self.gradient_hook(name))

    def gradient_hook(self, name):
        def capture(grad):
            # Scalars only; never retain the training graph in diagnostics.
            self.gradients[name] = grad.detach().float().norm()
        return capture

    def prepare_accelerate(self, optimizer, dataloader, scheduler):
        result = super().prepare_accelerate(optimizer, dataloader, scheduler)
        self.profile_optimizer = result[0]
        return result

    def optimizer_fingerprint(self):
        optimizer = self.profile_optimizer
        while not isinstance(optimizer, torch.optim.Optimizer):
            optimizer = optimizer.optimizer
        records = []
        for index, state in enumerate(optimizer.state.values()):
            entry = {"index": index}
            for key, value in state.items():
                if torch.is_tensor(value):
                    array = value.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy()
                    entry[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                                  "sha256": hashlib.sha256(array).hexdigest()}
                else:
                    entry[key] = value
            records.append(entry)
        return records

    def compute_loss(self, batch):
        torch.cuda.synchronize()
        self.step_start = time.monotonic()
        self.wait_seconds = self.step_start - self.previous_end
        self.gradients.clear()
        inputs = self.architecture.prepare_inputs(batch)
        torch.cuda.synchronize()
        self.preprocess_seconds = time.monotonic() - self.step_start
        result = self.architecture.compute_loss(
            **inputs, lambda_video=self.lambda_video, lambda_action=self.lambda_action,
        )
        return {"total": result["loss"], "video": result["loss_video"], "action": result["loss_action"]}

    def log_step(self, **kwargs):
        torch.cuda.synchronize()
        now = time.monotonic()
        row = {
            "micro_step": kwargs["global_step"], "optimizer_update": kwargs["opt_step"],
            "compute_seconds": now - self.step_start, "preprocess_seconds": self.preprocess_seconds,
            "wait_and_bookkeeping_seconds": self.wait_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "metrics": kwargs["metrics"],
            "state_gradient_norms": {k: float(v) for k, v in self.gradients.items()},
        }
        if not all(torch.isfinite(torch.tensor(v)) for v in row["state_gradient_norms"].values()):
            raise RuntimeError("Nonfinite State gradients")
        if not row["state_gradient_norms"]:
            raise RuntimeError("No State gradient hooks fired")
        if not all(torch.isfinite(torch.tensor(v)) for v in row["metrics"].values()):
            raise RuntimeError("Nonfinite training metrics")
        self.events.append(row)
        rank = self.accelerator.process_index
        with (self.profile_dir / f"steps_rank{rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if rank == 0 and (row["micro_step"] <= 3 or row["micro_step"] % 10 == 0):
            print(f"[profile] update={row['optimizer_update']} compute={row['compute_seconds']:.3f}s "
                  f"allocated={row['peak_allocated_bytes'] / 2**30:.2f}GiB metrics={row['metrics']}", flush=True)
        super().log_step(**kwargs)
        self.previous_end = time.monotonic()

    def resume_if_configured(self, *args):
        torch.cuda.synchronize()
        start = time.monotonic()
        result = super().resume_if_configured(*args)
        torch.cuda.synchronize()
        (self.profile_dir / f"resume_rank{self.accelerator.process_index}.json").write_text(
            json.dumps({"seconds": time.monotonic() - start, "restored_position": result,
                        "optimizer_state": self.optimizer_fingerprint()}) + "\n"
        )
        return result

    def finish_training(self, output_path, global_step, save_steps, keep_last_k, is_main, wandb_run):
        # Keep the final full state so the required resume audit can run.
        self.accelerator.wait_for_everyone()
        if is_main:
            (self.profile_dir / "checkpoint_directory.txt").write_text(str(output_path) + "\n")
        if wandb_run is not None:
            wandb_run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU micro-batch; released checkpoint uses 4")
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--world-trainable", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    launch_start = time.monotonic()
    if args.updates < 1 or args.updates > 500 or args.batch_size < 1 or args.accumulation < 1:
        raise ValueError("Require 1..500 optimizer updates and positive batch/accumulation")
    if int(os.environ.get("WORLD_SIZE", "1")) > 4:
        raise ValueError("This Goal permits at most four GPUs")
    # This executable never initiates online experiment tracking.
    os.environ["WANDB_MODE"] = "disabled"
    cfg = OmegaConf.load(ROOT / "configs/train.yaml")
    del cfg.defaults
    source_cfg = OmegaConf.load(args.checkpoint / "config.yaml")
    cfg = OmegaConf.merge(cfg, source_cfg)
    cfg.model.architecture.parameter_freeze = [] if args.world_trainable else ["video_backbone.dit"]
    cfg.dataloader.dataset_dir = str(args.dataset.resolve())
    cfg.dataloader.normalization_stats_path = str((args.checkpoint / "normalization_stats.npy").resolve())
    cfg.dataloader.variant = "clean_50"
    cfg.dataloader.seed = 42
    cfg.project.seed = 42
    cfg.project.output_dir = str(args.output.resolve())
    cfg.project.wandb.entity = None
    t = cfg.training
    t.debug = False
    t.num_epochs = None
    t.max_steps = args.updates * args.accumulation
    t.batch_size = args.batch_size
    t.gradient_accumulation_steps = args.accumulation
    t.dataset_num_workers = args.workers
    t.output_path = str(args.output.resolve())
    t.finetune_ckpt_path = None if args.resume else str(args.checkpoint.resolve())
    t.resume_ckpt_path = str(args.checkpoint.resolve()) if args.resume else None
    t.use_gradient_checkpointing = True
    t.save_steps = t.max_steps
    t.save_full_states_for_resume = True
    t.keep_last_k_ckpts = 2
    t.learning_rate, t.action_lr, t.video_lr = 1e-4, 1e-5, 1e-6
    t.lr_scheduler = None
    OmegaConf.resolve(cfg)
    profile_dir = args.output / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    accelerator = _build_accelerator(cfg)
    if accelerator.is_main_process:
        OmegaConf.save(cfg, profile_dir / "requested_config.yaml")
    dataset = build_dataset(cfg.dataloader, split="train")
    trainer = ProfileTrainer(cfg, accelerator=accelerator, dataset=dataset, profile_dir=profile_dir)
    # Time actual save calls in the original loop without replacing checkpoint logic.
    import openwam.train.openwam_trainer as trainer_module
    save_events = []
    originals = {}
    for name in ("save_weights", "save_full_state"):
        original = getattr(trainer_module, name)
        originals[name] = original

        def timed(*a, _name=name, _fn=original, **kw):
            torch.cuda.synchronize()
            start = time.monotonic()
            result = _fn(*a, **kw)
            torch.cuda.synchronize()
            save_events.append({"operation": _name, "seconds": time.monotonic() - start})
            return result

        setattr(trainer_module, name, timed)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    try:
        trainer.train()
    finally:
        for name, original in originals.items():
            setattr(trainer_module, name, original)
        elapsed = time.monotonic() - start
        rows = trainer.events
        steady = rows[10:] if len(rows) > 10 else rows
        times = [r["compute_seconds"] for r in steady]
        intervals = [r["compute_seconds"] + r["wait_and_bookkeeping_seconds"] for r in steady]
        report = {
            "goal_id": "r1/robotwin-smoke", "argv": sys.argv,
            "device": torch.cuda.get_device_name(), "world_size": accelerator.num_processes,
            "batch_per_gpu": args.batch_size, "gradient_accumulation": args.accumulation,
            "wall_seconds": elapsed, "micro_steps_observed": len(rows),
            "optimizer_update_final": rows[-1]["optimizer_update"] if rows else None,
            "steady_compute_seconds_mean": statistics.mean(times) if times else None,
            "steady_samples_per_second": args.batch_size * accelerator.num_processes / statistics.mean(times) if times else None,
            "steady_end_to_end_samples_per_second": args.batch_size * accelerator.num_processes / statistics.mean(intervals) if intervals else None,
            "global_batch": args.batch_size * accelerator.num_processes * args.accumulation,
            "sample_exposures": len(rows) * args.batch_size * accelerator.num_processes,
            "gpu_hours_training_and_save": elapsed * accelerator.num_processes / 3600,
            "startup_seconds": start - launch_start,
            "warmup_micro_steps": 10 if len(rows) > 10 else 0,
            "steady_preprocess_seconds_mean": statistics.mean(r["preprocess_seconds"] for r in steady) if steady else None,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "save_events": save_events,
            "timing_note": "Compute includes preprocessing/forward/backward/update; wait includes data and bookkeeping, not pure disk I/O.",
            "optimizer_state": trainer.optimizer_fingerprint() if hasattr(trainer, "profile_optimizer") else [],
        }
        (profile_dir / f"summary_rank{accelerator.process_index}.json").write_text(json.dumps(report, indent=2) + "\n")
    accelerator.wait_for_everyone()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
