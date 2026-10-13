"""Shared R2 engineering loop with paired exposures and full-state resume.

The CLI remains limited to engineering updates. Profiling and formal training
require their own validated plans and entry points.
"""

import argparse
from contextlib import ExitStack, nullcontext
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from openwam.dataloader.registry import build_dataset
from openwam.train.utils.ckpt_model_loader import copy_ckpt_artifacts
from openwam.train.utils.optimizer_groups import build_trainable_parameters
from openwam.train.utils.seeding import seed_everything
from scripts.statewam.initialize_checkpoint import build_skeleton
from scripts.statewam.r2.checkpoints import load_checkpoint, main_operation, save_checkpoint, validate_checkpoint
from scripts.statewam.r2.common import GOAL_ID, atomic_json, sha256, stable_hash
from scripts.statewam.r2.cuda_identity import verify_cuda_devices
from scripts.statewam.r2.exposure_sampler import ExposureBatchSampler
from scripts.statewam.r2.initialize_matrix import load_exact_initialization
from scripts.statewam.r2.matrix import GROUPS, apply_parameter_policy
from scripts.statewam.r2.paired_data import PairedExposureDataset, flow_randomness, randomness_hashes
from scripts.statewam.r2.profile_metrics import StepTimings, summarize_steps, timing_sample
from scripts.statewam.r2.source_snapshot import assert_sources_unchanged, capture_sources
from scripts.statewam.r2.workload import WorkloadTrace
from scripts.train import _build_accelerator


class LossModule(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture

    def forward(self, inputs, randomness):
        return self.architecture.compute_loss(**inputs, flow_randomness=randomness,
                                              lambda_video=1.0, lambda_action=1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--end-update", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--audit-workload", action="store_true")
    parser.add_argument("--audit-timing", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.updates <= 8 or min(args.micro_batch, args.accumulation, args.save_every) < 1 or args.workers < 0:
        raise ValueError("Engineering only: 1..8 updates and valid batch/save settings required")
    run_experiment(args)


def run_experiment(args, *, protocol_id="r2-model-engineering-v4", seed=20260913,
                   purpose="engineering", plan_identity=None):
    if purpose not in ("engineering", "profiling", "training"):
        raise ValueError("Formal training requires its own frozen entry point")
    if purpose == "engineering" and not 1 <= args.updates <= 8:
        raise ValueError("Engineering budget exceeded")
    if purpose == "profiling":
        from scripts.statewam.r2.profile_training import validate_plan
        plan = validate_plan(plan_identity["plan"])
        if (args.group not in plan["groups"] or args.updates != plan["updates"] or seed != plan["seed"]
                or protocol_id != plan["protocol_id"] or args.audit_workload):
            raise ValueError("Profiling call differs from registered plan")
    if purpose == "training":
        from scripts.statewam.r2.formal_training import validate_call
        validate_call(args, plan_identity, seed=seed, protocol_id=protocol_id)
    timing_enabled = purpose in ("profiling", "training") or getattr(args, "audit_timing", False)
    end_update = args.end_update or args.updates
    if not 1 <= end_update <= args.updates:
        raise ValueError("Invalid end update")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    actual_devices = verify_cuda_devices()
    matrix = json.loads((args.matrix / "initialization.json").read_text())
    marker = json.loads((args.matrix / "COMPLETE.json").read_text())
    if marker["initialization_sha256"] != sha256(args.matrix / "initialization.json"):
        raise ValueError("Matrix manifest integrity failed")
    source = args.matrix / args.group
    if sha256(source / "config.yaml") != matrix["groups"][args.group]["config_sha256"]:
        raise ValueError("Matrix group configuration changed")
    cfg = OmegaConf.load(source / "config.yaml")
    cfg.training.batch_size = args.micro_batch
    cfg.training.gradient_accumulation_steps = args.accumulation
    cfg.project.seed = seed
    if purpose == "training":
        from scripts.statewam.r2.formal_training import configure_formal
        cfg = configure_formal(cfg, plan_identity["plan"], args.group)
    seed_everything(cfg.project.seed, rank=int(os.environ.get("RANK", 0)))
    accelerator = _build_accelerator(cfg)
    if accelerator.num_processes != 4:
        raise ValueError("Real-model engineering uses four ranks")
    identity = {"goal_id": GOAL_ID, "protocol_id": protocol_id, "group": args.group,
                "matrix_sha256": marker["initialization_sha256"], "updates": args.updates,
                "micro_batch": args.micro_batch, "accumulation": args.accumulation,
                "world_size": accelerator.num_processes,
                "global_batch": args.micro_batch * args.accumulation * accelerator.num_processes,
                "seed": cfg.project.seed, "zero_stage": 2,
                "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
                "workload_trace": args.audit_workload,
                "capacity_pooling": "deterministic_separable_fp32_v2",
                "config_sha256": stable_hash(OmegaConf.to_container(cfg, resolve=True)),
                "episode_manifest_sha256": sha256(cfg.dataloader.episode_manifest),
                "statistics_sha256": sha256(cfg.dataloader.normalization_stats_path)}
    if timing_enabled:
        identity.update(purpose=purpose, timing="cuda_events_and_host_v1", workers=args.workers,
                        plan_identity=plan_identity)
    position = {"optimizer_update": 0, "next_occurrence": 0}
    if args.resume:
        position = main_operation(accelerator, lambda: validate_checkpoint(args.resume, identity=identity))
    if position["optimizer_update"] >= end_update:
        raise ValueError("Requested update range is already complete")

    def setup():
        args.output.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(cfg, args.output / "config.yaml")
        atomic_json(args.output / "identity.json", identity)
        copy_ckpt_artifacts(str(source), str(args.output))
        return capture_sources(args.output / "source_snapshot")

    source_manifest = main_operation(accelerator, setup)
    if args.resume:
        previous_sources = args.resume.parent.parent / "source_snapshot/manifest.json"
        if not previous_sources.exists() or json.loads(previous_sources.read_text()) != source_manifest:
            raise ValueError("Resume requires identical preserved executable sources")
    print(f"rank={accelerator.process_index} building {args.group}", flush=True)
    arch = build_skeleton(cfg, source)
    load_exact_initialization(arch, source / "checkpoint_step_0.safetensors")
    counts = apply_parameter_policy(arch)
    arch.init_training_schedulers(1000)
    arch.set_training_runtime(use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
                              max_timestep_boundary=cfg.training.max_timestep_boundary,
                              min_timestep_boundary=cfg.training.min_timestep_boundary)
    optimizer = torch.optim.AdamW(
        build_trainable_parameters(SimpleNamespace(architecture=arch), action_lr=cfg.training.action_lr),
        lr=cfg.training.learning_rate, betas=tuple(cfg.training.adam_betas),
        eps=cfg.training.adam_eps, weight_decay=cfg.training.weight_decay,
    )
    accelerator.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.micro_batch
    model, optimizer = accelerator.prepare(LossModule(arch), optimizer)
    arch.set_dtype_device(torch.bfloat16, accelerator.device)
    arch.move_frozen_to_device(accelerator.device)
    model.train()
    data = build_dataset(cfg.dataloader, split="train")
    paired = PairedExposureDataset(data, seed=cfg.project.seed, exposures=args.updates * identity["global_batch"])
    sampler = ExposureBatchSampler(updates=end_update, micro_batch=args.micro_batch, accumulation=args.accumulation,
                                    world_size=accelerator.num_processes, rank=accelerator.process_index,
                                    start_update=position["optimizer_update"])
    loader = torch.utils.data.DataLoader(paired, batch_sampler=sampler, num_workers=args.workers, collate_fn=list,
                                         generator=torch.Generator().manual_seed(cfg.project.seed), pin_memory=True)
    if args.resume:
        position = load_checkpoint(accelerator, args.resume, identity=identity)
    gradients = {}
    hooks = []
    for name, parameter in arch.named_parameters():
        if parameter.requires_grad:
            def capture(gradient, name=name):
                gradients[name] = torch.stack((gradient.detach().float().norm(), gradient.isfinite().all().float()))
            hooks.append(parameter.register_hook(capture))
    torch.cuda.reset_peak_memory_stats()
    main_operation(accelerator, lambda: assert_sources_unchanged(source_manifest))
    started = previous = time.monotonic()
    checkpoint = None
    stop_path = Path(os.environ["R2_STOP_FILE"])
    rank = accelerator.process_index
    trace = WorkloadTrace(arch) if args.audit_workload else None
    timings = StepTimings() if timing_enabled else None
    timing_rows = []
    invocation = {"start_update": position["optimizer_update"], "end_update": end_update,
                  "resume": str(args.resume) if args.resume else None, "purpose": purpose}
    atomic_json(args.output / f"invocation_rank{rank}.json", invocation)
    with ExitStack() as stack, (args.output / f"steps_rank{rank}.jsonl").open("x") as stream:
        if trace:
            stack.enter_context(trace)
        if timings:
            stack.enter_context(timings.engine_phases(model))
        for batch in loader:
            arrived = time.monotonic()
            gradients.clear()
            torch.cuda.synchronize()
            with (trace.during("preprocessing") if trace else nullcontext()), (timings.during("preprocessing") if timings else nullcontext()):
                inputs = arch.prepare_inputs(batch)
            torch.cuda.synchronize()
            prepared = time.monotonic()
            occurrences = [sample["pairing"]["occurrence"] for sample in batch]
            with timings.during("paired_noise_and_hashes") if timings else nullcontext():
                randomness = flow_randomness(cfg.project.seed, occurrences, video=inputs["input_latents"],
                                             actions=inputs["actions"],
                                             video_bounds=(int(inputs["min_timestep_boundary"] * 1000),
                                                           int(inputs["max_timestep_boundary"] * 1000)))
                noise_hashes = randomness_hashes(randomness)
            with accelerator.accumulate(model):
                with (trace.during("forward") if trace else nullcontext()), (timings.during("forward") if timings else nullcontext()):
                    losses = model(inputs, randomness)
                with (trace.during("backward_recomputation") if trace else nullcontext()), (timings.during("backward_and_update_wrapper") if timings else nullcontext()):
                    accelerator.backward(losses["loss"])
                optimizer.step()
                optimizer.zero_grad()
            torch.cuda.synchronize()
            finished = time.monotonic()
            if accelerator.sync_gradients:
                position["optimizer_update"] += 1
                position["next_occurrence"] = position["optimizer_update"] * identity["global_batch"]
            gradient_rows = {name: value.cpu().tolist() for name, value in gradients.items()}
            metrics = {name: value.detach().float().item() for name, value in losses.items()}
            if any(not torch.isfinite(torch.tensor(value)) for value in metrics.values()) or any(v[1] != 1 for v in gradient_rows.values()):
                raise ValueError("Nonfinite loss or gradients")
            row = {"position": dict(position), "samples": [sample["pairing"] for sample in batch],
                   "noise_hashes": noise_hashes, "metrics": metrics, "gradients": gradient_rows,
                   "wait_seconds": arrived - previous, "preprocess_seconds": prepared - arrived,
                   "compute_seconds": finished - prepared,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            if timings:
                row["phase_timings"] = timings.resolve()
                row["microstep_wall_seconds"] = time.monotonic() - previous
                timing_rows.append(timing_sample(row))
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            stopping = torch.tensor(int(stop_path.exists()), device=accelerator.device)
            dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
            boundary = accelerator.sync_gradients
            if boundary and (position["optimizer_update"] % args.save_every == 0 or stopping.item()
                             or position["optimizer_update"] == end_update):
                save_started = time.monotonic()
                main_operation(accelerator, lambda: assert_sources_unchanged(source_manifest))
                checkpoint = save_checkpoint(accelerator, args.output / "checkpoints", identity=identity, position=position)
                atomic_json(args.output / f"save_update{position['optimizer_update']}_rank{rank}.json",
                            {"seconds": time.monotonic() - save_started, "checkpoint": str(checkpoint)})
            atomic_json(args.output / f"progress_rank{rank}.json", {"position": position, "time": time.time()})
            if rank == 0:
                print(f"update={position['optimizer_update']} metrics={metrics} compute={finished-prepared:.3f}s", flush=True)
            if boundary and stopping.item():
                break
            previous = time.monotonic()
    if trace:
        atomic_json(args.output / f"workload_rank{rank}.json", {
            "identity": identity, **trace.report(),
            "parameter_inventory": {name: {"numel": parameter.numel(), "shape": list(parameter.shape),
                                            "trainable": parameter.requires_grad}
                                    for name, parameter in arch.named_parameters()},
        })
    for hook in hooks:
        hook.remove()
    if timings:
        warmup = 10 if purpose in ("profiling", "training") else 2
        observed_updates = len(timing_rows) // args.accumulation
        timing_report = {"identity": identity, "invocation": invocation, "completed_updates": observed_updates,
                         "warmup_updates": warmup, "steady_summary": None}
        if observed_updates > warmup:
            timing_report["steady_summary"] = summarize_steps(
                timing_rows, accumulation=args.accumulation, micro_batch=args.micro_batch,
                world_size=accelerator.num_processes, warmup_updates=warmup)
        atomic_json(args.output / f"timing_rank{rank}.json", timing_report)
    atomic_json(args.output / f"summary_rank{rank}.json", {
        "identity": identity, "position": position, "seconds": time.monotonic() - started,
        "parameter_counts": counts, "cuda_uuids": actual_devices, "checkpoint": str(checkpoint),
        "stopped": stop_path.exists(), "limits": f"{purpose} only; no convergence or efficacy conclusion",
    })
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
