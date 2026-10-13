"""Four-rank ZeRO-2 resume equivalence probe, separate from model profiling."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator, DeepSpeedPlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.statewam.r2.checkpoints import load_checkpoint, save_checkpoint
from scripts.statewam.r2.common import atomic_json
from scripts.statewam.r2.exposure_sampler import ExposureBatchSampler
from scripts.statewam.r2.cuda_identity import verify_cuda_devices
from scripts.statewam.r2.paired_data import stream_seed, tensor_hash


def fingerprints(optimizer):
    while not isinstance(optimizer, torch.optim.Optimizer):
        optimizer = optimizer.optimizer
    return [{key: tensor_hash(value) if torch.is_tensor(value) else value for key, value in row.items()}
            for row in optimizer.state.values()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--end-update", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.end_update <= 4:
        raise ValueError("Bounded to four optimizer updates")
    identity = {"protocol_id": "r2-zero2-probe-v1", "updates": 4, "world_size": 4,
                "global_batch": 16, "micro_batch": 2, "accumulation": 2, "seed": 7,
                "zero_stage": 2, "model": "linear-dropout-linear", "data": "synthetic-by-occurrence"}
    actual_devices = verify_cuda_devices()
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=2,
                              deepspeed_plugin=DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=2,
                                                             gradient_clipping=1.0))
    if accelerator.num_processes != 4:
        raise ValueError("Four ranks required")
    seed = 7 + accelerator.process_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.SiLU(), torch.nn.Dropout(0.1),
                                torch.nn.Linear(64, 8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    dummy = torch.utils.data.DataLoader(torch.arange(32), batch_size=2)
    model, optimizer, dummy, scheduler = accelerator.prepare(model, optimizer, dummy, scheduler)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if accelerator.is_main_process:
        args.output.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()
    started = time.monotonic()
    position = {"optimizer_update": 0, "next_occurrence": 0}
    if args.resume:
        position = load_checkpoint(accelerator, args.resume, identity=identity)
    sampler = ExposureBatchSampler(updates=args.end_update, micro_batch=2, accumulation=2,
                                    world_size=4, rank=accelerator.process_index,
                                    start_update=position["optimizer_update"])
    rows = []
    for occurrences in sampler:
        x = torch.stack([torch.randn(32, generator=torch.Generator().manual_seed(stream_seed(7, i, "probe")))
                         for i in occurrences]).to(accelerator.device, dtype=torch.bfloat16)
        target = x[:, :8] * 0.5
        with accelerator.accumulate(model):
            output = model(x)
            loss = (output.float() - target.float()).square().mean()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        if accelerator.sync_gradients:
            position["optimizer_update"] += 1
            position["next_occurrence"] = position["optimizer_update"] * identity["global_batch"]
        rows.append({"occurrences": occurrences, "input_sha256": tensor_hash(x),
                     "loss": loss.item(), "update": position["optimizer_update"],
                     "python_random": random.random(), "numpy_random": float(np.random.random()),
                     "torch_random": torch.rand(1).item(), "cuda_random": torch.rand(1, device="cuda").item()})
        if accelerator.sync_gradients and position["optimizer_update"] == args.end_update:
            checkpoint = save_checkpoint(accelerator, args.output / "checkpoints", identity=identity, position=position)
    model_state = {name: tensor_hash(value) for name, value in accelerator.unwrap_model(model).state_dict().items()}
    atomic_json(args.output / f"rank_{accelerator.process_index}.json", {
        "identity": identity, "position": position, "rows": rows, "model": model_state,
        "cuda_uuids": actual_devices, "python_executable": sys.executable,
        "optimizer": fingerprints(optimizer), "scheduler": scheduler.state_dict(),
        "checkpoint": str(checkpoint), "seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    })
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
