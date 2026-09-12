"""Real-weight, real-data baseline/off/gate-zero and deployment parity audit."""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from openwam.dataloader.registry import build_dataset  # noqa: E402
from openwam.deploy.model_loader import load_from_checkpoint_dir  # noqa: E402


def fixed_forward(model, prepared):
    device, dtype = model.device, model.dtype
    generator = torch.Generator(device=device).manual_seed(731)
    latents = torch.randn(prepared["input_latents"].shape, generator=generator, device=device, dtype=dtype)
    latents[:, :, :1] = prepared["first_frame_latents"]
    actions = torch.randn(prepared["actions"].shape, generator=generator, device=device, dtype=dtype)
    return model(
        actions, torch.tensor([600.], device=device, dtype=dtype),
        latents=latents, timestep=torch.tensor([300.], device=device, dtype=dtype),
        context=prepared["context"], seq_lens=prepared.get("seq_lens"),
        first_frame_latents=prepared["first_frame_latents"],
        proprio=prepared["proprio"], _proprio_sample_mask=prepared.get("proprio_mask"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.baseline / "config.yaml")
    cfg.dataloader.dataset_dir = str(args.dataset.resolve())
    cfg.dataloader.normalization_stats_path = str((args.baseline / "normalization_stats.npy").resolve())
    cfg.dataloader.color_jitter.enabled = False
    dataset = build_dataset(cfg.dataloader, split="train")
    torch.manual_seed(42)
    sample = dataset[0]
    print("Loading baseline", flush=True)
    _, baseline = load_from_checkpoint_dir(str(args.baseline), device="cuda:0")
    with torch.no_grad():
        prepared = baseline.prepare_inputs([sample])
        expected = tuple(v.cpu() for v in fixed_forward(baseline, prepared))
    del baseline
    gc.collect()
    torch.cuda.empty_cache()
    print("Loading converted checkpoint", flush=True)
    _, model = load_from_checkpoint_dir(str(args.state), device="cuda:0")
    result = {"device": torch.cuda.get_device_name(), "dtype": str(model.dtype), "checks": {}}
    with torch.no_grad():
        for source in ("off", "current", "future", "future_action"):
            model.interaction_state.source = source
            actual = fixed_forward(model, prepared)
            errors = [float((x.cpu().float() - y.float()).abs().max()) for x, y in zip(actual, expected)]
            for x, y in zip(actual, expected):
                torch.testing.assert_close(x.cpu(), y, rtol=0, atol=0)
            result["checks"][source] = {"max_absolute_error": errors, "pass": True}
            print(f"Parity {source}: {errors}", flush=True)
    model.interaction_state.source = "future_action"
    model.freeze_modules(list(cfg.model.freeze))
    model.freeze_parameters(["video_backbone.dit"])
    model.init_training_schedulers()
    model.train()
    optimizer = torch.optim.AdamW(model.interaction_state.parameters(), lr=1e-4)
    gradients = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(**{**prepared, "use_gradient_checkpointing": True})["loss"]
        loss.backward()
        row = {name: float(p.grad.float().norm()) for name, p in model.interaction_state.named_parameters()
               if p.grad is not None and name.endswith(("gate", "video_proj.weight", "output.weight"))}
        if not row or not all(np.isfinite(v) for v in row.values()):
            raise RuntimeError("Missing or nonfinite gradients")
        if step > 0 and not all(v > 0 for v in row.values()):
            raise RuntimeError("State readers/writers did not start learning")
        gradients.append({"step": step, "loss": float(loss), "gradients": row})
        optimizer.step()
        print(f"Backward {step}: {float(loss):.6f}", flush=True)
    result["gradients"] = gradients
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified: {args.output / 'verification.json'}", flush=True)


if __name__ == "__main__":
    main()
