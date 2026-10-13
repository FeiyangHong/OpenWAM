"""Validation of optional explicit flow-matching randomness."""

import torch


def validate_flow_randomness(randomness, *, video, actions, video_bounds, action_count):
    expected = {"video_timestep_ids", "video_noise"}
    if actions is not None:
        expected |= {"action_timestep_ids", "action_noise"}
    if set(randomness) != expected:
        raise ValueError(f"Explicit flow randomness requires exactly {sorted(expected)}")
    for name, reference, bounds in (
        ("video", video, video_bounds), ("action", actions, (0, action_count)),
    ):
        if reference is None:
            continue
        ids, noise = randomness[f"{name}_timestep_ids"], randomness[f"{name}_noise"]
        if (not isinstance(ids, torch.Tensor) or ids.device.type != "cpu" or ids.dtype != torch.int64
                or ids.shape != (len(video),) or (ids < bounds[0]).any() or (ids >= bounds[1]).any()):
            raise ValueError(f"Invalid explicit {name} timestep IDs")
        if (not isinstance(noise, torch.Tensor) or noise.shape != reference.shape
                or noise.device != reference.device or noise.dtype != reference.dtype
                or noise.requires_grad or not noise.isfinite().all()):
            raise ValueError(f"Invalid explicit {name} noise")
