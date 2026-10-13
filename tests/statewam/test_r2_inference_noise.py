from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openwam.model.architectures.base import BaseWAMArchitecture


def generate_noise(world_seed=42, action_seed=None):
    observed = {}

    def preprocess(**kwargs):
        observed["world"] = torch.randn(1, 2, 3, generator=torch.Generator().manual_seed(kwargs["seed"]))
        return {"latents": observed["world"].clone()}

    def forward(actions, action_timestep, **kwargs):
        observed["action"] = actions.clone()
        return torch.zeros_like(kwargs["latents"]), torch.zeros_like(actions)

    architecture = SimpleNamespace(
        eval=lambda: None, device=torch.device("cpu"), dtype=torch.float32,
        video_backbone=SimpleNamespace(preprocess_input_for_inference=preprocess),
        action_dim=80, uses_proprioception=False, normalizer=None,
        _resolve_inactive_action_dims=lambda *args: None,
        video_scheduler=SimpleNamespace(num_train_timesteps=1000),
        action_scheduler=SimpleNamespace(num_train_timesteps=1000, flow_step=lambda pred, sigma, next_sigma, latents: latents),
        forward=forward,
    )
    result = BaseWAMArchitecture.generate(architecture, [(1000, 1000), (0, 0)], "fixed instruction",
                                         action_num_frames=33, decode_video=False, seed=world_seed, action_seed=action_seed)
    return observed, result


def test_default_action_noise_remains_legacy_seeded_trajectory():
    observed, result = generate_noise()
    expected = torch.randn(1, 32, 80, generator=torch.Generator().manual_seed(42))
    assert torch.equal(observed["action"], expected)
    assert np.array_equal(result["actions"], expected.squeeze(0).numpy())


def test_explicit_action_seed_changes_only_action_noise_and_preserves_global_rng():
    before = torch.get_rng_state().clone()
    baseline, _ = generate_noise(42, 13)
    changed_action, _ = generate_noise(42, 14)
    changed_world, _ = generate_noise(43, 13)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(baseline["world"], changed_action["world"])
    assert not torch.equal(baseline["action"], changed_action["action"])
    assert torch.equal(baseline["action"], changed_world["action"])
    assert not torch.equal(baseline["world"], changed_world["world"])


@pytest.mark.parametrize("seed", [True, -1, 2**64, 1.5])
def test_invalid_action_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="action_seed"):
        generate_noise(action_seed=seed)


def test_joint_engine_forwards_explicit_seeds(monkeypatch):
    from omegaconf import OmegaConf
    from openwam.deploy import engine as module

    received = {}

    def generate(**kwargs):
        received.update(kwargs)
        return {"actions": np.zeros((32, 20))}

    cfg = OmegaConf.create({"inference": {"denoise_steps": 10, "denoise_mode": "sync"}})
    architecture = SimpleNamespace(generate=generate, video_scheduler=None, action_scheduler=None)
    monkeypatch.setattr(module, "make_schedule", lambda *args, **kwargs: [(1000, 1000), (0, 0)])
    engine = module.JointInferenceEngine(cfg, architecture=architecture)
    engine.generate({"prompt": "fixed", "seed": 101, "action_seed": 202})
    assert received["seed"] == 101 and received["action_seed"] == 202
    engine.generate({"prompt": "fixed"})
    assert received["seed"] == 42 and received["action_seed"] is None
