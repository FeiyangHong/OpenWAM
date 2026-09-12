"""State routing and autograd through real (small) Wan and ActionDiT blocks."""

import copy
from types import SimpleNamespace

import pytest
import torch

from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.statewam.architecture import StateWAMArchitecture
from openwam.model.architectures.statewam.state_expert import CrossAttention, InteractionStateExpert
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan_backbone import Wan21


def tiny_architecture(cls=StateWAMArchitecture, source="future_action"):
    cfg = dict(
        video_dim=96, dim=32, ffn_dim=64, num_heads=4, attn_head_dim=24,
        num_dit_layers=2, bridge_interval=1, text_dim=32, action_dim=7,
        state_dim=7, use_proprioception=True, attention_mask_mode="mutual",
        video_attention_mask_mode="first_frame_causal", mot_checkpoint_mixed_attn=True,
        interaction_state=dict(source=source, dim=32, num_tokens=4, num_heads=4, state_bridge_layers=[0, 1]),
    )
    arch = cls(cfg)
    dit = WanModel(
        dim=96, in_dim=4, ffn_dim=128, out_dim=4, text_dim=32, freq_dim=32,
        eps=1e-6, patch_size=(1, 2, 2), num_heads=4, num_layers=2,
        has_image_input=False, seperated_timestep=True, fuse_vae_embedding_in_latents=True,
    )
    arch.video_backbone = Wan21(SimpleNamespace(dit=dit, vace=None, motion_controller=None))
    if isinstance(arch, StateWAMArchitecture):
        arch._init_state_expert()
    else:
        arch.build_mot_driver()
    arch.set_dtype_device(torch.float32, torch.device("cpu"))
    return arch


def inputs():
    return dict(
        noisy_actions=torch.randn(2, 3, 7), action_timestep=torch.tensor([100., 800.]),
        latents=torch.randn(2, 4, 3, 4, 4), timestep=torch.tensor([700., 200.]),
        first_frame_latents=torch.randn(2, 4, 1, 4, 4), context=torch.randn(2, 4, 32),
        context_mask=torch.tensor([[True, True, False, False], [True, True, True, False]]),
        proprio=torch.randn(2, 7),
    )


def test_off_zero_gate_and_roundtrip(tmp_path):
    torch.manual_seed(11)
    model = tiny_architecture()
    baseline = tiny_architecture(DualSystemSelfAttnArchitecture)
    original = {k: v for k, v in model.state_dict().items() if not k.startswith("interaction_state.")}
    baseline.load_state_dict(original, strict=True)
    data = inputs()
    with torch.no_grad():
        expected = baseline(**data)
        actual = model(**data)
        model.interaction_state.source = "off"
        disabled = model(**data)
    for a, b, c in zip(expected, actual, disabled):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(a, c, rtol=0, atol=0)
    path = str(tmp_path / "checkpoint.safetensors")
    model.save_checkpoint(path)
    restored = tiny_architecture(source="off")
    restored.load_checkpoint(path)
    for a, b in zip(disabled, restored(**data)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_gate_then_reader_gradients_through_frozen_backbones(checkpointed):
    torch.manual_seed(12)
    model = tiny_architecture()
    model.freeze_parameters(["video_backbone", "action_backbone", "proprio_encoder"])
    model.train()
    optimizer = torch.optim.AdamW(model.interaction_state.parameters(), lr=0.02)
    data = inputs()
    data["use_gradient_checkpointing"] = checkpointed
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        video, action = model(**data)
        loss = video.square().mean() + action.square().mean()
        loss.backward()
        for layer in model.interaction_state.layers.values():
            assert layer.gate.grad is not None and layer.gate.grad.isfinite() and layer.gate.grad.abs() > 0
            grad = layer.video_proj.weight.grad
            assert grad is not None and grad.isfinite().all()
            assert (grad.abs().sum() == 0) if step == 0 else (grad.abs().sum() > 0)
        optimizer.step()
    assert all(p.grad is None for p in model.action_backbone.parameters())


def test_checkpoint_recomputation_matches_eager():
    torch.manual_seed(13)
    model = tiny_architecture()
    for layer in model.interaction_state.layers.values():
        layer.gate.data.fill_(0.4)
    other = copy.deepcopy(model)
    # deepcopy copies the plain driver's module references consistently.
    data = inputs()
    a = model(**data)
    b = other(**data, use_gradient_checkpointing=True)
    sum(x.square().mean() for x in a).backward()
    sum(x.square().mean() for x in b).backward()
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)
    for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
        if p.grad is not None:
            assert q.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6, msg=name)


def test_gradient_accumulation_matches_combined_batch():
    torch.manual_seed(23)
    model = tiny_architecture()
    for layer in model.interaction_state.layers.values():
        layer.gate.data.fill_(0.2)
    accumulated = copy.deepcopy(model)
    data = inputs()
    sum(x.square().mean() for x in model(**data, use_gradient_checkpointing=True)).backward()
    for index in range(2):
        microbatch = {key: value[index:index + 1] for key, value in data.items()}
        loss = sum(x.square().mean() for x in accumulated(**microbatch, use_gradient_checkpointing=True)) / 2
        loss.backward()
    for (name, p), (_, q) in zip(model.named_parameters(), accumulated.named_parameters()):
        if p.grad is not None:
            assert q.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6, msg=name)


@pytest.mark.parametrize("source", ["current", "future", "future_action"])
def test_current_prefix_and_state_source_isolation(source):
    torch.manual_seed(14)
    model = tiny_architecture(source=source).eval()
    for layer in model.interaction_state.layers.values():
        layer.gate.data.fill_(0.4)
    data = inputs()
    states = []
    handles = [layer.norm.register_forward_pre_hook(lambda mod, args: states.append(args[0].detach().clone()))
               for layer in model.interaction_state.layers.values()]
    with torch.no_grad():
        a = model(**data)
        recorded = states[:]
        states.clear()
        changed = dict(data)
        changed["latents"] = data["latents"].clone()
        changed["latents"][:, :, 1:] += 10
        changed["noisy_actions"] = data["noisy_actions"] + 10
        b = model(**changed)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(a[0][:, :, :1], b[0][:, :, :1], rtol=0, atol=0)
    if source == "current":
        for x, y in zip(recorded, states):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    else:
        assert any(not torch.allclose(x, y) for x, y in zip(recorded, states))


def test_all_invalid_and_nan_padding_are_inert():
    module = CrossAttention(16, 4)
    query = torch.randn(2, 3, 16)
    source = torch.full((2, 4, 16), float("nan"))
    valid = torch.zeros(2, 4, dtype=torch.bool)
    out = module(query, source, valid)
    assert out.isfinite().all() and torch.count_nonzero(out) == 0
    out.sum().backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in module.parameters())


def test_mask_metadata_and_no_action_branch():
    model = tiny_architecture()
    data = inputs()
    data["_state_video_is_pad"] = torch.ones(2, 2, dtype=torch.bool)
    data["_state_action_is_pad"] = torch.ones(2, 3, 7, dtype=torch.bool)
    video, action = model(**data)
    assert video.isfinite().all() and action.isfinite().all()
    data["noisy_actions"] = None
    video, action = model(**data)
    assert action is None and video.isfinite().all()


def test_source_validation():
    with pytest.raises(ValueError, match="Unknown State source"):
        InteractionStateExpert(video_dim=32, action_dim=32, attention_dim=32, num_layers=30, cfg={"source": "oracle"})
    with pytest.raises(ValueError, match="explicit state_bridge_layers"):
        InteractionStateExpert(video_dim=32, action_dim=32, attention_dim=32, num_layers=2, cfg={})
