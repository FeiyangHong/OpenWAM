import copy

import pytest
import torch

from openwam.model.architectures.statewam.state_expert import InteractionStateExpert
from tests.statewam.test_statewam import inputs, tiny_architecture
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from scripts.statewam.r2.matrix import apply_parameter_policy


def controlled_model(source):
    model = tiny_architecture(source=source)
    model.cfg["interaction_state"]["capacity_control"] = "current_resample"
    model._init_state_expert()
    for layer in model.interaction_state.layers.values():
        layer.gate.data.fill_(0.3)
    return model


@pytest.mark.parametrize("source", ["current", "future"])
def test_capacity_control_activates_same_parameters_as_e3(source):
    torch.manual_seed(2026)
    control = controlled_model(source)
    e3 = tiny_architecture(source="future_action")
    e3.load_state_dict(control.state_dict(), strict=True)
    data = inputs()
    active = []
    for model in (control, e3):
        sum(x.square().mean() for x in model(**data, use_gradient_checkpointing=True)).backward()
        parameters = {name: p for name, p in model.interaction_state.named_parameters()}
        assert all(p.grad is not None and p.grad.isfinite().all() for p in parameters.values())
        for name, p in parameters.items():
            if "action_proj" in name or "read_action" in name:
                assert p.grad.abs().sum() > 0, name
        active.append({name: p.numel() for name, p in parameters.items() if p.grad is not None})
    assert active[0] == active[1]


@pytest.mark.parametrize("source", ["current", "future"])
def test_control_branch_does_not_read_future_actions_or_action_mask(source):
    model = controlled_model(source).eval()
    data = inputs()
    captured = []
    handles = [layer.action_proj.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
               for layer in model.interaction_state.layers.values()]
    with torch.no_grad():
        model(**data)
        original = captured[:]
        captured.clear()
        changed = copy.deepcopy(data)
        changed["latents"][:, :, 1:] += 100
        changed["noisy_actions"] += 100
        changed["_state_action_is_pad"] = torch.ones(2, 3, 7, dtype=torch.bool)
        model(**changed)
    for handle in handles:
        handle.remove()
    for a, b in zip(original, captured):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_current_control_entire_state_is_isolated_and_current_sensitive():
    model = controlled_model("current").eval()
    data = inputs()
    captured = []
    handles = [layer.norm.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
               for layer in model.interaction_state.layers.values()]
    with torch.no_grad():
        model(**data)
        original = captured[:]
        captured.clear()
        changed = copy.deepcopy(data)
        changed["latents"][:, :, 1:] += 100
        changed["noisy_actions"] += 100
        model(**changed)
        for a, b in zip(original, captured):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        captured.clear()
        changed["latents"][:, :, :1] += 1
        model(**changed)
        assert any(not torch.allclose(a, b) for a, b in zip(original, captured))
    for handle in handles:
        handle.remove()


def test_pooling_masks_nan_and_has_no_trainable_parameters():
    video = torch.randn(2, 7, 12, requires_grad=True)
    valid = torch.tensor([[True, False, True, True, False, False, True], [False] * 7])
    masked = video.masked_fill(~valid[..., None], float("nan"))
    result, result_valid = InteractionStateExpert.current_capacity_source(masked, valid, tokens=3, dim=5)
    assert result.shape == (2, 3, 5) and result.isfinite().all()
    assert not result_valid[1].any() and torch.count_nonzero(result[1]) == 0
    result.sum().backward()
    assert torch.count_nonzero(video.grad[~valid]) == 0


@pytest.mark.parametrize("shape,target", [((2, 7, 12), (3, 5)), ((1, 120, 3072), (32, 1024)), ((2, 3, 4), (5, 7))])
def test_deterministic_pooling_matches_adaptive_bins_and_gradients(shape, target):
    video = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    valid = torch.rand(shape[:2]) > 0.3
    masked = video.masked_fill(~valid[..., None], 0)
    reference = torch.nn.functional.adaptive_avg_pool2d(masked.unsqueeze(1), target).squeeze(1)
    weights = torch.nn.functional.adaptive_avg_pool1d(valid.double().unsqueeze(1), target[0]).transpose(1, 2)
    reference = reference / weights.clamp_min(torch.finfo(torch.float64).tiny)
    actual, actual_valid = InteractionStateExpert.current_capacity_source(video, valid, tokens=target[0], dim=target[1])
    torch.testing.assert_close(reference, actual, rtol=1e-12, atol=1e-12)
    assert torch.equal(actual_valid, weights.squeeze(-1) > 0)
    upstream = torch.randn_like(actual)
    a = torch.autograd.grad(reference, video, upstream)[0]
    b = torch.autograd.grad(actual, video, upstream)[0]
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA deterministic pooling requires a GPU")
def test_cuda_pooling_backward_is_strictly_deterministic(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    enabled, warn_only = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        video = torch.randn(2, 120, 3072, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        valid = torch.rand(2, 120, device="cuda") > 0.3
        outputs, gradients = [], []
        for _ in range(2):
            output, _ = InteractionStateExpert.current_capacity_source(video, valid, tokens=32, dim=1024)
            outputs.append(output)
            gradients.append(torch.autograd.grad(output.float().square().sum(), video)[0])
        assert torch.equal(outputs[0], outputs[1]) and torch.equal(gradients[0], gradients[1])
        assert gradients[0].isfinite().all() and gradients[0].abs().sum() > 0
        assert torch.count_nonzero(gradients[0][~valid]) == 0
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_capacity_control_validation():
    with pytest.raises(ValueError, match="current-only control"):
        InteractionStateExpert(video_dim=32, action_dim=32, attention_dim=32, num_layers=30,
                               cfg={"source": "future_action", "capacity_control": "current_resample"})


@pytest.mark.parametrize("group", ["E0", "E1", "E2", "E3"])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_matrix_world_freeze_preserves_input_gradients(group, checkpointed):
    torch.manual_seed(17)
    if group == "E0":
        model = tiny_architecture(DualSystemSelfAttnArchitecture)
    elif group == "E3":
        model = tiny_architecture()
    else:
        model = controlled_model("current" if group == "E1" else "future")
    counts = apply_parameter_policy(model)
    assert counts["trainable"] < counts["total"]
    data = inputs()
    data["latents"].requires_grad_(True)
    outputs = model(**data, use_gradient_checkpointing=checkpointed)
    sum(value.square().mean() for value in outputs).backward()
    gradient = data["latents"].grad
    assert gradient is not None and gradient.isfinite().all() and gradient.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in model.video_backbone.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.action_backbone.parameters())


@pytest.mark.parametrize("path", ["video_backbone", "video_backbone.dit"])
def test_matrix_rejects_world_no_grad_wrapper(path):
    model = tiny_architecture()
    model.freeze_modules([path])
    with pytest.raises(ValueError, match="no_grad wrappers"):
        apply_parameter_policy(model)
