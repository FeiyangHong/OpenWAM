import copy

import pytest
import torch

from openwam.model.architectures.statewam.state_expert import InteractionStateExpert
from tests.statewam.test_statewam import inputs, tiny_architecture


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


def test_capacity_control_validation():
    with pytest.raises(ValueError, match="current-only control"):
        InteractionStateExpert(video_dim=32, action_dim=32, attention_dim=32, num_layers=30,
                               cfg={"source": "future_action", "capacity_control": "current_resample"})
