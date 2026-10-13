"""RTC seam weighting: indexing, masks, normalization, gradients and real loss path."""
import pytest
import torch

from openwam.model.architectures.utils.rtc_seam_loss import (
    RTCSeamLossConfig, configure_rtc_seam_loss, make_seam_weights, weighted_action_mean,
)
from openwam.model.architectures.utils.training_rtc import TrainingRTCConfig, make_training_prefix, configure_training_rtc
from tests.test_training_rtc import model_and_data


@pytest.mark.parametrize('dim_mask', [False, True])
def test_variable_prefix_mask_and_tail(dim_mask):
    actions = torch.zeros(4, 8, 6)
    pad = torch.zeros_like(actions, dtype=torch.bool) if dim_mask else torch.zeros(4, 8, dtype=torch.bool)
    pad[2, 6:] = True
    if dim_mask:
        pad[..., -1] = True
    before = pad.clone()
    _, _, rtc_pad = make_training_prefix(actions, actions + 1, torch.ones(4), pad,
                                        TrainingRTCConfig(True, 7), torch.tensor([0, 2, 4, 7]))
    weights = make_seam_weights(actions, pad, rtc_pad, RTCSeamLossConfig(True, 3, 2))
    expected = torch.ones(4, 8)
    expected[1, 2:5] = 2
    expected[2, 4:7] = 2  # t=6 padded: reducer must exclude it.
    expected[3, 7] = 2
    assert torch.equal(weights, expected)
    assert torch.equal(pad, before)
    pred = torch.ones_like(actions, requires_grad=True)
    loss = weighted_action_mean(pred.square(), rtc_pad, weights)
    assert torch.equal(loss, torch.ones(4))  # No global loss scale inflation.
    loss.sum().backward()
    expanded_pad = rtc_pad if dim_mask else rtc_pad[..., None].expand_as(actions)
    assert pred.grad[expanded_pad].eq(0).all()
    assert pred.grad[1, 2, 0] == 2 * pred.grad[1, 5, 0]


def test_numerical_weighted_mean_and_all_padding():
    error = torch.tensor([[[999., 999.], [1., 999.], [4., 9.], [16., 25.]],
                          [[7., 7.], [7., 7.], [7., 7.], [7., 7.]]])
    pad = torch.zeros_like(error, dtype=torch.bool)
    pad[0, 0] = True
    pad[0, 1, 1] = True
    pad[1] = True
    actual = weighted_action_mean(error, pad, torch.tensor([[1., 2., 1., 1.], [1., 2., 1., 1.]]))
    torch.testing.assert_close(actual, torch.tensor([(2 + 4 + 9 + 16 + 25) / 6, 0.]))


@pytest.mark.parametrize('cfg', [{'steps': 0}, {'steps': True}, {'weight': .9}, {'weight': float('nan')}, {'weight': float('inf')}, {'weight': True}, {'enabled': 1}])
def test_bad_config(cfg):
    with pytest.raises(ValueError):
        RTCSeamLossConfig.from_config(cfg)


@pytest.mark.parametrize('cfg', [None, {}, {'enabled': True, 'weight': 1.0}])
def test_original_path_exact_loss_and_rng(cfg):
    model, data = model_and_data()
    configure_training_rtc(model, {'enabled': True, 'max_prefix_steps': 2})
    torch.manual_seed(71)
    before = model.compute_loss(**data, rtc_prefix_lengths=torch.tensor([1, 0]))
    rng = torch.get_rng_state()
    configure_rtc_seam_loss(model, cfg)
    torch.manual_seed(71)
    after = model.compute_loss(**data, rtc_prefix_lengths=torch.tensor([1, 0]))
    assert torch.equal(rng, torch.get_rng_state())
    for key in before:
        assert torch.equal(before[key], after[key])


def test_enabled_integration_no_change_to_forward_video_or_rng():
    model, data = model_and_data()
    configure_training_rtc(model, {'enabled': True, 'max_prefix_steps': 2})
    captured = []
    original = model._compute_action_loss
    def capture(pred, target, timestep_ids, scheduler, inputs, device):
        captured.append((pred.detach().clone(), target.detach().clone(), inputs.copy()))
        return original(pred, target, timestep_ids, scheduler, inputs, device)
    model._compute_action_loss = capture
    torch.manual_seed(72)
    baseline = model.compute_loss(**data, rtc_prefix_lengths=torch.tensor([1, 0]))
    rng = torch.get_rng_state()
    configure_rtc_seam_loss(model, {'enabled': True, 'steps': 1, 'weight': 2.0})
    torch.manual_seed(72)
    weighted = model.compute_loss(**data, rtc_prefix_lengths=torch.tensor([1, 0]))
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(baseline['loss_video'], weighted['loss_video'])
    assert torch.equal(captured[0][0], captured[1][0])
    assert torch.equal(captured[0][1], captured[1][1])
    assert torch.equal(captured[1][2]['rtc_seam_weights'], torch.tensor([[1., 2., 1.], [1., 1., 1.]]))
    assert not torch.equal(baseline['loss_action'], weighted['loss_action'])
    weighted['loss'].backward()
    grads = [p.grad for p in model.action_backbone.parameters() if p.grad is not None]
    assert grads and all(g.isfinite().all() for g in grads) and any(g.abs().sum() > 0 for g in grads)
    # All-zero-prefix warmup stays on the exact old reduction, even when enabled.
    captured.clear()
    model.compute_loss(**data, rtc_prefix_lengths=torch.tensor([0, 0]))
    assert 'rtc_seam_weights' not in captured[0][2]


def test_resume_guard_and_no_trainable_parameters():
    model, _ = model_and_data()
    configure_training_rtc(model, {'enabled': True, 'max_prefix_steps': 2})
    keys = tuple(model.state_dict())
    configure_rtc_seam_loss(model, {'enabled': True})
    assert tuple(model.state_dict()) == keys
    with pytest.raises(ValueError, match='Resume'):
        configure_rtc_seam_loss(model, {'enabled': True}, saved_cfg=None, resume=True)
    configure_rtc_seam_loss(model, {'enabled': True}, saved_cfg={'enabled': True}, resume=True)
    configure_training_rtc(model, {'enabled': False})
    with pytest.raises(ValueError, match='requires TrainingRTC'):
        configure_rtc_seam_loss(model, {'enabled': True})
