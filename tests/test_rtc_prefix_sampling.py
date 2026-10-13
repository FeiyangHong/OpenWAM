"""Optional prefix PMF: original RNG path, distribution, padding, resume and wiring."""
import pytest
import torch

from openwam.model.architectures.utils.rtc_prefix_sampling import (
    RTCPrefixSamplingConfig, configure_rtc_prefix_sampling, sample_prefix_lengths,
)
from openwam.model.architectures.utils.training_rtc import (
    TrainingRTCConfig, make_training_prefix, configure_training_rtc,
)
from openwam.model.architectures.utils.rtc_seam_loss import configure_rtc_seam_loss
from tests.test_training_rtc import model_and_data


@pytest.mark.parametrize('cfg', [None, {}, {'enabled': False, 'probabilities': [0., 0., 1.]}])
def test_disabled_preserves_full_loss_and_rng(cfg):
    model, data = model_and_data()
    configure_training_rtc(model, {'enabled': True, 'max_prefix_steps': 2})
    configure_rtc_seam_loss(model, {'enabled': True, 'steps': 1})
    torch.manual_seed(109)
    before = model.compute_loss(**data)
    rng = torch.get_rng_state()
    configure_rtc_prefix_sampling(model, cfg)
    torch.manual_seed(109)
    after = model.compute_loss(**data)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_empirical_pmf_and_rng_restore():
    probs = [.1] + [.1 / 7] * 7 + [.15, .15, .35, .1, .05]
    cfg = RTCPrefixSamplingConfig.from_config({'enabled': True, 'probabilities': probs})
    torch.manual_seed(113)
    state = torch.get_rng_state()
    sampled = sample_prefix_lengths(200000, 12, cfg)
    torch.set_rng_state(state)
    assert torch.equal(sampled, sample_prefix_lengths(200000, 12, cfg))
    actual = torch.bincount(sampled, minlength=13).double() / len(sampled)
    torch.testing.assert_close(actual, torch.tensor(probs, dtype=torch.float64), atol=.004, rtol=0)


@pytest.mark.parametrize('dim_mask', [False, True])
def test_valid_window_limits_and_clean_prefix(dim_mask):
    actions = torch.randn(4, 6, 5)
    noisy = torch.randn_like(actions)
    pad = torch.zeros_like(actions, dtype=torch.bool) if dim_mask else torch.zeros(4, 6, dtype=torch.bool)
    pad[1, 3:] = True
    pad[2, 1:] = True
    pad[3] = True
    if dim_mask:
        pad[..., -1] = True
    original = pad.clone()
    cfg = RTCPrefixSamplingConfig(True, (0, 0, 0, 0, 1))
    # PMF replaces p0, even when the legacy zero-prefix probability is 1.
    x, t, mask = make_training_prefix(actions, noisy, torch.ones(4), pad,
        TrainingRTCConfig(True, 4, 1), sampling_config=cfg)
    added = mask & ~pad
    if dim_mask:
        added = added.any(-1)
    assert added.sum(-1).tolist() == [4, 2, 0, 0]
    assert torch.equal(x[0, :4], actions[0, :4])
    assert torch.equal(x[0, 4:], noisy[0, 4:])
    assert t[0, :4].eq(0).all() and t[0, 4:].eq(1).all()
    assert torch.equal(original, pad)


def test_explicit_lengths_bypass_pmf_and_zero_mass_keeps_scalar_path():
    actions, noisy = torch.randn(2, 6, 4), torch.randn(2, 6, 4)
    t = torch.ones(2)
    rtc = TrainingRTCConfig(True, 4)
    pmf = RTCPrefixSamplingConfig(True, (0, 0, 0, 0, 1))
    rng = torch.get_rng_state()
    a = make_training_prefix(actions, noisy, t, None, rtc, torch.tensor([1, 2]), sampling_config=pmf)
    b = make_training_prefix(actions, noisy, t, None, rtc, torch.tensor([1, 2]))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert torch.equal(rng, torch.get_rng_state())
    x, times, pad = make_training_prefix(actions, noisy, t, None, rtc,
        sampling_config=RTCPrefixSamplingConfig(True, (1, 0, 0, 0, 0)))
    assert x is noisy and times is t and pad is None
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('cfg', [
    {'enabled': True}, {'enabled': 1}, {'probabilities': []}, {'probabilities': [.1, .2]},
    {'probabilities': [-.1, 1.1]}, {'probabilities': [True, 0.]},
    {'probabilities': [float('nan'), 1.]}, {'probabilities': [float('inf')]},
])
def test_bad_config(cfg):
    with pytest.raises(ValueError):
        RTCPrefixSamplingConfig.from_config(cfg)


def test_resume_contract_and_architecture_guard():
    model, _ = model_and_data()
    configure_training_rtc(model, {'enabled': True, 'max_prefix_steps': 2})
    cfg = {'enabled': True, 'probabilities': [.1, .2, .7]}
    keys = tuple(model.state_dict())
    configure_rtc_prefix_sampling(model, cfg)
    assert keys == tuple(model.state_dict())
    configure_rtc_prefix_sampling(model, cfg, saved_cfg=cfg, resume=True)
    for saved in [None, {'enabled': True, 'probabilities': [.1, .3, .6]}]:
        with pytest.raises(ValueError, match='Resume'):
            configure_rtc_prefix_sampling(model, cfg, saved_cfg=saved, resume=True)
    with pytest.raises(ValueError, match='exactly'):
        configure_rtc_prefix_sampling(model, {'enabled': True, 'probabilities': [0, 1]})
    configure_training_rtc(model, {'enabled': False})
    with pytest.raises(ValueError, match='requires TrainingRTC'):
        configure_rtc_prefix_sampling(model, cfg)


def test_actual_compute_loss_uses_sampled_prefix_and_existing_seam_weights():
    model, data = model_and_data()
    configure_training_rtc(model, {'enabled': True, 'max_prefix_steps': 2})
    configure_rtc_prefix_sampling(model, {'enabled': True, 'probabilities': [0., 1., 0.]})
    configure_rtc_seam_loss(model, {'enabled': True, 'steps': 1})
    captured = {}
    original = model._compute_action_loss
    def capture(pred, target, timestep_ids, scheduler, inputs, device):
        captured.update(inputs)
        return original(pred, target, timestep_ids, scheduler, inputs, device)
    model._compute_action_loss = capture
    loss = model.compute_loss(**data)['loss']
    assert torch.equal(captured['rtc_seam_weights'], torch.tensor([[1., 2., 1.], [1., 2., 1.]]))
    loss.backward()
    grads = [p.grad for p in model.action_backbone.parameters() if p.grad is not None]
    assert grads and all(g.isfinite().all() for g in grads) and any(g.abs().sum() > 0 for g in grads)
