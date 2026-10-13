"""Physical-unit bimanual augmentation and real TrainingRTC loss regression."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openwam.model.architectures.utils.rtc_state_augmentation import (
    RTCStateAugmentationConfig, RTCStateAugmenter, build_rtc_state_augmenter,
    configure_rtc_state_augmentation,
)
from openwam.model.architectures.utils.training_rtc import configure_training_rtc
from tests.test_training_rtc import model_and_data


def augmenter(config=None):
    return RTCStateAugmenter(config or RTCStateAugmentationConfig(True, 1.0, .005),
                             [[0, 1, 2], [34, 35, 36]], [[2., 3., 4.], [5., 6., 7.]], 80)


@pytest.mark.parametrize('shape', [(128, 80), (128, 1, 80)])
def test_shared_physical_translation_preserves_every_other_dimension(shape):
    aug=augmenter();x=torch.zeros(shape);x[..., 50]=.33;original=x.clone()
    eligible=torch.arange(128)%2==0
    torch.manual_seed(17);y=aug(x,eligible)
    delta=(y-x).reshape(128,80)
    left=delta[:,[0,1,2]]/aug.scale[0]
    right=delta[:,[34,35,36]]/aug.scale[1]
    torch.testing.assert_close(left,right,atol=1e-9,rtol=1e-6)
    assert left.norm(dim=-1).max()<=.00500001
    assert left[eligible].norm(dim=-1).min()>0
    assert torch.equal(y[~eligible],x[~eligible]) and torch.equal(x,original)
    other=[i for i in range(80) if i not in [0,1,2,34,35,36]]
    assert delta[:,other].eq(0).all()


@pytest.mark.parametrize('config,eligible', [
    (RTCStateAugmentationConfig(False), [True,True]),
    (RTCStateAugmentationConfig(True,0), [True,True]),
    (RTCStateAugmentationConfig(True,1,0), [True,True]),
    (RTCStateAugmentationConfig(True,1), [False,False]),
])
def test_inactive_preserves_tensor_identity_and_rng(config,eligible):
    x=torch.randn(2,80);rng=torch.get_rng_state()
    assert augmenter(config)(x,eligible) is x
    assert torch.equal(rng,torch.get_rng_state())


def test_probability_masks_and_rng_replay():
    aug=augmenter(RTCStateAugmentationConfig(True,.1,.005))
    x=torch.zeros(2000,1,80);mask=torch.ones_like(x,dtype=torch.bool)
    mask[0:10,0,34]=False
    torch.manual_seed(42);rng=torch.get_rng_state()
    a=aug(x,torch.ones(2000,dtype=torch.bool),mask)
    torch.set_rng_state(rng);b=aug(x,torch.ones(2000,dtype=torch.bool),mask)
    assert torch.equal(a,b) and a[:10].eq(0).all()
    changed=a.reshape(2000,80).ne(0).any(-1)
    assert 140 < changed.sum() < 260


def test_quantile_boundaries_and_invalid_config():
    x=torch.full((4,80),1.0);x[3]=2.0
    y=augmenter()(x,[True,True,True,False])
    assert y[:3,[0,1,2,34,35,36]].abs().max()<=1
    assert torch.equal(y[3],x[3])
    for cfg in [{'probability':-1},{'probability':float('nan')},{'max_translation_m':-1},{'enabled':'false'}]:
        with pytest.raises(ValueError):RTCStateAugmentationConfig.from_config(cfg)


def test_reader_state_scatter_map_and_resume_guard():
    from openwam.dataloader.wuji_real_task import WujiRealTaskDataset
    d=object.__new__(WujiRealTaskDataset)
    d.rot6d_convention='column';d._normalize_mode='quantile';d._unify=True;d._unify_dim=80
    d._normalization_stats={'q01':np.zeros(58),'q99':np.linspace(1,2,58)}
    # A different state scatter must be honored, not the action scatter.
    d._unify_dst_index=np.arange(58);d._unify_state_dst_index=np.arange(58)+10
    aug=build_rtc_state_augmenter(d,RTCStateAugmentationConfig(True,1,.005))
    assert aug.xyz_indices.tolist()==[[10,11,12],[39,40,41]]
    model=SimpleNamespace(training_rtc=SimpleNamespace(enabled=True))
    configure_rtc_state_augmentation(model,{},resume=True)
    with pytest.raises(ValueError,match='Resume'):
        configure_rtc_state_augmentation(model,{'enabled':True},dataset=d,resume=True)
    configure_rtc_state_augmentation(model,{'enabled':True},dataset=d,saved_cfg={'enabled':True},resume=True)
    with pytest.raises(ValueError,match='requires training.rtc'):
        configure_rtc_state_augmentation(SimpleNamespace(),{'enabled':True},dataset=d)


def test_disabled_matches_existing_rtc_loss_rng_and_gradients():
    model,data=model_and_data()
    configure_training_rtc(model,{'enabled':True,'max_prefix_steps':2})
    lengths=torch.tensor([2,1])
    keys=tuple(model.state_dict())
    torch.manual_seed(91);before=model.compute_loss(**data,rtc_prefix_lengths=lengths)
    rng=torch.get_rng_state();before['loss'].backward()
    grads={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    configure_rtc_state_augmentation(model,{'enabled':False})
    torch.manual_seed(91);after=model.compute_loss(**data,rtc_prefix_lengths=lengths)
    assert torch.equal(rng,torch.get_rng_state())
    after['loss'].backward()
    for k in before:assert torch.equal(before[k],after[k])
    for k,p in model.named_parameters():
        if k in grads:assert torch.equal(grads[k],p.grad)
    assert keys==tuple(model.state_dict())


@pytest.mark.parametrize('checkpointed',[False,True])
def test_enabled_real_forward_backward_keeps_prefix_targets_and_zero_prefix_samples(checkpointed):
    model,data=model_and_data();data['proprio']=torch.zeros(2,7)
    data['action_is_pad']=torch.zeros_like(data['actions'],dtype=torch.bool)
    data['action_is_pad'][1,2]=True
    data['use_gradient_checkpointing']=checkpointed
    original={k:v.clone() for k,v in data.items() if torch.is_tensor(v)}
    configure_training_rtc(model,{'enabled':True,'max_prefix_steps':2})
    model.rtc_state_augmentation=RTCStateAugmentationConfig(True,1,.005)
    model.rtc_state_augmenter=RTCStateAugmenter(model.rtc_state_augmentation,[[0,1,2],[3,4,5]],[[2]*3,[4]*3],7)
    captured={}
    def capture(module,args,kwargs):
        captured['actions']=args[0].detach().clone();captured['proprio']=kwargs['proprio'].detach().clone()
    handle=model.register_forward_pre_hook(capture,with_kwargs=True)
    torch.manual_seed(72)
    result=model.compute_loss(**data,rtc_prefix_lengths=torch.tensor([2,0]))
    handle.remove()
    assert torch.equal(captured['actions'][0,:2],data['actions'][0,:2])
    assert captured['proprio'][0,:6].ne(0).any()
    assert captured['proprio'][1].eq(0).all()
    assert captured['proprio'][:,6].eq(0).all()
    for k,v in original.items():assert torch.equal(data[k],v)
    assert result['loss'].isfinite();result['loss'].backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(g.isfinite().all() for g in grads)


def test_all_zero_prefix_without_padding_keeps_loss_and_rng():
    model, data = model_and_data()
    data.pop('action_is_pad', None)
    configure_training_rtc(model, {'enabled': True, 'max_prefix_steps': 2})
    lengths = torch.zeros(2, dtype=torch.long)
    torch.manual_seed(13)
    before = model.compute_loss(**data, rtc_prefix_lengths=lengths)
    rng = torch.get_rng_state()
    model.rtc_state_augmentation = RTCStateAugmentationConfig(True, 1, .005)
    model.rtc_state_augmenter = RTCStateAugmenter(
        model.rtc_state_augmentation, [[0, 1, 2], [3, 4, 5]], [[2]*3, [4]*3], 7)
    torch.manual_seed(13)
    after = model.compute_loss(**data, rtc_prefix_lengths=lengths)
    for key in before:
        assert torch.equal(before[key], after[key])
    assert torch.equal(rng, torch.get_rng_state())


def test_enabled_augmentation_does_not_enter_generation():
    model, _ = model_and_data()
    configure_training_rtc(model, {'enabled': True, 'max_prefix_steps': 2})
    model.video_backbone.preprocess_input_for_inference = lambda **kw: {
        'latents': torch.ones(1, 4, 3, 4, 4), 'context': torch.ones(1, 4, 32),
        'uncond_context': torch.zeros(1, 4, 32),
    }
    model.forward = lambda actions, action_timestep, **kw: (
        torch.ones_like(kw['latents']), torch.full_like(actions, 9.))
    prefix = torch.zeros(2, 7)

    def generate():
        torch.manual_seed(51)
        result = model.generate(
            [(1000, 1000), (500, 500), (0, 0)], 'test', action_num_frames=6,
            rtc_prefix=prefix, proprio=torch.zeros(1, 7), decode_video=False,
            active_action_mask=torch.tensor([True] * 6 + [False]),
            cfg_scale=1, cfg_merge=False)
        return result['actions'], torch.get_rng_state()

    before, rng = generate()
    model.rtc_state_augmentation = RTCStateAugmentationConfig(True, 1, .005)

    def forbidden(*args, **kwargs):
        raise AssertionError('Inference called the training augmenter')

    model.rtc_state_augmenter = forbidden
    after, after_rng = generate()
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(after[:2], prefix.numpy())
    assert torch.equal(rng, after_rng)
