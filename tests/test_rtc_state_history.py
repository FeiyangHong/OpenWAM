from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch

from openwam.model.architectures.utils.rtc_state_history import (
    RTCStateHistoryConfig, apply_history, bounded_history, read_history,
    configure_rtc_state_history, collate_history, rotation_distance_deg,
)
from openwam.model.architectures.utils.training_rtc import configure_training_rtc
from tests.test_training_rtc import model_and_data


def raw_state():
    v=np.zeros(58,dtype=np.float32)
    v[[3,7,32,36]]=1
    return v


def test_bounds_symmetric_xyz_rotation_and_all_hand_joints():
    cfg=RTCStateHistoryConfig(enabled=True)
    current=raw_state()
    for base in (0,29):
        past=np.repeat(current[None],4,axis=0)
        past[0,base]=.024
        past[1,base]=.026
        angle=np.radians(4)
        past[2,base+3:base+9]=[np.cos(angle),np.sin(angle),0,-np.sin(angle),np.cos(angle),0]
        past[3,base+28]=.051
        assert bounded_history(current,past,[True]*4,cfg).tolist()==[True,False,False,False]
        assert rotation_distance_deg(past[2,base+3:base+9],current[base+3:base+9])==pytest.approx(4,abs=1e-4)


def test_null_caps_keep_full_history_but_reject_invalid_data():
    cfg=RTCStateHistoryConfig.from_config(dict(enabled=True,max_translation_m=None,max_rotation_deg=None,max_joint_rad=None))
    current=raw_state()
    for base in (0,29):
        past=np.repeat(current[None],6,axis=0)
        past[0,base]=.08
        angle=np.radians(12)
        past[1,base+3:base+9]=[np.cos(angle),np.sin(angle),0,-np.sin(angle),np.cos(angle),0]
        past[2,base+28]=.2
        past[3,base]=np.nan
        past[4,base+3:base+9]=0
        assert bounded_history(current,past,[True]*5+[False],cfg).tolist()==[True,True,True,False,False,False]
        assert bounded_history(current,past[:3],[True]*3,RTCStateHistoryConfig(enabled=True)).tolist()==[False]*3


def test_episode_boundary_task_boundary_fallback_and_real_normalization():
    from openwam.dataloader.wuji_real_task import WujiRealTaskDataset
    d=object.__new__(WujiRealTaskDataset)
    d.rtc_state_history=RTCStateHistoryConfig(enabled=True)
    d.rot6d_convention='column';d._normalize_mode='quantile'
    d._normalization_stats={'q01':np.full(58,-1.),'q99':np.ones(58)}
    d._unify=True;d._unify_dim=80;d.ACTION_DIM=80
    d._unify_state_dst_index=np.array(list(range(9))+list(range(10,30))+list(range(34,43))+list(range(44,64)))
    d._unify_proprio_dim_mask=np.zeros(80,bool);d._unify_proprio_dim_mask[d._unify_state_dst_index]=True
    d._cum_n_starts=np.array([0,5,10]);d._ep_valid_start=np.array([0,0]);d._ep_data_row_offset=np.array([0,0]);d._window_stride=1
    d._eps_df=pd.DataFrame({'data/chunk_index':[0,0],'data/file_index':[0,1]})
    values=np.repeat(raw_state()[None],5,axis=0)
    values[:,0]=np.arange(5)*.005;values[:,29]=np.arange(5)*.004
    values[:,9]=np.arange(5)*.006;values[:,38]=np.arange(5)*.007
    table=pa.Table.from_pydict({'observation.state':values.tolist(),'task_index':[0,0,1,1,1]})
    d._load_data_table=lambda *args:table
    assert not read_history(d,5)['rtc_history_valid'].any()  # second episode start
    h=read_history(d,4)
    assert h['rtc_history_valid'].tolist()==[True,True,False]  # task boundary
    np.testing.assert_array_equal(h['rtc_history_states'][0,d._unify_state_dst_index],values[3])
    assert h['rtc_history_states'][:,~d._unify_proprio_dim_mask].eq(0).all()


def test_prefix_gate_probability_fallback_rng_and_padding():
    x=torch.zeros(4000,80)
    history=torch.stack([x+.01,x+.02,x+.03],1)
    history[:,:,79]=0
    valid=torch.ones(4000,3,dtype=torch.bool)
    valid[:100]=False;valid[100:200,1:]=False
    eligible=torch.arange(4000)%2==0
    cfg=RTCStateHistoryConfig(enabled=True)
    torch.manual_seed(42);rng=torch.get_rng_state()
    y,stats=apply_history(x,eligible,history,valid,cfg)
    assert torch.equal(rng,torch.get_rng_state())
    z,_=apply_history(x,eligible,history,valid,cfg)
    assert torch.equal(y,z) and y[~eligible].eq(0).all() and y[:100].eq(0).all()
    assert y[100:200].max()<=.01 and y[:,79].eq(0).all()
    assert .12<stats['selected_fraction']<.18
    disabled,_=apply_history(x,eligible,None,None,RTCStateHistoryConfig())
    assert disabled is x


def test_disabled_loss_gradients_rng_and_checkpoint_keys_unchanged():
    model,data=model_and_data()
    configure_training_rtc(model,{'enabled':True,'max_prefix_steps':2})
    keys=tuple(model.state_dict())
    torch.manual_seed(91);before=model.compute_loss(**data,rtc_prefix_lengths=torch.tensor([2,1]))
    rng=torch.get_rng_state();before['loss'].backward()
    grads={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    configure_rtc_state_history(model,{'enabled':False})
    torch.manual_seed(91);after=model.compute_loss(**data,rtc_prefix_lengths=torch.tensor([2,1]))
    assert torch.equal(rng,torch.get_rng_state())
    after['loss'].backward()
    for k in before:assert torch.equal(before[k],after[k])
    for k,p in model.named_parameters():
        if k in grads:assert torch.equal(grads[k],p.grad)
    assert keys==tuple(model.state_dict())


@pytest.mark.parametrize('lengths', [[0,0],[2,0],[2,1]])
def test_real_loss_receives_history_only_for_prefix_and_preserves_targets(lengths):
    model,data=model_and_data()
    configure_training_rtc(model,{'enabled':True,'max_prefix_steps':2})
    model.rtc_state_history=RTCStateHistoryConfig(enabled=True,probability=1)
    original=data['proprio'].clone()
    history=original.reshape(2,-1)[:,None].repeat(1,3,1)+.125
    valid=torch.ones(2,3,dtype=torch.bool)
    seen={}
    def capture(module,args,kwargs):
        seen['state']=kwargs['proprio'].clone();seen['actions']=args[0].clone()
        assert 'rtc_history_states' not in kwargs and 'rtc_history_valid' not in kwargs
    hook=model.register_forward_pre_hook(capture,with_kwargs=True)
    actions=data['actions'].clone()
    loss=model.compute_loss(**data,rtc_prefix_lengths=torch.tensor(lengths),rtc_history_states=history,rtc_history_valid=valid)
    hook.remove();loss['loss'].backward()
    assert loss['loss'].isfinite() and torch.equal(data['actions'],actions) and torch.equal(data['proprio'],original)
    for i,k in enumerate(lengths):
        torch.testing.assert_close(seen['state'][i],original[i]+(.125 if k else 0))
        assert torch.equal(seen['actions'][i,:k],actions[i,:k])


def test_resume_and_config_guards():
    m=SimpleNamespace()
    configure_rtc_state_history(m,{})
    with pytest.raises(ValueError,match='Resume'):
        configure_rtc_state_history(m,{'enabled':True},resume=True)
    with pytest.raises(ValueError,match='requires training.rtc'):
        configure_rtc_state_history(m,{'enabled':True})
    for cfg in [{'enabled':'false'},{'probability':-1},{'max_history_steps':4},{'max_joint_rad':float('nan')}]:
        with pytest.raises(ValueError):RTCStateHistoryConfig.from_config(cfg)
    with pytest.raises(ValueError,match='Mixed'):
        collate_history([{'rtc_history_states':torch.zeros(3,80)},{}])


def test_prepare_inputs_carries_history_without_changing_existing_fields():
    model,data=model_and_data()
    model.preprocess=lambda **kwargs:{'input_latents':data['input_latents']}
    sample={'video':[object()], 'prompt':'test', 'action':data['actions'][0],
            'proprio':data['proprio'][0], 'proprio_mask':torch.ones(1,dtype=torch.bool)}
    clean=model.prepare_inputs([dict(sample)])
    model.rtc_state_history=RTCStateHistoryConfig(enabled=True)
    hist=dict(sample,rtc_history_states=torch.zeros(3,data['proprio'].shape[-1]),
              rtc_history_valid=torch.ones(3,dtype=torch.bool))
    actual=model.prepare_inputs([hist])
    assert actual['rtc_history_states'].shape==(1,3,data['proprio'].shape[-1])
    for key in clean:
        if isinstance(clean[key],torch.Tensor): assert torch.equal(clean[key],actual[key])


def test_real_dataset_enabled_adds_only_history_and_disabled_keeps_rng():
    from pathlib import Path
    from omegaconf import OmegaConf
    from openwam.dataloader.wuji_real_task import WujiRealTaskDataset
    root=Path(__file__).resolve().parents[1]
    cfg=OmegaConf.load(root/'configs/train_wuji_rtc_ab.yaml')
    if not Path(cfg.dataloader.dataset_dir).exists():pytest.skip('H20 training dataset required')
    d=WujiRealTaskDataset.from_config(cfg.dataloader)
    idx=100
    torch.manual_seed(7);before=d[idx];rng=torch.get_rng_state()
    d.rtc_state_history=RTCStateHistoryConfig(enabled=True)
    torch.manual_seed(7);after=d[idx]
    assert torch.equal(rng,torch.get_rng_state())
    assert set(after)-set(before)=={'rtc_history_states','rtc_history_valid'}
    for key in ('action','action_mask','proprio','proprio_mask','video_mask'):
        assert torch.equal(before[key],after[key])
    assert before['prompt']==after['prompt']
    for a,b in zip(before['video'],after['video']):np.testing.assert_array_equal(a,b)
    d.rtc_state_history=RTCStateHistoryConfig(enabled=False)
    assert set(d[idx])==set(before)
