"""CPU-only audit of actual historical states, caps and normalized inputs."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwam.dataloader.wuji_real_task import WujiRealTaskDataset
from openwam.model.architectures.utils.rtc_state_history import (
    RTCStateHistoryConfig, rotation_distance_deg, read_history,
)


def quantiles(x):
    return dict(zip(('p50','p90','p99','max'),np.percentile(x,[50,90,99,100]).tolist())) if len(x) else None


def summary(state, action):
    result={}
    for side,base in [('left',0),('right',29)]:
        result[side]={
            'xyz_mm':quantiles(np.linalg.norm(state[:,base:base+3]-action[:,base:base+3],axis=1)*1000),
            'rotation_deg':quantiles(rotation_distance_deg(state[:,base+3:base+9],action[:,base+3:base+9])),
            'hand_max_joint_rad':quantiles(np.max(abs(state[:,base+9:base+29]-action[:,base+9:base+29]),axis=1)),
        }
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='outputs/wuji_rtc_history/preflight/input_audit.json')
    args=p.parse_args()
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]
    cfg=OmegaConf.merge(OmegaConf.load(root/'configs/train_wuji_rtc_ab.yaml'),OmegaConf.load(root/'configs/train_wuji_rtc_history.yaml'))
    d=WujiRealTaskDataset.from_config(cfg.dataloader)
    history_cfg=RTCStateHistoryConfig.from_config(dict(cfg.training.rtc_state_history,enabled=True))
    d.rtc_state_history=history_cfg
    rng=np.random.default_rng(42)
    all_current=[];all_selected=[];all_action=[];all_k=[];all_requested=[];all_prefix=[];all_available=[]
    for ep,row in d._eps_df.iterrows():
        table=d._load_data_table(int(row['data/chunk_index']),int(row['data/file_index']))
        start,end=int(d._ep_valid_start[ep]),int(d._ep_valid_end[ep])
        raw=table.slice(int(d._ep_data_row_offset[ep]),end).to_pandas()
        states=d._read_vector(raw,'observation.state');actions=d._read_vector(raw,'action')
        offsets=start+np.arange(int(d._cum_n_starts[ep+1]-d._cum_n_starts[ep]))*d._window_stride
        now=states[offsets];n=len(now);kmax=history_cfg.max_history_steps
        valid=np.zeros((n,kmax),bool);past=np.repeat(now[:,None],kmax,axis=1)
        tasks=np.asarray(raw['task_index'])
        for j in range(kmax):
            k=j+1;idx=np.maximum(offsets-k,start);past[:,j]=states[idx]
            v=offsets-k>=start
            for step in range(1,k+1):v &= tasks[np.maximum(offsets-step,start)]==tasks[offsets]
            v &= np.isfinite(now).all(1)&np.isfinite(past[:,j]).all(1)
            for base in (0,29):
                if history_cfg.max_translation_m is not None:
                    v &= np.linalg.norm(now[:,base:base+3]-past[:,j,base:base+3],axis=1)<=history_cfg.max_translation_m
                angle=rotation_distance_deg(now[:,base+3:base+9],past[:,j,base+3:base+9])
                v &= np.isfinite(angle)
                if history_cfg.max_rotation_deg is not None:
                    v &= angle<=history_cfg.max_rotation_deg
                if history_cfg.max_joint_rad is not None:
                    v &= np.max(abs(now[:,base+9:base+29]-past[:,j,base+9:base+29]),axis=1)<=history_cfg.max_joint_rad
            valid[:,j]=v
        # Check vectorized population audit against the actual reader implementation.
        for local in [0,n//2,n-1]:
            actual=read_history(d,int(d._cum_n_starts[ep])+local)
            np.testing.assert_array_equal(actual['rtc_history_valid'],valid[local])
        prefix=(rng.random(n)>=.25)&(end-offsets>1)
        requested=prefix&(rng.random(n)<history_cfg.probability)
        requested_k=rng.integers(1,kmax+1,n)
        ks=np.arange(1,kmax+1)[None]
        chosen=np.where(valid&(ks<=requested_k[:,None]),ks,0).max(1)
        chosen=np.where(requested,chosen,0)
        selected=np.where((chosen>0)[:,None],past[np.arange(n),np.maximum(chosen-1,0)],now)
        all_current.append(now);all_selected.append(selected);all_action.append(actions[offsets])
        all_k.append(chosen);all_requested.append(requested);all_prefix.append(prefix);all_available.append(valid)
    current=np.concatenate(all_current);selected=np.concatenate(all_selected);action=np.concatenate(all_action)
    ks=np.concatenate(all_k);requested=np.concatenate(all_requested);prefix=np.concatenate(all_prefix);valid=np.concatenate(all_available)
    changed=ks>0
    lo=np.asarray(d._normalization_stats['q01']);hi=np.asarray(d._normalization_stats['q99'])
    norm_current=d._normalize_array(current);norm_selected=d._normalize_array(selected);norm_action=d._normalize_array(action)
    # Physical equivalents after actual quantile clipping and model bf16 conversion.
    def physical(normalized):
        return (torch.from_numpy(normalized).to(torch.bfloat16).float().numpy()+1)*np.maximum(hi-lo,1e-6)/2+lo
    result={
        'config':vars(history_cfg),'windows':len(current),'episodes':len(d._eps_df),'seed':42,
        'counts':{'prefix':int(prefix.sum()),'requested':int(requested.sum()),'selected':int(changed.sum()),
                  'actually_changed_after_normalization_bf16':int((physical(norm_selected)!=physical(norm_current)).any(1).sum()),
                  'fallback_to_clean':int((requested&~changed).sum()),
                  'k_counts':{str(k):int((ks==k).sum()) for k in range(4)}},
        'selected_fraction_all':float(changed.mean()),
        'selected_fraction_nonempty_prefix':float(changed.sum()/prefix.sum()),
        'candidate_valid_fraction_by_k':valid.mean(0).tolist(),
        'raw_clean_state_to_action0':summary(current,action),
        'raw_augmented_subset_state_to_action0':summary(selected[changed],action[changed]),
        'raw_augmented_subset_added_state_change':summary(selected[changed],current[changed]),
        'model_bf16_clean_state_to_action0':summary(physical(norm_current),physical(norm_action)),
        'model_bf16_augmented_subset_state_to_action0':summary(physical(norm_selected)[changed],physical(norm_action)[changed]),
        'clipped_component_fraction_clean':float(((current<lo)|(current>hi)).mean()),
        'clipped_component_fraction_augmented_subset':float(((selected[changed]<lo)|(selected[changed]>hi)).mean()),
        'alignment':{'old':{'world_size':4,'batch_size':32,'accumulation':4},'new':{'world_size':8,'batch_size':32,'accumulation':2},
                     'effective_batch':512,'nominal_optimizer_updates_per_epoch':math.ceil(len(d)/512),
                     'old_microsteps_per_epoch':math.ceil(math.ceil(len(d)/32)/4),
                     'new_microsteps_per_epoch':math.ceil(math.ceil(len(d)/32)/8)},
    }
    target=root/args.output;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
