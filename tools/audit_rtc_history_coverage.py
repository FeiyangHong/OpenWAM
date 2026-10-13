"""Read-only CPU comparison of bounded/unbounded history across motion strata.

Speed/hand phases are COMMAND trajectory proxies, not physical speed/contact labels.
Training windows retain all 33 frames; all variants use the same prefix and state.
Uniform k=1,2,3 is enumerated exactly; no model inference or training is run.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.ndimage import maximum_filter1d, minimum_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openwam.dataloader.wuji_real_task import WujiRealTaskDataset
from openwam.model.architectures.utils.rtc_state_history import rotation_distance_deg, read_history, RTCStateHistoryConfig

BINS = np.array([0, 10, 30, 60, 100, 200, np.inf])


def q(x, weights=None):
    x=np.asarray(x).reshape(-1)
    if not len(x):return None
    if weights is None:
        a=np.percentile(x,[50,90,99,100])
    else:
        weights=np.asarray(weights).reshape(-1);m=weights>0;x=x[m];weights=weights[m]
        if not len(x):return None
        order=np.argsort(x);x=x[order];weights=weights[order]
        cdf=(np.cumsum(weights)-.5*weights)/weights.sum()
        a=np.interp([.5,.9,.99,1],cdf,x)
    return dict(zip(['p50','p90','p99','max'],a.tolist()))


def trajectory(prefix):
    delta=np.diff(prefix[:,:,29:32],axis=1)
    speed=np.linalg.norm(delta,axis=-1).mean(1)*30000
    span=np.linalg.norm(np.ptp(prefix[:,:,29:32],axis=1),axis=1)*1000
    hand_span=np.ptp(prefix[:,:,38:58],axis=1).max(1)
    speedbin=np.searchsorted(BINS[1:],speed,side='right')
    handactive=hand_span>.03
    stratum=speedbin*2+handactive
    return {'speed':speed,'span':span,'hand_span':hand_span,'speedbin':speedbin,'stratum':stratum,
            'masks':{'all':np.ones(len(prefix),bool),'arm_moving':span>5,
                     'arm_quiet':span<=2,'hand_changing':handactive,
                     'arm_quiet_hand_changing':(span<=2)&handactive,
                     'both_quiet':(span<=2)&(hand_span<=.02)}}


def metrics(state, first):
    out={}
    for side,base in [('left',0),('right',29)]:
        delta=state[...,base:base+3]-first[...,base:base+3]
        out[side+'_xyz']=np.linalg.norm(delta,axis=-1)*1000
        out[side+'_rotation']=rotation_distance_deg(state[...,base+3:base+9],first[...,base+3:base+9])
        out[side+'_hand']=abs(state[...,base+9:base+29]-first[...,base+9:base+29]).max(-1)
    out['right_z']=(state[...,31]-first[...,31])*1000
    return out


def describe(m, mask=None, weights=None):
    ans={}
    for key,x in m.items():
        v=x if mask is None else x[mask]
        w=weights if mask is None or weights is None else weights[mask]
        ans[key]=q(v,w)
    x=m['right_xyz'] if mask is None else m['right_xyz'][mask]
    w=weights if mask is None or weights is None else weights[mask]
    if len(x):
        ans['right_xyz_over_mm']={str(t):float(np.mean(x>t) if w is None else np.average(x>t,weights=w)) for t in [17.3,20,25,30,35]}
    return ans


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    cfg=OmegaConf.load(ROOT/'configs/train_wuji_rtc_ab.yaml')
    d=WujiRealTaskDataset.from_config(cfg.dataloader)
    d.rtc_state_history=RTCStateHistoryConfig(enabled=True)
    pack={k:[] for k in ['current','past','prefix','available','xyz_ok','rotation_ok','hand_ok','episode','local_span','state_local_span']}
    for ep,row in d._eps_df.iterrows():
        start,end=int(d._ep_valid_start[ep]),int(d._ep_valid_end[ep])
        table=d._load_data_table(int(row['data/chunk_index']),int(row['data/file_index']))
        df=table.slice(int(d._ep_data_row_offset[ep]),end).to_pandas()
        states=d._read_vector(df,'observation.state');actions=d._read_vector(df,'action');tasks=np.asarray(df['task_index'])
        offsets=np.arange(start,end-32,d._window_stride);n=len(offsets)
        if not n:continue
        current=states[offsets];past=np.stack([states[np.maximum(offsets-k,start)] for k in (1,2,3)],1)
        xyz=actions[start:end,29:32]
        local_span=np.linalg.norm(maximum_filter1d(xyz,151,axis=0,mode='nearest')-minimum_filter1d(xyz,151,axis=0,mode='nearest'),axis=-1)[offsets-start]*1000
        sampled_states=states[np.clip(offsets[:,None]+np.arange(-64,65,16)[None],start,end-1),29:32]
        state_local_span=np.linalg.norm(np.ptp(sampled_states,axis=1),axis=-1)*1000
        available=np.stack([offsets-k>=start for k in (1,2,3)],1)
        for j,k in enumerate((1,2,3)):
            for step in range(1,k+1):available[:,j]&=tasks[np.maximum(offsets-step,start)]==tasks[offsets]
        tests={name:np.ones((n,3,2),bool) for name in ['xyz_ok','rotation_ok','hand_ok']}
        for side,base in enumerate((0,29)):
            tests['xyz_ok'][:,:,side]=np.linalg.norm(past[:,:,base:base+3]-current[:,None,base:base+3],axis=-1)<=.025
            tests['rotation_ok'][:,:,side]=rotation_distance_deg(past[:,:,base+3:base+9],current[:,None,base+3:base+9])<=3
            tests['hand_ok'][:,:,side]=abs(past[:,:,base+9:base+29]-current[:,None,base+9:base+29]).max(-1)<=.05
        valid=available.copy()
        for v in tests.values():valid&=v.all(-1)
        for local in [0,n//2,n-1]:
            actual=read_history(d,int(d._cum_n_starts[ep])+local)
            np.testing.assert_array_equal(actual['rtc_history_valid'],valid[local])
        for k,v in dict(current=current,past=past,prefix=np.stack([actions[offsets+j] for j in range(10)],1),available=available,episode=np.full(n,ep),local_span=local_span,state_local_span=state_local_span,**tests).items():pack[k].append(v)
    p={k:np.concatenate(v) for k,v in pack.items()};n=len(p['current']);trajectory_info=trajectory(p['prefix'])
    trajectory_info['masks'].update(local_hold_30mm=p['local_span']<=30,local_move_50mm=p['local_span']>=50)
    trajectory_info['masks'].update(state_local_hold_30mm=p['state_local_span']<=30,state_local_move_50mm=p['state_local_span']>=50)
    # Preserve compact sufficient arrays for local follow-up analysis.
    np.savez_compressed(Path(args.output).with_suffix('.npz'),**p)
    variants={'clean':[], 'uncapped':[], 'bounded':['xyz_ok','rotation_ok','hand_ok'],
              'no_xyz_cap':['rotation_ok','hand_ok'],'no_rotation_cap':['xyz_ok','hand_ok'],
              'no_hand_cap':['xyz_ok','rotation_ok'],'xyz_cap_only':['xyz_ok']}
    selected={};chosen={};all_metrics={}
    lo=np.asarray(d._normalization_stats['q01']);hi=np.asarray(d._normalization_stats['q99'])
    def model_physical(x):
        y=d._normalize_array(x)
        return (torch.from_numpy(y).bfloat16().float().numpy()+1)*np.maximum(hi-lo,1e-6)/2+lo
    first=p['prefix'][:,0,None,:]
    result={'windows':n,'episodes':len(d._eps_df),'speed_bins_mm_s':['0-10','10-30','30-60','60-100','100-200','200+'],
            'phase_definition':'Prefix command XYZ bbox diagonal >5mm moving, <=2mm quiet; any hand joint prefix range >0.03rad changing; both quiet requires hand range <=0.02rad. Additional local +/-75 command frames (~5s) bbox <=30mm means near-position hold, >=50mm means moving. These do not identify true contact or task completion.',
            'training':{},'cap_failures':{},'real':{},'speed_hand_matched':{}}
    for name,limits in variants.items():
        valid=p['available'].copy()
        for key in limits:valid&=p[key].all(-1)
        if name=='clean':valid[:]=False
        choice=np.stack([np.where(valid&(np.arange(1,4)[None]<=request),np.arange(1,4),0).max(1) for request in (1,2,3)],1)
        values=np.where((choice>0)[...,None],p['past'][np.arange(n)[:,None],np.maximum(choice-1,0)],p['current'][:,None])
        chosen[name]=choice;selected[name]=values
        m=metrics(values,first);all_metrics[name]=m
        model=metrics(model_physical(values),model_physical(first))
        result['training'][name]={
            'effective_fraction_of_requested':float((choice>0).mean()),
            'requested_to_chosen_counts':[[int((choice[:,j]==k).sum()) for k in range(4)] for j in range(3)],
            'phases':{label:{'windows':int(mask.sum()),'metrics':describe(m,mask),'model_metrics':describe(model,mask),
                              'chosen_counts':[int((choice[mask]==k).sum()) for k in range(4)]} for label,mask in trajectory_info['masks'].items()},
            'speed_bins':{str(i):{'windows':int((trajectory_info['speedbin']==i).sum()),'metrics':describe(m,trajectory_info['speedbin']==i)} for i in range(6)},
            'by_requested_k':{str(j+1):describe({key:v[:,j] for key,v in m.items()}) for j in range(3)},
        }
    for label,mask in trajectory_info['masks'].items():
        a=p['available'][mask];tests={key:~p[key][mask].all(-1)&a for key in ['xyz_ok','rotation_ok','hand_ok']}
        any_fail=np.logical_or.reduce(list(tests.values()))
        entry={'available_candidates':int(a.sum()),'failed_any':int(any_fail.sum()),'failures_by_constraint':{},'exact_patterns':{}}
        for key,v in tests.items():
            other=np.logical_or.reduce([vv for kk,vv in tests.items() if kk!=key])
            entry['failures_by_constraint'][key]={'any':int(v.sum()),'only_this_constraint':int((v&~other).sum()),
                'left':int((~p[key][mask,:,0]&a).sum()),'right':int((~p[key][mask,:,1]&a).sum())}
        code=tests['xyz_ok'].astype(int)+tests['rotation_ok'].astype(int)*2+tests['hand_ok'].astype(int)*4
        entry['exact_patterns']={str(i):int(((code==i)&a).sum()) for i in range(8)}
        result['cap_failures'][label]=entry
    real=json.loads(Path(args.inputs).read_text());train_stratum=trajectory_info['stratum']
    for name,run in real.items():
        samples=run['samples'];s=np.array([v['state'] for v in samples]);prefix=np.array([v['prefix'] for v in samples]);keep=~np.array([v['last10s'] for v in samples])
        rt=trajectory(prefix);rm=metrics(s,prefix[:,0])
        local_span=np.array([v['local_command_5s_span_mm'] for v in samples])
        rt['masks'].update(local_hold_30mm=local_span<=30,local_move_50mm=local_span>=50)
        local_state_span=np.array([v['local_state_5s_span_mm'] for v in samples])
        rt['masks'].update(state_local_hold_30mm=local_state_span<=30,state_local_move_50mm=local_state_span>=50)
        result['real'][name]={'all':describe(rm),'exclude_last10':describe(rm,keep),
            'phases':{label:{'requests':int((mask&keep).sum()),'metrics':describe(rm,mask&keep)} for label,mask in rt['masks'].items()},
            'command_speed_mm_s':q(rt['speed'][keep]),
            'stratum_counts':np.bincount(rt['stratum'][keep],minlength=12).tolist()}
        weights=np.zeros(n)
        for idx in range(12):
            tm=train_stratum==idx;fraction=float((rt['stratum'][keep]==idx).mean())
            if fraction and not tm.any():raise ValueError('No matching training stratum '+str(idx))
            if tm.any():weights[tm]=fraction/tm.sum()
        result['speed_hand_matched'][name]={}
        for variant,m in all_metrics.items():
            w=np.repeat(weights[:,None]/3,3,1)
            mm=describe(m,weights=w)
            threshold=mm['right_xyz']['p99']
            mm['real_fraction_above_matched_train_p99']=float((rm['right_xyz'][keep]>threshold).mean())
            result['speed_hand_matched'][name][variant]=mm
    # Equal run weights so a long run does not dominate calibration.
    def multi_run(names):
        weights=np.zeros(n)
        for name in names:
            c=np.array(result['real'][name]['stratum_counts']);c=c/c.sum()/len(names)
            for idx in range(12):
                m=train_stratum==idx
                if m.any():weights[m]+=c[idx]/m.sum()
        return {name:describe(m,weights=np.repeat(weights[:,None]/3,3,1)) for name,m in all_metrics.items()}
    result['speed_hand_matched']['equal_runs_all5']=multi_run(list(real))
    result['speed_hand_matched']['equal_runs_earlier2']=multi_run(['0928','0929'])
    result['speed_hand_matched']['equal_runs_today3']=multi_run(['A','B','C'])
    result['notes']=['No model inference/training or robot execution.','Uniform requested k=1..3 enumerated for every window; summaries describe candidate augmentation including fallback, not the final 70% clean mixture.',
                     'state_local phases use actual logged request-state bbox within +/-2.5s; training uses 9 state points spaced 16 frames over +/-64 frames (~4.27s), approximating request sampling. Sparse samples and run endpoints limit exact phase attribution.',
                     'Motion matching is descriptive reweighting by command speed bin and hand activity; it is not matched physical scenes or causal identification.',
                     'Model physical summaries use the existing quantile normalization and BF16 for BOTH training state and training action. Real metrics use physical logged state/prefix.',
                     'Arm quiet does not guarantee terminal task phase. Prefix movement can itself include model oscillation.']
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({'windows':n,'output':args.output,'cap_failures':result['cap_failures']['all']},indent=2))


if __name__=='__main__':main()
