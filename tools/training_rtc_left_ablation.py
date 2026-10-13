"""Robot-free paired tests of logged left emergency conditions in absolute EEF.

No SDK is imported. Images/right inputs are deliberately fixed to isolate model
conditioning. Swapping logged left inputs is not a synchronized visual replay.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from training_rtc_eval import seam_metrics, dump, sha256


TEACHER_MODES=['baseline','thumb_state','thumb_both','hand_state','hand_prefix',
               'hand_both','arm_both','full_left','local_left_hold']
MODEL_MODES=['baseline','thumb_both','hand_both','full_left']


def condition(state,prefix,observed_left,target_left,mode):
    s=state.copy();p=prefix.copy()
    state_dims={'baseline':[], 'thumb_state':list(range(9,13)), 'thumb_both':list(range(9,13)),
        'hand_state':list(range(9,29)), 'hand_prefix':[], 'hand_both':list(range(9,29)),
        'arm_both':list(range(9)), 'full_left':list(range(29)), 'local_left_hold':[]}[mode]
    prefix_dims={'baseline':[], 'thumb_state':[], 'thumb_both':list(range(9,13)),
        'hand_state':[], 'hand_prefix':list(range(9,29)), 'hand_both':list(range(9,29)),
        'arm_both':list(range(9)), 'full_left':list(range(29)), 'local_left_hold':list(range(29))}[mode]
    if state_dims:s[state_dims]=observed_left[state_dims]
    if prefix_dims:p[:,prefix_dims]=p[0,prefix_dims] if mode=='local_left_hold' else target_left[prefix_dims]
    assert np.array_equal(s[29:],state[29:]) and np.array_equal(p[:,29:],prefix[:,29:])
    assert np.array_equal(s[[i for i in range(29) if i not in state_dims]],state[[i for i in range(29) if i not in state_dims]])
    assert np.array_equal(p[:,[i for i in range(29) if i not in prefix_dims]],prefix[:,[i for i in range(29) if i not in prefix_dims]])
    return s,p


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','checkpoint-dir','selection','emergency','reference-run','out']:
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'results.jsonl').exists()
    sys.path.insert(0,str(args.repo))
    from openwam.dataloader.wuji_real_task import WujiRealTaskDataset
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.server import merge_deploy_cfg
    from openwam.deploy.engine import JointInferenceEngine
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.50,device=0)
    cfg=OmegaConf.load(args.checkpoint_dir/'config.yaml')
    assert cfg.dataloader.action_mode=='eef' and cfg.dataloader.rot6d_convention=='column'
    dataset=WujiRealTaskDataset.from_config(cfg.dataloader)
    selected=json.loads(args.selection.read_text());emergency=json.loads(args.emergency.read_text())
    emergency_cases={r['case']:r for r in emergency['cases']}
    bootstrap={(r['case'],r['seed']):np.array(r['actions'],dtype=np.float32) for r in map(json.loads,(args.reference_run/'bootstrap.jsonl').read_text().splitlines())}
    reference={}
    for line in (args.reference_run/'results.jsonl').open():
        r=json.loads(line)
        if r['image_offset']==0 and r['state_mode']=='xyz_shift':
            reference[r['case'],r['seed'],r['prefix_mode'],r['state_backshift_mm']]=np.array(r['actions'],dtype=np.float32)
    prepared=[];cache={}
    for case,item in enumerate(selected):
        ep=item['episode'];off=item['offset'];idx=item['dataset_index']
        if ep not in cache:
            df=pd.read_parquet(Path(cfg.dataloader.dataset_dir)/f'data/chunk-000/episode_{ep:06d}.parquet')
            cache[ep]=(np.stack(df['action']).astype(np.float32),np.stack(df['observation.state']).astype(np.float32))
        actions,states=cache[ep]
        assert int(np.searchsorted(dataset._cum_n_starts,idx,side='right')-1)==item['ep_local']
        sample=dataset[idx];image=sample['first_frame_image'][0]
        prepared.append(dict(image=image,prompt=sample['prompt'],state=states[off].copy(),
            norm_state=sample['proprio'].numpy().reshape(-1),norm_action=sample['action'].numpy(),raw=actions[off:off+32],
            image_sha256=hashlib.sha256(np.asarray(image).tobytes()).hexdigest()))
        print('PREPARE',case,flush=True)
    protocol=dict(checkpoint=9000,windows=len(selected),seeds=[42,123],
        teacher_modes=TEACHER_MODES,model_modes=MODEL_MODES,
        right_state_backshift_interactions=dict(modes=['baseline','full_left'],mm=[5,15]),
        expected_calls=len(selected)*2*(len(TEACHER_MODES)+len(MODEL_MODES)+8),
        compile=False,dit_cache=False,rtc_dit_cache=False,denoise_steps=10,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        emergency_source_sha256=emergency['log_sha256'],bootstrap_sha256=sha256(args.reference_run/'bootstrap.jsonl'),
        reference_results_sha256=sha256(args.reference_run/'results.jsonl'),
        config_sha256=sha256(args.checkpoint_dir/'config.yaml'),stats_sha256=sha256(args.checkpoint_dir/'normalization_stats.npy'),
        caveats='Recorded training images unchanged; no synchronized image of transplanted left state, no physical feedback, no SDK semantics validation',
        model_prefix='Reuse previous unperturbed bootstrap actions[6:16]; right prefix is identical across all left conditions',
        local_left_hold='Freeze all 29 left prefix dimensions to the first prefix frame; retain original recorded state')
    dump(args.out/'protocol.json',protocol);dump(args.out/'selection.json',selected);dump(args.out/'emergency.json',emergency)
    dump(args.out/'observation_manifest.json',[dict(case=i,image_sha256=o['image_sha256']) for i,o in enumerate(prepared)])
    deploy=OmegaConf.load(args.repo/'configs/deploy.yaml')
    for k,v in {'optimization.compile.enabled':False,'optimization.dit_cache.enabled':False,
                'optimization.rtc_dit_cache.enabled':False,'optimization.save_video':False,'optimization.decode_video':False,
                'inference.rtc.enabled':True,'inference.inference_mode':'sync','inference.denoise_steps':10,
                'inference.inference_horizon':None,'inference.inference_delay_steps':None}.items():OmegaConf.update(deploy,k,v)
    train,architecture=load_from_checkpoint_dir(str(args.checkpoint_dir),device='cuda:0',ckpt_name='checkpoint_step_9000.safetensors')
    engine=JointInferenceEngine(merge_deploy_cfg(train,deploy),architecture=architecture)
    assert not getattr(engine,'_rtc_dit_cache_enabled',False) and engine._dit_cache is None
    for o in prepared:
        assert np.allclose(architecture.normalizer.normalize(o['state']),o['norm_state'],atol=2e-6)
        assert np.allclose(np.clip(architecture.normalizer.normalize_action(o['raw']),-1,1),o['norm_action'],atol=2e-6)
        o['gt']=architecture.normalizer.unnormalize(o['norm_action']).astype(np.float32)
    dump(args.out/'checkpoint.json',dict(sha256=sha256(args.checkpoint_dir/'checkpoint_step_9000.safetensors')))
    started=time.monotonic();count=0
    try:
        for case,(item,obs) in enumerate(zip(selected,prepared)):
            observed_left=np.array(emergency_cases[case]['state_left29'],dtype=np.float32)
            target_left=np.array(emergency_cases[case]['prefix_left29'],dtype=np.float32)
            direction=obs['gt'][9,29:32]-obs['gt'][0,29:32];direction/=np.linalg.norm(direction)
            for seed in [42,123]:
                for prefix_mode,modes,baseprefix in [('teacher',TEACHER_MODES,obs['gt'][:10]),('model',MODEL_MODES,bootstrap[case,seed][6:16])]:
                    configs=[(mode,0) for mode in modes]+[(mode,mm) for mm in [5,15] for mode in ['baseline','full_left']]
                    baselines={}
                    for mode,mm in configs:
                        state,prefix=condition(obs['state'],baseprefix,observed_left,target_left,mode)
                        state[29:32]-=mm/1000*direction
                        assert np.array_equal(state[32:],obs['state'][32:])
                        assert np.array_equal(prefix[:,29:],baseprefix[:,29:])
                        t=time.monotonic()
                        with torch.inference_mode():
                            result=engine.generate(dict(first_frame_image=[obs['image']],prompt=obs['prompt'],proprio=state,rtc_prefix=prefix,seed=seed))
                        a=np.asarray(result['actions'],dtype=np.float32);seconds=time.monotonic()-t
                        assert a.shape==(32,58) and np.isfinite(a).all()
                        count+=1
                        if mode=='baseline':baselines[mm]=a.copy()
                        b=baselines[mm]
                        row=dict(case=case,episode=item['episode'],group=item['group'],seed=seed,prefix_mode=prefix_mode,
                            mode=mode,right_state_backshift_mm=mm,emergency_request_id=emergency_cases[case]['request_id'],seconds=seconds,
                            metrics=seam_metrics(a,prefix,obs['gt'],state),
                            paired_first_delta_mm=float(np.linalg.norm(a[10,29:32]-b[10,29:32])*1000),
                            paired_suffix_delta_mm=float(np.linalg.norm(a[10:26,29:32]-b[10:26,29:32],axis=1).mean()*1000),
                            actions=a.tolist(),state=state.tolist(),prefix=prefix.tolist(),gt=obs['gt'].tolist())
                        if mode=='baseline':
                            ref=reference[case,seed,prefix_mode,mm]
                            row['reference_right_max_abs_m']=float(np.max(np.abs(a[:,29:32]-ref[:,29:32])))
                            assert row['reference_right_max_abs_m']<1e-5
                        with (args.out/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print('PROGRESS',count,'/',protocol['expected_calls'],'case',case,'seed',seed,'prefix',prefix_mode,'seconds',round(time.monotonic()-started,1),flush=True)
        assert count==protocol['expected_calls']
        dump(args.out/'complete.json',dict(calls=count,seconds=time.monotonic()-started))
    finally:engine.shutdown()


if __name__=='__main__':main()
