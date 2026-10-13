"""Paired absolute-EEF RTC sensitivity tests. No robot interfaces are imported.

Recorded observations do not respond to generated commands. State shifts are
controlled sensitivity probes, not a physical tracking simulation.
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


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--checkpoint-dir',type=Path,required=True)
    ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'results.jsonl').exists()
    sys.path.insert(0,str(args.repo))
    from openwam.dataloader.wuji_real_task import WujiRealTaskDataset
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.server import merge_deploy_cfg
    from openwam.deploy.engine import JointInferenceEngine
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.50, device=0)
    cfg=OmegaConf.load(args.checkpoint_dir/'config.yaml')
    assert cfg.dataloader.action_mode=='eef' and cfg.dataloader.rot6d_convention=='column'
    dataset=WujiRealTaskDataset.from_config(cfg.dataloader)
    selected=json.loads(args.selection.read_text())
    cache={}
    prepared={}
    for case,item in enumerate(selected):
        ep=item['episode']; offset=item['offset']
        assert offset>=6
        if ep not in cache:
            df=pd.read_parquet(Path(cfg.dataloader.dataset_dir)/f'data/chunk-000/episode_{ep:06d}.parquet')
            cache[ep]=(np.stack(df['action']).astype(np.float32),np.stack(df['observation.state']).astype(np.float32),df['timestamp'].to_numpy())
        actions,states,timestamps=cache[ep]
        for delta in [0,-2,-3,-6]:
            off=offset+delta
            idx=item['dataset_index']+delta
            assert int(np.searchsorted(dataset._cum_n_starts,idx,side='right')-1)==item['ep_local']
            sample=dataset[idx]
            image=sample['first_frame_image'][0]
            prepared[case,delta]=dict(image=image,prompt=sample['prompt'],state=states[off].copy(),
                norm_state=sample['proprio'].numpy().reshape(-1),norm_action=sample['action'].numpy(),
                raw=actions[off:off+32].copy(),timestamp=float(timestamps[off]),
                image_sha256=hashlib.sha256(np.asarray(image).tobytes()).hexdigest())
        assert all(prepared[case,d]['prompt']==prepared[case,0]['prompt'] for d in [-2,-3,-6])
        print('PREPARE',case,flush=True)
    protocol=dict(checkpoint=9000,windows=len(selected),seeds=[42,123],image_offsets=[0,-2,-3],
        state_backshift_mm=[0,5,15],prefix_modes=['teacher','model'],prefix_steps=10,new_steps=16,
        bootstrap='Unperturbed inference at t-6; committed prefix is bootstrap actions[6:16]',
        direction='Ground-truth prefix endpoint minus its first point, right XYZ only',
        extra_control='For each prefix mode and seed, image and full recorded state both at t-2',
        observations='Recorded training observations; no physical closed-loop simulation',
        image_shift='Additional relative frame shift; original camera/state alignment is unknown',
        compile=False,cache=False,denoise_steps=10,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        expected_calls=len(selected)*2*(1+2*3*3+2),
        config_sha256=sha256(args.checkpoint_dir/'config.yaml'),stats_sha256=sha256(args.checkpoint_dir/'normalization_stats.npy'))
    dump(args.out/'protocol.json',protocol)
    dump(args.out/'selection.json',selected)
    dump(args.out/'observation_manifest.json',[dict(case=c,delta=d,timestamp=o['timestamp'],image_sha256=o['image_sha256']) for (c,d),o in prepared.items()])
    deploy=OmegaConf.load(args.repo/'configs/deploy.yaml')
    for k,v in {'optimization.compile.enabled':False,'optimization.dit_cache.enabled':False,
                'optimization.save_video':False,'optimization.decode_video':False,
                'inference.rtc.enabled':True,'inference.inference_mode':'sync',
                'inference.denoise_steps':10,'inference.inference_horizon':None,
                'inference.inference_delay_steps':None}.items():OmegaConf.update(deploy,k,v)
    train,architecture=load_from_checkpoint_dir(str(args.checkpoint_dir),device='cuda:0',ckpt_name='checkpoint_step_9000.safetensors')
    engine=JointInferenceEngine(merge_deploy_cfg(train,deploy),architecture=architecture)
    for o in prepared.values():
        assert np.allclose(architecture.normalizer.normalize(o['state']),o['norm_state'],atol=2e-6)
        assert np.allclose(np.clip(architecture.normalizer.normalize_action(o['raw']),-1,1),o['norm_action'],atol=2e-6)
        o['gt']=architecture.normalizer.unnormalize(o['norm_action']).astype(np.float32)
    dump(args.out/'checkpoint.json',dict(sha256=sha256(args.checkpoint_dir/'checkpoint_step_9000.safetensors')))
    count=0
    started=time.monotonic()
    def infer(image_obs,state,prefix,seed):
        nonlocal count
        t=time.monotonic()
        with torch.inference_mode():
            result=engine.generate(dict(first_frame_image=[image_obs['image']],prompt=image_obs['prompt'],proprio=state,rtc_prefix=prefix,seed=seed))
        a=np.asarray(result['actions'],dtype=np.float32)
        assert a.shape==(32,58) and np.isfinite(a).all()
        count+=1
        return a,time.monotonic()-t
    try:
        for case,item in enumerate(selected):
            obs=prepared[case,0]; older=prepared[case,-6]
            direction=obs['gt'][9,29:32]-obs['gt'][0,29:32]
            assert np.linalg.norm(direction)>1e-7
            direction/=np.linalg.norm(direction)
            for seed in [42,123]:
                bootstrap,_=infer(older,older['state'],None,seed)
                with (args.out/'bootstrap.jsonl').open('a') as f:f.write(json.dumps(dict(case=case,seed=seed,actions=bootstrap.tolist()))+'\n')
                for mode,prefix in [('teacher',obs['gt'][:10].copy()),('model',bootstrap[6:16].copy())]:
                    conditions=[(delta,mm,'xyz_shift') for delta in [0,-2,-3] for mm in [0,5,15]]+[(-2,0,'recorded_state_minus2')]
                    baseline=None
                    for delta,mm,state_mode in conditions:
                        state=obs['state'].copy() if state_mode=='xyz_shift' else prepared[case,-2]['state'].copy()
                        state[29:32]-=(mm/1000)*direction
                        a,seconds=infer(prepared[case,delta],state,prefix,seed)
                        if delta==0 and mm==0:baseline=a.copy()
                        row=dict(case=case,episode=item['episode'],group=item['group'],seed=seed,prefix_mode=mode,
                            image_offset=delta,state_backshift_mm=mm,state_mode=state_mode,
                            state_delta_xyz_mm=((state[29:32]-obs['state'][29:32])*1000).tolist(),seconds=seconds,
                            metrics=seam_metrics(a,prefix,obs['gt'],state),
                            paired_first_delta_mm=float(np.linalg.norm(a[10,29:32]-baseline[10,29:32])*1000),
                            actions=a.tolist(),prefix=prefix.tolist(),state=state.tolist(),gt=obs['gt'].tolist())
                        with (args.out/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print('PROGRESS',count,'/',protocol['expected_calls'],'case',case,'seed',seed,'prefix',mode,'seconds',round(time.monotonic()-started,1),flush=True)
        assert count==protocol['expected_calls']
        dump(args.out/'complete.json',dict(calls=count,seconds=time.monotonic()-started))
    finally:engine.shutdown()

if __name__=='__main__':main()
