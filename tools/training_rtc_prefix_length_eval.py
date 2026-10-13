"""Robot-free prefix-length comparison, with a fixed physical handoff anchor.

For d=7..10, request later by 10-d recorded frames and preserve the same
committed endpoint and next 16 ground-truth actions. A fixed-observation
teacher-prefix control separates this from simply moving the seam earlier.
State displacements probe sensitivity; they do not simulate a robot.
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
    for name in ['repo','checkpoint-dir','selection','reference-run','out']:
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--shard',type=int,choices=[0,1],required=True)
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
    selected=json.loads(args.selection.read_text())
    cases=[i for i in range(len(selected)) if i%2==args.shard]
    bootstrap={(r['case'],r['seed']):np.array(r['actions'],dtype=np.float32)
        for r in map(json.loads,(args.reference_run/'bootstrap.jsonl').read_text().splitlines())}
    reference={}
    for line in (args.reference_run/'results.jsonl').open():
        r=json.loads(line)
        if r['image_offset']==0 and r['state_mode']=='xyz_shift':
            reference[r['case'],r['seed'],r['prefix_mode'],r['state_backshift_mm']]=np.array(r['actions'],dtype=np.float32)
    prepared={};cache={}
    for case in cases:
        item=selected[case];ep=item['episode']
        if ep not in cache:
            df=pd.read_parquet(Path(cfg.dataloader.dataset_dir)/f'data/chunk-000/episode_{ep:06d}.parquet')
            cache[ep]=(np.stack(df['action']).astype(np.float32),np.stack(df['observation.state']).astype(np.float32),df['timestamp'].to_numpy())
        actions,states,timestamps=cache[ep]
        for delta in range(4):
            off=item['offset']+delta;idx=item['dataset_index']+delta
            # Only the aligned prefix + 16 evaluated actions must be real.
            # The dataset may pad unused final horizon rows near episode end.
            assert item['offset']+26<=len(actions)
            assert int(np.searchsorted(dataset._cum_n_starts,idx,side='right')-1)==item['ep_local']
            sample=dataset[idx];image=sample['first_frame_image'][0]
            assert np.all(sample['action_mask'].numpy()[:26-delta].sum(axis=-1)==58)
            prepared[case,delta]=dict(image=image,prompt=sample['prompt'],state=states[off].copy(),
                norm_state=sample['proprio'].numpy().reshape(-1),norm_action=sample['action'].numpy(),
                raw=actions[off:off+32].copy(),action_mask=sample['action_mask'].numpy(),timestamp=float(timestamps[off]),
                image_sha256=hashlib.sha256(np.asarray(image).tobytes()).hexdigest())
        print('PREPARE',case,flush=True)
    protocol=dict(checkpoint=9000,cases=cases,seeds=[42,123],prefix_steps=[7,8,9,10],new_steps=16,
        state_backshift_mm=[0,5,15],main='fixed_handoff: observation and prefix start advance by 10-d frames; endpoint and evaluated suffix unchanged',
        control='fixed_observation: teacher prefixes of lengths 7,8,9; same observation, earlier seam; d10 shared with main',
        prefix_modes=['teacher','model'],model_prefix='Same unperturbed bootstrap at original t-6; slice [16-d:16]',
        perturbation_direction='Common original GT[9]-GT[0] for all d, normalized, only right XYZ',
        caveats='Train-split recorded observations; no physics, no historical-image replay, no left emergency transplant, no retraining',
        expected_calls=len(cases)*2*(4*2*3+3*3),compile=False,dit_cache=False,rtc_dit_cache=False,denoise_steps=10,
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),config_sha256=sha256(args.checkpoint_dir/'config.yaml'),
        stats_sha256=sha256(args.checkpoint_dir/'normalization_stats.npy'),bootstrap_sha256=sha256(args.reference_run/'bootstrap.jsonl'))
    dump(args.out/'protocol.json',protocol);dump(args.out/'selection.json',selected)
    dump(args.out/'observation_manifest.json',[dict(case=c,delta=d,timestamp=o['timestamp'],image_sha256=o['image_sha256']) for (c,d),o in prepared.items()])
    deploy=OmegaConf.load(args.repo/'configs/deploy.yaml')
    for k,v in {'optimization.compile.enabled':False,'optimization.dit_cache.enabled':False,
        'optimization.rtc_dit_cache.enabled':False,'optimization.save_video':False,'optimization.decode_video':False,
        'inference.rtc.enabled':True,'inference.inference_mode':'sync','inference.denoise_steps':10,
        'inference.inference_horizon':None,'inference.inference_delay_steps':None}.items():OmegaConf.update(deploy,k,v)
    train,architecture=load_from_checkpoint_dir(str(args.checkpoint_dir),device='cuda:0',ckpt_name='checkpoint_step_9000.safetensors')
    engine=JointInferenceEngine(merge_deploy_cfg(train,deploy),architecture=architecture)
    assert not getattr(engine,'_rtc_dit_cache_enabled',False) and engine._dit_cache is None
    for o in prepared.values():
        assert np.allclose(architecture.normalizer.normalize(o['state']),o['norm_state'],atol=2e-6)
        n=len(o['raw'])
        assert np.allclose(np.clip(architecture.normalizer.normalize_action(o['raw']),-1,1),o['norm_action'][:n],atol=2e-6)
        o['gt']=architecture.normalizer.unnormalize(o['norm_action']).astype(np.float32)
    dump(args.out/'checkpoint.json',dict(sha256=sha256(args.checkpoint_dir/'checkpoint_step_9000.safetensors')))
    started=time.monotonic();count=0
    try:
        for case in cases:
            item=selected[case];base=prepared[case,0]
            direction=base['gt'][9,29:32]-base['gt'][0,29:32]
            assert np.linalg.norm(direction)>1e-7
            direction/=np.linalg.norm(direction)
            for seed in [42,123]:
                for design in ['fixed_handoff','fixed_observation']:
                    for d in ([10,9,8,7] if design=='fixed_handoff' else [9,8,7]):
                        delta=10-d if design=='fixed_handoff' else 0
                        obs=prepared[case,delta]
                        assert obs['prompt']==base['prompt']
                        assert np.all(obs['action_mask'][:d+16].sum(axis=-1)==58)
                        for pm in (['teacher','model'] if design=='fixed_handoff' else ['teacher']):
                            prefix=(obs['gt'][:d] if pm=='teacher' else bootstrap[case,seed][16-d:16]).copy()
                            if design=='fixed_handoff':
                                original_prefix=base['gt'][:10] if pm=='teacher' else bootstrap[case,seed][6:16]
                                assert np.array_equal(prefix,original_prefix[10-d:])
                                assert np.array_equal(obs['gt'][d:d+16],base['gt'][10:26])
                                assert abs(obs['timestamp']-base['timestamp']-delta/30)<1e-5
                            for mm in [0,5,15]:
                                state=obs['state'].copy();state[29:32]-=mm/1000*direction
                                assert np.array_equal(state[:29],obs['state'][:29]) and np.array_equal(state[32:],obs['state'][32:])
                                t=time.monotonic()
                                with torch.inference_mode():
                                    result=engine.generate(dict(first_frame_image=[obs['image']],prompt=obs['prompt'],proprio=state,rtc_prefix=prefix,seed=seed))
                                a=np.asarray(result['actions'],dtype=np.float32);seconds=time.monotonic()-t
                                assert a.shape==(32,58) and np.isfinite(a).all()
                                row=dict(case=case,episode=item['episode'],group=item['group'],seed=seed,design=design,
                                    prefix_mode=pm,d=d,observation_delta=delta,state_backshift_mm=mm,seconds=seconds,
                                    metrics=seam_metrics(a,prefix,obs['gt'],state,d=d),actions=a.tolist(),
                                    state=state.tolist(),prefix=prefix.tolist(),gt=obs['gt'].tolist())
                                if d==10:
                                    ref=reference[case,seed,pm,mm]
                                    row['reference_right_max_abs_m']=float(np.max(np.abs(a[:,29:32]-ref[:,29:32])))
                                    assert row['reference_right_max_abs_m']<1e-5
                                with (args.out/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                                count+=1
                print('PROGRESS',count,'/',protocol['expected_calls'],'case',case,'seed',seed,'seconds',round(time.monotonic()-started,1),flush=True)
        assert count==protocol['expected_calls']
        dump(args.out/'complete.json',dict(calls=count,seconds=time.monotonic()-started))
    finally:engine.shutdown()


if __name__=='__main__':main()
