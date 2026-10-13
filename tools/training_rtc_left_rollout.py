"""Recorded-observation RTC rollouts with the same left condition from bootstrap.

Generated right prefixes recur; recorded observations do not respond physically.
"""
from pathlib import Path
import argparse,json,os,sys,time
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from training_rtc_eval import dump,seam_metrics,sha256


def main():
    ap=argparse.ArgumentParser()
    for name in ['repo','checkpoint-dir','trajectories','emergency','reference','out']:
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'results.jsonl').exists()
    sys.path.insert(0,str(args.repo))
    from openwam.dataloader.wuji_real_task import WujiRealTaskDataset
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.server import merge_deploy_cfg
    from openwam.deploy.engine import JointInferenceEngine
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.5,device=0)
    cfg=OmegaConf.load(args.checkpoint_dir/'config.yaml');dataset=WujiRealTaskDataset.from_config(cfg.dataloader)
    assert cfg.dataloader.action_mode=='eef' and cfg.dataloader.rot6d_convention=='column'
    selected=json.loads(args.trajectories.read_text());em={r['case']:r for r in json.loads(args.emergency.read_text())['cases']}
    reference={}
    for line in args.reference.open():
        r=json.loads(line)
        if r['checkpoint']==9000 and r['mode']=='rollout_raw':reference[r['case'],r['request']]=np.array(r['actions'],dtype=np.float32)
    prepared={}
    for item in selected:
        df=pd.read_parquet(Path(cfg.dataloader.dataset_dir)/f"data/chunk-000/episode_{item['episode']:06d}.parquet")
        acts=np.stack(df['action']).astype(np.float32);states=np.stack(df['observation.state']).astype(np.float32)
        for request in range(7):
            delta=0 if request==0 else 6+16*(request-1)
            off=item['offset']+delta;idx=item['dataset_index']+delta
            assert int(np.searchsorted(dataset._cum_n_starts,idx,side='right')-1)==item['ep_local']
            sample=dataset[idx]
            prepared[item['case'],request]=dict(image=sample['first_frame_image'][0],prompt=sample['prompt'],state=states[off].copy(),
                norm_state=sample['proprio'].numpy().reshape(-1),norm_action=sample['action'].numpy(),raw=acts[off:off+32],offset=off)
        print('PREPARE',item['case'],flush=True)
    deploy=OmegaConf.load(args.repo/'configs/deploy.yaml')
    for k,v in {'optimization.compile.enabled':False,'optimization.dit_cache.enabled':False,'optimization.rtc_dit_cache.enabled':False,
                'optimization.save_video':False,'optimization.decode_video':False,'inference.rtc.enabled':True,
                'inference.inference_mode':'sync','inference.denoise_steps':10,'inference.inference_horizon':None,'inference.inference_delay_steps':None}.items():OmegaConf.update(deploy,k,v)
    train,architecture=load_from_checkpoint_dir(str(args.checkpoint_dir),device='cuda:0',ckpt_name='checkpoint_step_9000.safetensors')
    engine=JointInferenceEngine(merge_deploy_cfg(train,deploy),architecture=architecture)
    for o in prepared.values():
        assert np.allclose(architecture.normalizer.normalize(o['state']),o['norm_state'],atol=2e-6)
        assert np.allclose(np.clip(architecture.normalizer.normalize_action(o['raw']),-1,1),o['norm_action'],atol=2e-6)
        o['gt']=architecture.normalizer.unnormalize(o['norm_action']).astype(np.float32)
    expected=len(selected)*2*3*7
    dump(args.out/'protocol.json',dict(trajectories=selected,seeds=[42,123],modes=['baseline','hand_both','full_left'],requests=6,
        expected_calls=expected,bootstrap_steps=16,prefix_steps=10,adopted_suffix_steps=16,
        left_condition='Same recorded left state and fixed target from bootstrap through all 6 requests',
        observations='Recorded images and right state advance at offsets 0,6,22,38,54,70,86, independent of generated commands',
        exclusions='No physical closed loop; no SDK filters or hand/rotation step projection except explicit left hold',
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),checkpoint_sha256=sha256(args.checkpoint_dir/'checkpoint_step_9000.safetensors')))
    count=0;started=time.monotonic()
    def infer(obs,state,prefix,seed):
        nonlocal count
        with torch.inference_mode():r=engine.generate(dict(first_frame_image=[obs['image']],prompt=obs['prompt'],proprio=state,rtc_prefix=prefix,seed=seed))
        a=np.asarray(r['actions'],dtype=np.float32);assert a.shape==(32,58) and np.isfinite(a).all();count+=1
        return a
    try:
        for item in selected:
            case=item['case'];observed=np.array(em[case]['state_left29'],dtype=np.float32);target=np.array(em[case]['prefix_left29'],dtype=np.float32)
            for seed in [42,123]:
                for mode in ['baseline','hand_both','full_left']:
                    dims=[] if mode=='baseline' else list(range(9,29)) if mode=='hand_both' else list(range(29))
                    obs=prepared[case,0];state=obs['state'].copy()
                    if dims:state[dims]=observed[dims]
                    first=infer(obs,state,None,seed);queue=first[:16].copy()
                    if dims:queue[:,dims]=target[dims]
                    with (args.out/'bootstrap.jsonl').open('a') as f:f.write(json.dumps(dict(case=case,seed=seed,mode=mode,state=state.tolist(),actions=first.tolist(),committed=queue.tolist()))+'\n')
                    for request in range(1,7):
                        obs=prepared[case,request];state=obs['state'].copy()
                        if dims:state[dims]=observed[dims]
                        prefix=queue[-10:].copy();a=infer(obs,state,prefix,seed)
                        queue=a[10:26].copy()
                        if dims:queue[:,dims]=target[dims]
                        row=dict(case=case,episode=item['episode'],group=item['group'],seed=seed,mode=mode,request=request,offset=obs['offset'],
                            metrics=seam_metrics(a,prefix,obs['gt'],state),state=state.tolist(),prefix=prefix.tolist(),actions=a.tolist(),committed=queue.tolist(),gt=obs['gt'].tolist())
                        if mode=='baseline' and seed==42:
                            row['reference_right_max_abs_m']=float(np.max(np.abs(a[:,29:32]-reference[case,request][:,29:32])))
                            assert row['reference_right_max_abs_m']<1e-5
                        with (args.out/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print('PROGRESS',count,'/',expected,'case',case,'seed',seed,'mode',mode,'seconds',round(time.monotonic()-started,1),flush=True)
        assert count==expected;dump(args.out/'complete.json',dict(calls=count,seconds=time.monotonic()-started))
    finally:engine.shutdown()


if __name__=='__main__':main()
