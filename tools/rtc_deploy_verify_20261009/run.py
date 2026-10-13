"""Offline paired image-age / prefix-filter interventions; never controls robots."""
from pathlib import Path
import argparse,json,hashlib,sys,time,os
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as Rot,Slerp
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path.insert(0,str(REPO))
from openwam.deploy.model_loader import load_from_checkpoint_dir
from openwam.deploy.server import merge_deploy_cfg
from openwam.deploy.engine import JointInferenceEngine
from openwam.dataloader.wuji_real_task import WujiRealTaskDataset

def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def rot(x):
 u=x[:3]/np.linalg.norm(x[:3]);v=x[3:]-u*(u@x[3:]);v=v/np.linalg.norm(v);return Rot.from_matrix(np.stack([u,v,np.cross(u,v)],axis=1))
def filt(actions,initial):
 y=np.asarray(initial,dtype=np.float64).copy();out=[]
 for a in actions:
  z=np.asarray(a,dtype=np.float64).copy()
  for k in [0,29]:
   z[k:k+3]=.35*a[k:k+3]+.65*y[k:k+3];m=Slerp([0,1],Rot.concatenate([rot(y[k+3:k+9]),rot(a[k+3:k+9])]))([.35]).as_matrix()[0];z[k+3:k+9]=m[:,:2].T.reshape(-1)
  y=z;out.append(y.copy())
 return np.asarray(out,np.float32)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['old','new'],required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 dest=args.out/args.model;dest.mkdir();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.27,0)
 old=REPO/'outputs/grasp_anything_base_rtc/2026-09-22_11-26-18';new=REPO/'outputs/wuji_rtc_history/history/2026-10-08_23-55-31';cfg0=OmegaConf.load(old/'config.yaml');ds=WujiRealTaskDataset.from_config(cfg0.dataloader)
 sel=json.loads((REPO/'outputs/rtc_history_paired_20261009/new/selection.json').read_text());bank=json.loads((REPO/'outputs/rtc_history_paired_20261009/new/bank.json').read_text())
 import pandas as pd
 frames={};prepared={};manifest=[]
 for b in bank:
  case=b['case'];item=sel[case];ep=item['episode'];off=item['offset'];idx=item['dataset_index']
  if case in prepared:continue
  if ep not in frames:frames[ep]=pd.read_parquet(Path(cfg0.dataloader.dataset_dir)/f'data/chunk-000/episode_{ep:06d}.parquet')
  df=frames[ep];assert off>=6
  images={lag:ds[idx-lag]['first_frame_image'][0] for lag in [0,2,3]};o=ds[idx]
  prepared[case]=dict(image=images,prompt=o['prompt'],state=np.asarray(df.iloc[off]['observation.state'],np.float32),initial=np.asarray(df.iloc[off-6]['observation.state'],np.float32),normalized_actions=o['action'].numpy())
  manifest.append(dict(case=case,episode=ep,offset=off,image_hash={lag:sha(np.asarray(im)) for lag,im in images.items()},state_hash=sha(prepared[case]['state'])))
 (dest/'inputs.json').write_text(json.dumps(manifest,indent=2))
 root,step=(old,9000) if args.model=='old' else (new,4500)
 cfg,arch=load_from_checkpoint_dir(str(root),device='cuda:0',ckpt_name=f'checkpoint_step_{step}.safetensors');deploy=OmegaConf.load(REPO/'configs/deploy.yaml')
 for k,v in {'optimization.compile.enabled':False,'optimization.dit_cache.enabled':False,'optimization.rtc_dit_cache.enabled':False,'optimization.save_video':False,'optimization.decode_video':False,'inference.rtc.enabled':True,'inference.inference_mode':'sync','inference.denoise_steps':10,'inference.inference_horizon':None,'inference.inference_delay_steps':None}.items():OmegaConf.update(deploy,k,v,merge=True)
 engine=JointInferenceEngine(merge_deploy_cfg(cfg,deploy),architecture=arch);start=time.monotonic();calls=0
 (dest/'protocol.json').write_text(json.dumps(dict(model=args.model,checkpoint=str(root/f'checkpoint_step_{step}.safetensors'),producer='common history4500 bank from previous paired test',image_lag_frames=[0,2,3],prefix=10,control_hz=30,alpha=.35,filter='both EEF XYZ and rotations; hands unchanged',limitation='training-set diagnostic images; filtered prefixes approximate SDK EMA, not IK/native desired; not historical robot replay or closed-loop simulation'),indent=2))
 print('READY',args.model,flush=True)
 try:
  with (dest/'results.jsonl').open('x') as output:
   for b in bank:
    case=b['case'];o=prepared[case];raw=np.asarray(b['limited'],np.float32);f=filt(raw,o['initial']);teacher=arch.normalizer.unnormalize(o['normalized_actions']).astype(np.float32)
    for family,prefix in [('generated_raw',raw[6:16]),('generated_filtered',f[6:16]),('teacher',teacher[:10])]:
     for lag in ([0,2,3] if family!='teacher' else [0,3]):
      with torch.inference_mode():result=engine.generate(dict(first_frame_image=[o['image'][lag]],prompt=o['prompt'],proprio=o['state'],rtc_prefix=prefix,seed=b['seed']))
      a=np.asarray(result['actions'],np.float32);assert a.shape==(32,58) and np.isfinite(a).all()
      row=dict(model=args.model,case=case,seed=b['seed'],family=family,image_lag=lag,prefix=prefix.tolist(),committed_raw=raw[6:16].tolist(),committed_filtered=f[6:16].tolist(),state=o['state'].tolist(),actions=a.tolist(),executed_filter_suffix=filt(a[10:26],f[-1]).tolist())
      output.write(json.dumps(row)+'\n');output.flush();calls+=1
      if calls%24==0:print('PROGRESS',args.model,calls,round(time.monotonic()-start,1),flush=True)
   # Repeated baseline verifies reproducibility after all calls.
   b=bank[0];o=prepared[b['case']];prefix=np.asarray(b['limited'],np.float32)[6:16]
   with torch.inference_mode():repeat=engine.generate(dict(first_frame_image=[o['image'][0]],prompt=o['prompt'],proprio=o['state'],rtc_prefix=prefix,seed=b['seed']))
   first=json.loads((dest/'results.jsonl').open().readline());assert np.array_equal(np.asarray(first['actions'],np.float32),np.asarray(repeat['actions'],np.float32))
  (dest/'complete.json').write_text(json.dumps(dict(calls=calls+1,deterministic=True,seconds=time.monotonic()-start,peak_allocated_GB=torch.cuda.max_memory_allocated()/1e9),indent=2));print('DONE',args.model,flush=True)
 finally:engine.shutdown()
if __name__=='__main__':main()
