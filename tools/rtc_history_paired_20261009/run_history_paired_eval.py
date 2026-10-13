"""Paired RTC checkpoint audit: recorded observations, no robot or transport.

Data windows are diagnostic training-set samples, not held-out success trials.
"""
from pathlib import Path
import argparse,json,hashlib,os,sys,time,logging
from types import SimpleNamespace
import numpy as np
import torch
from omegaconf import OmegaConf
import pandas as pd

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path[:0]=[str(REPO),str(HERE/'client_snapshot')]
from openwam.deploy.model_loader import load_from_checkpoint_dir
from openwam.deploy.server import merge_deploy_cfg
from openwam.deploy.engine import JointInferenceEngine
from openwam.dataloader.wuji_real_task import WujiRealTaskDataset
from openwam.model.architectures.utils.rtc_state_history import rotation_distance_deg
from openwam_robot_client.rtc_controller import RTCController
from openwam_robot_client.geometry import eef9_column_to_pose7

OLD=REPO/'outputs/grasp_anything_base_rtc/2026-09-22_11-26-18'
NEW=REPO/'outputs/wuji_rtc_history/history/2026-10-08_23-55-31'

def dump(p,x):p.write_text(json.dumps(x,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def cosine(v,w):
 n=np.linalg.norm(v)*np.linalg.norm(w)
 return float(v@w/n) if n>1e-12 else None
def metrics(a,p,s,gt):
 d=0 if p is None else len(p);out={}
 for side,k in [('left',0),('right',29)]:
  for part,sl,scale in [('arm',slice(k,k+3),1000),('hand',slice(k+9,k+29),1)]:
   err=a[d:d+16,sl]-gt[d:d+16,sl]
   m=dict(first_error=float(np.linalg.norm(err[0])*scale),suffix_error=float(np.linalg.norm(err,axis=1).mean()*scale),
          suffix_mae=float(abs(err).mean()*scale),suffix_max_component_error=float(abs(err).max()*scale))
   if d:
    v=p[-1,sl]-p[-2,sl];w=a[d,sl]-p[-1,sl]
    c=cosine(v,w);back=s[sl]-p[-1,sl]
    m.update(incoming=float(np.linalg.norm(v)*scale),outgoing=float(np.linalg.norm(w)*scale),
       max_component=float(abs(w).max()*scale),cos=c,reverse=c is not None and c<0,
       toward_state_cos=cosine(w,back),state_first_gap=float(np.linalg.norm(s[sl]-p[0,sl])*scale),
       state_first_max_component=float(abs(s[sl]-p[0,sl]).max()*scale))
   out[side+'_'+part]=m
  out[side+'_rotation_suffix_deg']=float(np.mean(rotation_distance_deg(a[d:d+16,k+3:k+9],gt[d:d+16,k+3:k+9])))
 return out
def project(a,anchor):
 obj=RTCController.__new__(RTCController);obj.left_init_pose=None;obj.left_init_hand=None
 obj.max_eef_step=.03;obj.max_rotation_step_rad=np.deg2rad(10)
 obj.workspace_min=(anchor[:3]-2,anchor[29:32]-2);obj.workspace_max=(anchor[:3]+2,anchor[29:32]+2)
 obj.hands=SimpleNamespace(max_step=.1)
 poses=(eef9_column_to_pose7(anchor[:9]),eef9_column_to_pose7(anchor[29:38]))
 return np.stack([x.raw for x in obj.prepare_from(a,poses,(anchor[9:29],anchor[38:58]))])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['old','new'],required=True)
 ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 out=args.out/args.model;out.mkdir(parents=True,exist_ok=True)
 assert not (out/'results.jsonl').exists(),'Refusing to overwrite results'
 logging.basicConfig(level=logging.WARNING);torch.set_num_threads(4)
 torch.cuda.set_per_process_memory_fraction(.70,0)
 oldcfg=OmegaConf.load(OLD/'config.yaml');newcfg=OmegaConf.load(NEW/'config.yaml')
 assert OmegaConf.to_container(oldcfg.model)==OmegaConf.to_container(newcfg.model)
 assert sha(OLD/'normalization_stats.npy')==sha(NEW/'normalization_stats.npy')
 ds=WujiRealTaskDataset.from_config(oldcfg.dataloader)
 selected=json.loads((REPO/'outputs/rtc_offline_eval/20260929_114935/run/selection.json').read_text())
 assert len(selected)==36
 fine=[0,1,2,3,12,13,14,15,24,25,26,27];roll=[0,1,12,13,24,25]
 prepared={};frames={};manifest=[]
 def obs(case,delta=0):
  key=case,delta
  if key in prepared:return prepared[key]
  item=selected[case];ep=item['episode'];off=item['offset']+delta;idx=item['dataset_index']+delta
  assert int(np.searchsorted(ds._cum_n_starts,idx,side='right')-1)==item['ep_local']
  if ep not in frames:
   df=pd.read_parquet(Path(oldcfg.dataloader.dataset_dir)/f'data/chunk-000/episode_{ep:06d}.parquet')
   frames[ep]=(np.stack(df['observation.state']).astype(np.float32),np.stack(df['action']).astype(np.float32),np.asarray(df['task_index']))
  states,actions,tasks=frames[ep]
  history_start=max(0,off-3) if delta==-6 else off-3
  assert history_start>=0 and off+32<=len(states) and np.all(tasks[history_start:off+32]==tasks[off]), (case,delta,ep,off,len(states))
  sample=ds[idx];image=sample['first_frame_image'][0]
  # Bootstrap observations need no history intervention, including at episode start.
  history=np.stack([states[max(0,off-k)] for k in [1,2,3]])
  value=dict(image=image,prompt=sample['prompt'],state=states[off],history=history,
    normalized_actions=sample['action'].numpy(),normalized_state=sample['proprio'].numpy().reshape(-1))
  prepared[key]=value
  manifest.append(dict(case=case,delta=delta,episode=ep,offset=off,image_sha256=ah(np.asarray(image)),
    state_sha256=ah(value['state']),history_sha256=ah(value['history']),action_sha256=ah(value['normalized_actions'])))
  return value
 for case in range(36):obs(case)
 for case in fine:obs(case,-6)
 for case in roll:
  for delta in [16,32]:obs(case,delta)
 dump(out/'inputs.json',manifest);dump(out/'selection.json',selected)
 ckpt,step=(OLD,9000) if args.model=='old' else (NEW,4500)
 cfg,arch=load_from_checkpoint_dir(str(ckpt),device='cuda:0',ckpt_name=f'checkpoint_step_{step}.safetensors')
 deploy=OmegaConf.load(REPO/'configs/deploy.yaml')
 for k,v in {'optimization.compile.enabled':False,'optimization.dit_cache.enabled':False,'optimization.rtc_dit_cache.enabled':False,
             'optimization.save_video':False,'optimization.decode_video':False,'inference.rtc.enabled':True,
             'inference.inference_mode':'sync','inference.denoise_steps':10,
             'inference.inference_horizon':None,'inference.inference_delay_steps':None}.items():OmegaConf.update(deploy,k,v,merge=True)
 engine=JointInferenceEngine(merge_deploy_cfg(cfg,deploy),architecture=arch);norm=arch.normalizer
 for o in prepared.values():
  o['gt']=norm.unnormalize(o['normalized_actions']).astype(np.float32)
  assert np.allclose(norm.normalize(o['state']),o['normalized_state'],atol=2e-6)
 calls=0;start=time.monotonic()
 def infer(o,s,p,seed):
  nonlocal calls
  with torch.inference_mode():
   result=engine.generate(dict(first_frame_image=[o['image']],prompt=o['prompt'],proprio=s,rtc_prefix=p,seed=seed))
  a=np.asarray(result['actions'],np.float32);assert a.shape==(32,58) and np.isfinite(a).all();calls+=1
  return a
 def record(family,case,mode,seed,o,s,p,**extra):
  t=time.monotonic();a=infer(o,s,p,seed)
  row=dict(model=args.model,family=family,case=case,group=selected[case]['group'],mode=mode,seed=seed,
      seconds=time.monotonic()-t,actions=a.tolist(),state=np.asarray(s).tolist(),prefix=None if p is None else np.asarray(p).tolist(),
      gt=o['gt'].tolist(),metrics=metrics(a,p,s,o['gt']),**extra)
  # Quantile clipping is measured for each intervention, not assumed absent.
  lo=np.asarray(norm.stats['q01']);hi=np.asarray(norm.stats['q99'])
  row['clipped_state_components']=np.flatnonzero((s<lo)|(s>hi)).tolist()
  with (out/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if calls%20==0:print('PROGRESS',args.model,calls,round(time.monotonic()-start,1),flush=True)
  return a
 protocol=dict(model=args.model,checkpoint=str(ckpt/f'checkpoint_step_{step}.safetensors'),clean_cases=36,
  stress_cases=fine,rollout_cases=roll,seeds=[42,123],prefix=10,execution_horizon=16,denoise_steps=10,
  cache=False,compile=False,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,
  generated_bank='Each producer supplies the identical raw/projected prefix to both consumer checkpoints',
  state_shift='Both arms/hands shifted symmetrically by their own prefix trend; actual history replaces all 58 components',
  limitations='Training-set diagnostic windows. Recorded-image/state rollout is not physical closed-loop simulation.')
 dump(out/'protocol.json',protocol);print('MODEL_READY',args.model,flush=True)
 try:
  bank=[]
  for case in fine:
   o=obs(case,-6)
   for seed in [42,123]:
    a=infer(o,o['state'],None,seed)
    bank.append(dict(case=case,seed=seed,raw=a[:16].tolist(),limited=project(a[:16],o['state']).tolist()))
  temp=out/'bank.tmp';dump(temp,bank);temp.rename(out/'bank.json')
  # Larger common clean/history set first, followed by controlled stress probes.
  for case in range(36):
   o=obs(case);s=o['state'];p=o['gt'][:10].copy()
   for seed in [42,123]:
    base=record('teacher',case,'clean',seed,o,s,p)
    if case==0 and seed==42:
     repeated=infer(o,s,p,seed);assert np.array_equal(base,repeated),'Seed repeat mismatch'
     dump(out/'determinism.json',dict(exact=True,action_sha256=ah(base)))
    for k in [1,2,3]:record('teacher',case,f'history{k}',seed,o,o['history'][k-1],p)
    if case not in fine:continue
    record('teacher',case,'empty',seed,o,s,None)
    def shifted(arm_m=0,hand_rad=0):
     state=s.copy()
     for k in [0,29]:
      v=p[-1,k:k+3]-p[0,k:k+3];v=v/max(np.linalg.norm(v),1e-8)
      h=p[-1,k+9:k+29]-p[0,k+9:k+29];h=h/max(abs(h).max(),1e-8)
      state[k:k+3]+=arm_m*v;state[k+9:k+29]+=hand_rad*h
     return state
    for mm in [-15,-30,-45,15]:record('stress',case,f'arm{mm:+d}mm',seed,o,shifted(arm_m=mm/1000),p)
    for rad in [-.05,-.10,.05]:record('stress',case,f'hand{rad:+.2f}',seed,o,shifted(hand_rad=rad),p)
    record('stress',case,'combined_30mm_005rad',seed,o,shifted(-.03,-.05),p)
   print('TEACHER_CASE_DONE',args.model,case,flush=True)
  deadline=time.monotonic()+600
  while not all((args.out/m/'bank.json').exists() for m in ['old','new']):
   if time.monotonic()>deadline:raise TimeoutError('Other model prefix bank not ready')
   time.sleep(2)
  for producer in ['old','new']:
   for b in json.loads((args.out/producer/'bank.json').read_text()):
    o=obs(b['case']);s=o['state']
    for kind in ['raw','limited']:
     p=np.asarray(b[kind],np.float32)[6:16]
     record('cross_generated',b['case'],producer+'_'+kind,b['seed'],o,s,p,producer=producer)
  # Rolling continuation uses recorded observations at t, t+16, t+32.
  own={(b['case'],b['seed']):b for b in bank}
  for case in roll:
   for seed in [42,123]:
    for kind in ['raw','limited']:
     p=np.asarray(own[case,seed][kind],np.float32)[6:16]
     for round_id,delta in enumerate([0,16,32]):
      o=obs(case,delta);s=o['state']
      a=record('rollout',case,kind,seed,o,s,p,round=round_id,observation_delta=delta)
      new=a[10:26]
      if kind=='limited':new=project(new,p[-1])
      p=new[6:16].copy()
  dump(out/'complete.json',dict(calls=calls,seconds=time.monotonic()-start,peak_allocated_GB=torch.cuda.max_memory_allocated()/1e9))
 finally:engine.shutdown()

if __name__=='__main__':main()
