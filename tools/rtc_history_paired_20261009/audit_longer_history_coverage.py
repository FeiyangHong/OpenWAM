"""Read-only population coverage; does not change the training history sampler."""
from pathlib import Path
import sys,json
import numpy as np
import torch
from omegaconf import OmegaConf
root=Path('/gaozt-test1/fyhong/OpenWAM');sys.path.insert(0,str(root))
from openwam.dataloader.wuji_real_task import WujiRealTaskDataset
from openwam.model.architectures.utils.rtc_state_history import rotation_distance_deg
torch.set_num_threads(2)
cfg=OmegaConf.load(root/'outputs/wuji_rtc_history/history/2026-10-08_23-55-31/config.yaml')
d=WujiRealTaskDataset.from_config(cfg.dataloader)
lo=np.asarray(d._normalization_stats['q01']);hi=np.asarray(d._normalization_stats['q99']);scale=np.maximum(hi-lo,1e-6)
def physical(v):
 a=d._normalize_array(v)
 return (torch.from_numpy(a).bfloat16().float().numpy()+1)*scale/2+lo
ks=[0,1,2,3,4,5,6,9,12];values={k:[] for k in ks}
for ep,row in d._eps_df.iterrows():
 table=d._load_data_table(int(row['data/chunk_index']),int(row['data/file_index']))
 start,end=int(d._ep_valid_start[ep]),int(d._ep_valid_end[ep])
 raw=table.slice(int(d._ep_data_row_offset[ep]),end).to_pandas()
 states=d._read_vector(raw,'observation.state');actions=d._read_vector(raw,'action');tasks=np.asarray(raw['task_index'])
 offsets=start+np.arange(int(d._cum_n_starts[ep+1]-d._cum_n_starts[ep]))*d._window_stride
 for k in ks:
  off=offsets[offsets-k>=start];past=states[off-k];target=actions[off];valid=np.isfinite(past).all(1)&np.isfinite(target).all(1)
  for j in range(1,k+1):valid&=tasks[off-j]==tasks[off]
  for b in [0,29]:valid&=np.isfinite(rotation_distance_deg(past[:,b+3:b+9],states[off,b+3:b+9]))
  past=physical(past[valid]);target=physical(target[valid]);cols=[]
  for b in [0,29]:
   cols.extend([np.linalg.norm(past[:,b:b+3]-target[:,b:b+3],axis=1)*1000,
                rotation_distance_deg(past[:,b+3:b+9],target[:,b+3:b+9]),
                abs(past[:,b+9:b+29]-target[:,b+9:b+29]).max(1)])
  values[k].append(np.stack(cols,axis=1))
result={}
names=['left_xyz_mm','left_rot_deg','left_hand_max_rad','right_xyz_mm','right_rot_deg','right_hand_max_rad']
for k,parts in values.items():
 a=np.concatenate(parts);result[k]=dict(n=len(a),**{n:dict(zip(['p50','p90','p99'],np.percentile(a[:,i],[50,90,99]).tolist())) for i,n in enumerate(names)})
out=root/'outputs/rtc_history_paired_20261009/longer_history_coverage.json';out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
