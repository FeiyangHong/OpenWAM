from pathlib import Path
import json
import numpy as np
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2
root=Path('/gaozt-test1/fyhong/OpenWAM')
path=root/'wandb/run-20261008_235538-mqbewlsq/run-mqbewlsq.wandb'
stream=datastore.DataStore();stream.open_for_scan(str(path));rows=[]
while True:
 raw=stream.scan_data()
 if raw is None:break
 rec=wandb_internal_pb2.Record();rec.ParseFromString(raw)
 if not rec.HasField('history'):continue
 row={}
 for it in rec.history.item:
  key=it.key or '/'.join(it.nested_key)
  if key.startswith('rtc_history/') or key in ['train/global_step','global_step','_step']:
   row[key]=json.loads(it.value_json)
 if any(k.startswith('rtc_history/') for k in row):rows.append(row)
keys=sorted({k for r in rows for k in r if k.startswith('rtc_history/')})
means={k:float(np.mean([r[k] for r in rows if k in r])) for k in keys}
result=dict(history_records=len(rows),mean_fraction_all_samples=means,
 selected_fraction_nonempty=means['rtc_history/selected_fraction']/means['rtc_history/prefix_fraction'])
(root/'outputs/rtc_history_paired_20261009/training_metrics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
