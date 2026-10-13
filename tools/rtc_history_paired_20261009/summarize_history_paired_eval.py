"""Summarize paired inference; counts use common prefix activity eligibility."""
from pathlib import Path
import argparse,json,collections
import numpy as np

def q(x):
 a=np.asarray(x,float)
 return dict(zip(['p50','p90','mean','max'],[float(np.median(a)),float(np.percentile(a,90)),float(a.mean()),float(a.max())])) if len(a) else None
def summary(rows):
 out=dict(n=len(rows))
 for name in ['right_arm','right_hand','left_arm','left_hand']:
  ms=[r['metrics'][name] for r in rows];th=.1 if name.endswith('arm') else .003
  eligible=[m for m in ms if m.get('incoming',0)>th]
  rev=[m for m in eligible if m['reverse']]
  out[name]=dict(eligible=len(eligible),reverse=len(rev),reverse_pct=100*len(rev)/len(eligible) if eligible else None,
    reverse_over_threshold=sum(m['outgoing']>3 if name.endswith('arm') else m['max_component']>.05 for m in rev),
    outgoing=q([m['outgoing'] for m in ms if 'outgoing'in m]),max_component=q([m['max_component'] for m in ms if 'max_component'in m]),
    first_error=q([m['first_error'] for m in ms]),suffix_error=q([m['suffix_error'] for m in ms]),suffix_mae=q([m['suffix_mae'] for m in ms]),
    state_first_gap=q([m['state_first_gap'] for m in ms if 'state_first_gap'in m]),
    reverse_toward_state=sum((m['toward_state_cos'] or 0)>0 for m in rev))
 for side in ['left','right']:out[side+'_rotation_suffix_deg']=q([r['metrics'][side+'_rotation_suffix_deg'] for r in rows])
 return out
def key(r):return r['family'],r['case'],r['mode'],r['seed'],r.get('round',-1)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out
 data={m:[json.loads(x) for x in (out/m/'results.jsonl').open()] for m in ['old','new']}
 indexes={m:{key(r):r for r in data[m]} for m in data}
 assert indexes['old'].keys()==indexes['new'].keys()
 assert json.loads((out/'old/inputs.json').read_text())==json.loads((out/'new/inputs.json').read_text())
 comparisons={};groups={};inputchecks=0
 for m,rows in data.items():
  buckets=collections.defaultdict(list)
  for r in rows:
   name=r['family']+'/'+r['mode']
   if r['family']=='rollout':name+='/round'+str(r['round'])
   buckets[name].append(r)
  groups[m]={k:summary(v) for k,v in buckets.items()}
 for name in groups['old']:
  paired=[]
  for k,r in indexes['old'].items():
   this=r['family']+'/'+r['mode']+('/round'+str(r['round']) if r['family']=='rollout' else '')
   if this!=name:continue
   n=indexes['new'][k]
   if r['family']!='rollout':
    assert r['state']==n['state'] and r['prefix']==n['prefix'] and r['gt']==n['gt'];inputchecks+=1
   item={'case':r['case'],'seed':r['seed']}
   for part in ['right_arm','right_hand']:
    x,y=r['metrics'][part],n['metrics'][part]
    item[part]=dict(suffix_error_delta=y['suffix_error']-x['suffix_error'],
     first_error_delta=y['first_error']-x['first_error'])
    if 'reverse'in x:item[part].update(old_reverse=x['reverse'],new_reverse=y['reverse'],outgoing_delta=y['outgoing']-x['outgoing'])
   paired.append(item)
  comp={'n':len(paired)}
  for part in ['right_arm','right_hand']:
   xx=[r[part] for r in paired];comp[part]={
      'suffix_error_delta':q([x['suffix_error_delta'] for x in xx]),'suffix_error_improved_n':sum(x['suffix_error_delta']<0 for x in xx),
      'first_error_delta':q([x['first_error_delta'] for x in xx])}
   if 'old_reverse'in xx[0]:comp[part].update(
       reverse_resolved=sum(x['old_reverse'] and not x['new_reverse'] for x in xx),
       reverse_introduced=sum(not x['old_reverse'] and x['new_reverse'] for x in xx),
       outgoing_delta=q([x['outgoing_delta'] for x in xx]))
  comparisons[name]=comp
 sensitivity={}
 for m in ['old','new']:
  si={}
  for mode,part,sl,scale in [('arm-15mm','right_arm',slice(29,32),1000),('arm-30mm','right_arm',slice(29,32),1000),
      ('arm-45mm','right_arm',slice(29,32),1000),('hand-0.05','right_hand',slice(38,58),1),('hand-0.10','right_hand',slice(38,58),1)]:
   vals=[]
   for r in data[m]:
    if r['family']!='stress' or r['mode']!=mode:continue
    b=indexes[m]['teacher',r['case'],'clean',r['seed'],-1]
    delta=np.asarray(r['state'])[sl]-np.asarray(b['state'])[sl]
    change=np.asarray(r['actions'])[10,sl]-np.asarray(b['actions'])[10,sl]
    vals.append(dict(gain=float(change@delta/(delta@delta)) if delta@delta>1e-12 else 0,
      output_change=float(np.linalg.norm(change)*scale)))
   si[mode]={field:q([v[field] for v in vals]) for field in ['gain','output_change']}
  sensitivity[m]=si
 result=dict(rows={m:len(v) for m,v in data.items()},shared_input_pairs_checked=inputchecks,
  groups=groups,paired=comparisons,sensitivity=sensitivity,
  complete={m:json.loads((out/m/'complete.json').read_text()) for m in data})
 (out/'summary.json').write_text(json.dumps(result,indent=2))
 for name in groups['old']:
  print(name)
  for m in ['old','new']:
   g=groups[m][name]
   print(m,'n',g['n'],*[f"{p}:rev={g[p]['reverse']}/{g[p]['eligible']} seam50={g[p]['outgoing']['p50'] if g[p]['outgoing'] else None} err50={g[p]['suffix_error']['p50']:.5f}" for p in ['right_arm','right_hand']])
 print('SENSITIVITY',json.dumps(sensitivity))
if __name__=='__main__':main()
