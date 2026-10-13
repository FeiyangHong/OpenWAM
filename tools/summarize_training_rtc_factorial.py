"""Summarize paired factorial experiments; resample windows, not individual seeds."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def key(r):
    return f"{r['prefix_mode']}|image{r['image_offset']}|state{r['state_backshift_mm']}|{r['state_mode']}"


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[key(r)].append(r)
    def reverse(r):
        m=r['metrics']['right']
        return m['cosine'] is not None and m['cosine']<0 and min(m['before_mm'],m['after_mm'])>0.1
    result={}
    for k,rs in groups.items():
        ms=[r['metrics']['right'] for r in rs]
        rev=[r for r in rs if reverse(r)]
        def med(field):return float(np.median([m[field] for m in ms]))
        strong=[m for m in ms if min(m['before_mm'],m['after_mm'])>3]
        result[k]=dict(n=len(rs),defined=sum(min(m['before_mm'],m['after_mm'])>0.1 for m in ms),
            reverse=len(rev),reverse_step_median_mm=float(np.median([r['metrics']['right']['after_mm'] for r in rev])) if rev else None,
            reverse_over3mm=sum(r['metrics']['right']['after_mm']>3 for r in rev),
            strong_defined=len(strong),strong_reverse=sum(m['cosine']<0 for m in strong),
            outgoing_median_mm=med('after_mm'),first_gt_error_median_mm=med('first_error_mm'),
            suffix_gt_error_median_mm=med('suffix_error_mm'),prefix_gt_error_median_mm=med('prefix_error_mm'),
            paired_first_delta_median_mm=float(np.median([r['paired_first_delta_mm'] for r in rs])),
            state_first_median_mm=med('state_to_first_mm'),state_end_median_mm=med('state_to_end_mm'),
            reverse_by_speed={g:sum(reverse(r) for r in rs if r['group']==g) for g in ['slow','medium','fast']},
            return_prefix_max_error_mm=max(m['returned_prefix_xyz_error_mm'] for m in ms))
    paired={}
    baseline_keys=['teacher|image0|state0|xyz_shift','model|image0|state0|xyz_shift']
    rng=np.random.default_rng(20260929)
    for k,rs in groups.items():
        basekey=baseline_keys[int(k.startswith('model'))]
        bases={(r['case'],r['seed']):r for r in groups[basekey]}
        windows=defaultdict(list)
        first_deltas=[]; gains=[]
        for r in rs:
            b=bases.get((r['case'],r['seed']))
            if b is None:continue
            windows[r['case']].append(int(reverse(r))-int(reverse(b)))
            da=np.array(r['actions'][10][29:32])-np.array(b['actions'][10][29:32])
            ds=np.array(r['state'][29:32])-np.array(b['state'][29:32])
            first_deltas.append(float(np.linalg.norm(da)*1000))
            if ds@ds>1e-12:gains.append(float((da@ds)/(ds@ds)))
        values=np.array([np.mean(v) for v in windows.values()])
        boot=np.mean(rng.choice(values,size=(5000,len(values)),replace=True),axis=1)
        paired[k]=dict(reverse_rate_difference=float(values.mean()),window_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
            first_action_delta_median_mm=float(np.median(first_deltas)),
            state_to_output_projection_gain_median=float(np.median(gains)) if gains else None)
    contrasts={}
    pairs=[('teacher|image0|state0|xyz_shift','model|image0|state0|xyz_shift'),
           ('teacher|image-2|state0|xyz_shift','teacher|image-2|state0|recorded_state_minus2'),
           ('model|image-2|state0|xyz_shift','model|image-2|state0|recorded_state_minus2')]
    for basekey,testkey in pairs:
        if basekey not in groups or testkey not in groups:continue
        bases={(r['case'],r['seed']):r for r in groups[basekey]}
        windows=defaultdict(list)
        for r in groups[testkey]:
            b=bases.get((r['case'],r['seed']))
            if b is not None:windows[r['case']].append(int(reverse(r))-int(reverse(b)))
        values=np.array([np.mean(v) for v in windows.values()])
        boot=np.mean(rng.choice(values,size=(5000,len(values)),replace=True),axis=1)
        contrasts[basekey+' -> '+testkey]=dict(reverse_rate_difference=float(values.mean()),window_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist())
    return dict(rows=len(rows),groups=result,paired_vs_same_prefix_baseline=paired,contrasts=contrasts)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args()
    rows=[]
    for line in (args.run/'results.jsonl').read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:break
    summary=summarize(rows)
    (args.run/'summary.json').write_text(json.dumps(summary,indent=2))
    for k,s in summary['groups'].items():
        print(k,'n',s['n'],'reverse',s['reverse'],'>3mm',s['reverse_over3mm'],'revstep',s['reverse_step_median_mm'],'error',s['suffix_gt_error_median_mm'])
