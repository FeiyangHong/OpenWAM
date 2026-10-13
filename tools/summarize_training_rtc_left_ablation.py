"""Summaries and episode-clustered paired contrasts for left-input ablations."""
from collections import defaultdict
from pathlib import Path
import argparse,json
import numpy as np


def key(r):return f"{r['prefix_mode']}|{r['mode']}|right{r['right_state_backshift_mm']}"


def reversed_row(r):
    m=r['metrics']['right']
    return m['cosine'] is not None and m['cosine']<0 and min(m['before_mm'],m['after_mm'])>.1


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[key(r)].append(r)
    summary={};paired={};rng=np.random.default_rng(20260930)
    for k,rs in groups.items():
        ms=[r['metrics']['right'] for r in rs];rev=[r for r in rs if reversed_row(r)]
        def med(field):return float(np.median([m[field] for m in ms]))
        strong=[m for m in ms if min(m['before_mm'],m['after_mm'])>3]
        summary[k]=dict(n=len(rs),defined=sum(min(m['before_mm'],m['after_mm'])>.1 for m in ms),
            reverse=len(rev),reverse_over3mm=sum(r['metrics']['right']['after_mm']>3 for r in rev),
            reverse_step_median_mm=float(np.median([r['metrics']['right']['after_mm'] for r in rev])) if rev else None,
            strong_defined=len(strong),strong_reverse=sum(m['cosine']<0 for m in strong),
            first_gt_error_median_mm=med('first_error_mm'),suffix_gt_error_median_mm=med('suffix_error_mm'),
            paired_first_delta_mm=dict(zip(['p50','p90','max'],np.percentile([r['paired_first_delta_mm'] for r in rs],[50,90,100]).tolist())),
            paired_suffix_delta_median_mm=float(np.median([r['paired_suffix_delta_mm'] for r in rs])),
            reverse_by_speed={g:sum(reversed_row(r) for r in rs if r['group']==g) for g in ['slow','medium','fast']},
            large_reverse_by_speed={g:sum(reversed_row(r) and r['metrics']['right']['after_mm']>3 for r in rs if r['group']==g) for g in ['slow','medium','fast']})
        basekey=f"{rs[0]['prefix_mode']}|baseline|right{rs[0]['right_state_backshift_mm']}"
        bases={(r['case'],r['seed']):r for r in groups[basekey]}
        episodes=defaultdict(list);error_changes=[]
        for r in rs:
            b=bases.get((r['case'],r['seed']))
            if b is None:continue
            episodes[r['episode']].append(int(reversed_row(r))-int(reversed_row(b)))
            error_changes.append(r['metrics']['right']['suffix_error_mm']-b['metrics']['right']['suffix_error_mm'])
        sums=np.array([np.sum(v) for v in episodes.values()]);counts=np.array([len(v) for v in episodes.values()])
        indices=rng.integers(0,len(sums),size=(5000,len(sums)))
        boot=sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
        paired[k]=dict(reverse_rate_difference=float(sums.sum()/counts.sum()),episode_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
            paired_suffix_gt_error_change_median_mm=float(np.median(error_changes)))
    return dict(rows=len(rows),groups=summary,paired_left_effect_at_same_right_shift=paired)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args()
    rows=[]
    for line in (args.run/'results.jsonl').open():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:break
    s=summarize(rows);(args.run/'summary.json').write_text(json.dumps(s,indent=2))
    for k,g in s['groups'].items():
        print(k,'n',g['n'],'reverse',g['reverse'],'>3mm',g['reverse_over3mm'],'paired_mm',round(g['paired_first_delta_mm']['p50'],3),'error_mm',round(g['suffix_gt_error_median_mm'],3))
