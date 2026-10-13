"""Summarize paired prefix-length sensitivity; cluster uncertainty by episode."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np


def reverse(r):
    c=r['metrics']['right']['cosine']
    return c is not None and c<0


def large(r):
    return reverse(r) and r['metrics']['right']['after_mm']>3


def summarize(rows):
    index={(r['case'],r['seed'],r['design'],r['prefix_mode'],r['d'],r['state_backshift_mm']):r for r in rows}
    assert len(index)==len(rows)
    groups=collections.defaultdict(list)
    for r in rows:groups[f"{r['design']}|{r['prefix_mode']}|d{r['d']}|shift{r['state_backshift_mm']}"].append(r)
    result={};pairs={}
    for key,rr in sorted(groups.items()):
        m=[r['metrics']['right'] for r in rr]
        result[key]=dict(n=len(rr),reverse=sum(reverse(r) for r in rr),large_reverse=sum(large(r) for r in rr),
            reverse_amplitude_median_mm=float(np.median([r['metrics']['right']['after_mm'] for r in rr if reverse(r)])) if any(reverse(r) for r in rr) else None,
            seam_amplitude_median_mm=float(np.median([x['after_mm'] for x in m])),
            suffix_error_median_mm=float(np.median([x['suffix_error_mm'] for x in m])),
            velocity_change_median_mm=float(np.median([x['seam_velocity_change_mm_per_tick'] for x in m])),
            state_to_first_median_mm=float(np.median([x['state_to_first_mm'] for x in m])),
            state_to_end_median_mm=float(np.median([x['state_to_end_mm'] for x in m])),
            strong_both_n=sum(x['before_mm']>3 and x['after_mm']>3 for x in m),
            strong_both_reverse=sum(reverse(r) and r['metrics']['right']['before_mm']>3 and r['metrics']['right']['after_mm']>3 for r in rr),
            by_speed={g:dict(n=sum(r['group']==g for r in rr),large_reverse=sum(large(r) for r in rr if r['group']==g),reverse=sum(reverse(r) for r in rr if r['group']==g)) for g in ['slow','medium','fast']})
        if rr[0]['d']==10:continue
        pp=[]
        for r in rr:
            b=index[r['case'],r['seed'],'fixed_handoff',r['prefix_mode'],10,r['state_backshift_mm']]
            pp.append(dict(episode=r['episode'],case=r['case'],seed=r['seed'],large_delta=int(large(r))-int(large(b)),
                reverse_delta=int(reverse(r))-int(reverse(b)),
                suffix_error_delta_mm=r['metrics']['right']['suffix_error_mm']-b['metrics']['right']['suffix_error_mm'],
                first_action_change_mm=float(np.linalg.norm(np.array(r['actions'])[r['d'],29:32]-np.array(b['actions'])[10,29:32])*1000),
                reverse_resolved=reverse(b) and not reverse(r),reverse_introduced=not reverse(b) and reverse(r),
                large_resolved=large(b) and not large(r),large_introduced=not large(b) and large(r)))
        episodes=sorted(set(p['episode'] for p in pp));rng=np.random.default_rng(20260929)
        sums=np.array([[sum(p['large_delta'] for p in pp if p['episode']==e),sum(p['episode']==e for p in pp)] for e in episodes])
        draws=rng.integers(0,len(episodes),(5000,len(episodes)));totals=sums[draws].sum(axis=1)
        pairs[key]=dict(large_rate_difference=float(np.mean([p['large_delta'] for p in pp])),
            episode_bootstrap_95ci=np.percentile(totals[:,0]/totals[:,1],[2.5,97.5]).tolist(),
            reverse_resolved=sum(p['reverse_resolved'] for p in pp),reverse_introduced=sum(p['reverse_introduced'] for p in pp),
            large_resolved=sum(p['large_resolved'] for p in pp),large_introduced=sum(p['large_introduced'] for p in pp),
            paired_suffix_error_delta_median_mm=float(np.median([p['suffix_error_delta_mm'] for p in pp])),
            paired_first_action_change_median_mm=float(np.median([p['first_action_change_mm'] for p in pp])),episodes=len(episodes))
    return dict(n=len(rows),groups=result,paired_vs_d10=pairs)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
    rows=[json.loads(line) for p in sorted(args.root.glob('shard*/results.jsonl')) for line in p.open()]
    summary=summarize(rows);(args.root/'summary.json').write_text(json.dumps(summary,indent=2))
    for k,v in summary['groups'].items():print(k,json.dumps(v))
