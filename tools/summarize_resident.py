"""Recompute resident trial ratios without converting them into a speedup claim."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(path):
    result=json.loads(path.read_text())
    if result['status']!='passed':raise ValueError('Incomplete/failed run')
    rows=result['trials'];cycles=result.get('cycles',2);tokens=result.get('tokens',32)
    if [r['arm'] for r in rows]!=[0,1]+[0,1,1,0]*cycles or [r['warmup'] for r in rows]!=[True,True]+[False]*(4*cycles):
        raise ValueError('Unexpected trial protocol')
    rows=rows[2:]
    medians={str(arm):statistics.median(tokens*1e9/r['decode_ns'] for r in rows if r['arm']==arm) for arm in (0,1)}
    ratios=[(rows[i]['decode_ns']+rows[i+3]['decode_ns'])/(rows[i+1]['decode_ns']+rows[i+2]['decode_ns']) for i in range(0,4*cycles,4)]
    return {'mode':result['mode'],'median_tokens_per_second':medians,
            'arm1_over_arm0_median_rate':medians['1']/medians['0'],
            'cycle_latency_ratios':ratios,
            'median_reported_throttle_usec_over_wall_usec':statistics.median(r['throttled_usec']*1000/r['decode_ns'] for r in rows),
            'scope':'Fixed-token decode only; cgroup throttle counter is not a causal critical-path attribution.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summaries',nargs='+',type=Path)
    a=p.parse_args();print(json.dumps([summarize(path) for path in a.summaries],indent=2))


if __name__=='__main__':main()
