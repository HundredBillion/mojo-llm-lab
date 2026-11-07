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
    if any(r.get("tokens")!=tokens or type(r.get("decode_ns")) is not int or r["decode_ns"]<=0 for r in rows):
        raise ValueError("Trial workload or timing mismatch")
    if "mojo_arm" not in result and any("mojo_arm" in r for r in rows):
        raise ValueError("Partial candidate context metadata")
    mojo_arm=result.get("mojo_arm",1)
    if type(mojo_arm) is not int or mojo_arm not in (0,1):raise ValueError("Invalid candidate context")
    if "mojo_arm" in result and any(type(r.get("mojo_arm")) is not int or r["mojo_arm"]!=mojo_arm for r in rows):
        raise ValueError("Candidate context metadata mismatch")
    rows=rows[2:]
    medians={str(arm):statistics.median(tokens*1e9/r['decode_ns'] for r in rows if r['arm']==arm) for arm in (0,1)}
    ratios=[(rows[i]['decode_ns']+rows[i+3]['decode_ns'])/(rows[i+1]['decode_ns']+rows[i+2]['decode_ns']) for i in range(0,4*cycles,4)]
    return {'mojo_arm':mojo_arm,'mode':result['mode'],'protocol':result.get('protocol'),
            'cuda_wait_verified':result.get('cuda_wait_verified'),
            'prime_tokens':result.get('prime_tokens',int(result.get('prime_decode',False))),
            'cpu_quota_us':result.get('cpu_quota_us',1000),
            'cpu_period_us':result.get('cpu_period_us',5000),'median_tokens_per_second':medians,
            'arm1_over_arm0_median_rate':medians['1']/medians['0'],
            'cycle_latency_ratios':ratios,
            'cuda_over_mojo_latency_ratios':([r if mojo_arm==1 else 1/r for r in ratios] if result['mode']=='mojo' else None),
            'median_reported_throttle_usec_over_wall_usec':statistics.median(r['throttled_usec']*1000/r['decode_ns'] for r in rows),
            'scope':'Fixed-token decode only; cgroup throttle counter is not a causal critical-path attribution.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summaries',nargs='+',type=Path)
    a=p.parse_args();print(json.dumps([summarize(path) for path in a.summaries],indent=2))


if __name__=='__main__':main()
