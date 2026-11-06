"""Recompute the bounded operation-speed claim from retained raw measurements."""
import json
from pathlib import Path
import statistics
from benchmark_statistics import paired_abba

ROOT=Path(__file__).resolve().parents[1]
RUNS=['q4-gate-order','q4-gate-order-repeat','q4-pr-proof']


def verify_run(directory):
    summary=json.loads((directory/'summary.json').read_text())
    if summary['status']!='passed' or summary['variant']!='gate-order':
        raise ValueError('Require a completed gate-order run')
    records={name:json.loads((directory/(name+'.json')).read_text()) for name in ['control','comparison']}
    for record in records.values():
        if record['correct'] is not True or record['red_zones']!='intact' or record['source_rows']!=9728:
            raise ValueError('Correctness or complete-row qualification failed')
        if record['baseline_kernel_nodes']!=2 or record['mojo_kernel_nodes']!=2:
            raise ValueError('Unmatched operation graphs')
    if records['control']['same_graph_control'] is not True or records['comparison']['same_graph_control'] is not False:
        raise ValueError('Incorrect control labeling')
    clocks={(x['sm_clock_before_mhz'],x['sm_clock_after_mhz'],x['memory_clock_before_mhz'],x['memory_clock_after_mhz'])
            for x in records['comparison']['samples']}
    if len(clocks)!=1 or any(a!=b or c!=d for a,b,c,d in clocks):
        raise ValueError('Unmatched sampled clocks')
    control=paired_abba(records['control']['samples'],'ggml_control')
    paired=paired_abba(records['comparison']['samples'],'mojo')
    def median(record,mode):
        return statistics.median(x['gpu_us_per_operation'] for x in record['samples'] if x['mode']==mode)
    a=median(records['control'],'ggml'); b=median(records['control'],'ggml_control')
    if abs(a/b-1)>.05:
        raise ValueError('Same-graph control exceeds 5%')
    cuda=median(records['comparison'],'ggml'); mojo=median(records['comparison'],'mojo')
    return summary, {'run':directory.name,'cuda_median_us':cuda,'mojo_median_us':mojo,
        'median_speed_ratio':cuda/mojo,'control_ratio':a/b,
        'paired_round_ratios':paired['baseline_over_candidate_round_ratios'],
        'paired_geometric_mean':paired['geometric_mean_ratio'],
        'control_paired_geometric_mean':control['geometric_mean_ratio']}


def report():
    rows=[]; identity=None
    for name in RUNS:
        summary,row=verify_run(ROOT/'results/a2000'/name)
        current=[summary[k] for k in ['fixture_sha256','ptx_sha256','benchmark_sha256','versions']]
        if identity is not None and current!=identity:
            raise ValueError('Runs do not use identical artifacts and fixtures')
        identity=current; rows.append(row)
    ratios=[v for row in rows for v in row['paired_round_ratios']]
    return {'scope':'Isolated fused gate/up plus matched activation quantization; not whole-model throughput.',
        'runs':rows,'paired_rounds':len(ratios),'rounds_favoring_mojo':sum(x>1 for x in ratios),
        'minimum_round_ratio':min(ratios),
        'limitation':'Within-run rounds are correlated; no independence assumption or universal speed guarantee.'}


if __name__=='__main__':
    print(json.dumps(report(),indent=2))
