"""Recompute the preregistered fixed-token decode confirmation decision."""
import argparse
import json
import math
from pathlib import Path

from summarize_resident import summarize
from run_resident_decode import validate_trial


def decision(candidates, controls):
    if len(candidates) != 4 or len(controls) != 2:
        raise ValueError('Expected four candidates and two controls')
    if any(not math.isfinite(x) or x <= 0 for x in candidates + controls):
        raise ValueError('Invalid latency ratio')
    margin = min(map(math.log, candidates)) - max(abs(math.log(x)) for x in controls)
    return {'candidate_ratios': candidates, 'control_ratios': controls,
            'log_margin': margin, 'required_log_margin': 0.005,
            'passed': all(x > 1 for x in candidates) and margin >= 0.005}


def verify(root):
    schedule = [('control', 1), ('mojo', 0), ('mojo', 1),
                ('mojo', 1), ('mojo', 0), ('control', 1)]
    candidates, controls = [], []
    reference = None
    keys = ('versions', 'model_name', 'model_bytes', 'worker_sha256',
            'observer_sha256', 'ptx_sha256', 'wait_sha256', 'worker_capabilities')
    for index, (mode, arm) in enumerate(schedule):
        path = root / f'confirmation-{index:02d}-{mode}-arm{arm}' / 'summary.json'
        raw = json.loads(path.read_text())
        summary = summarize(path)
        expected = {'mode': mode, 'mojo_arm': arm, 'tokens': 128, 'cycles': 1,
                    'prime_tokens': 2, 'cpu_period_us': 5000, 'cpu_quota_us': 1000,
                    'protocol': 'resident-block-v2', 'cuda_wait_requested': 'blocking',
                    'cuda_wait_verified': {'device': 0, 'policy': 'blocking', 'flags': 4}}
        if any(raw.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Settings mismatch: {path}')
        identity = {k: raw[k] for k in keys}
        identity['worker_capabilities'] = dict(identity['worker_capabilities'])
        if identity['worker_capabilities'].pop('mojo_arm') != arm:
            raise ValueError('Worker context mismatch')
        if reference is None:
            reference = identity
        if identity != reference:
            raise ValueError(f'Artifact mismatch: {path}')
        previous = 0
        for row in raw['trials']:
            previous = validate_trial(row, row['arm'], previous, mode, 128, 2, arm)
            if row.get('top_token') != 11140 or row.get('prime_tokens') != 2:
                raise ValueError('Fixture output or priming mismatch')
            if row.get('timed_replacement_calls') != 0 or row.get('timed_captured_calls') != 0:
                raise ValueError('Timed host replacement/capture detected')
        ratios = summary['cuda_over_mojo_latency_ratios'] if mode == 'mojo' else summary['cycle_latency_ratios']
        (candidates if mode == 'mojo' else controls).append(ratios[0])
    return decision(candidates, controls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    result = verify(parser.parse_args().directory)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
