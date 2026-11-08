"""Run one guarded numerical-coverage process: full prompt prefill, free-running generation, full-logit comparison.

Numerical evidence only; timings from this worker are never performance evidence.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess as S
import time
import uuid

from run_benchmark import cold_gate, safe, digest
from run_launch_observer import ModelSensors, validate_model_ptx
from run_pdf_smoke import cleanup
from cuda_wait_policy import verify_wait_policy

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ('control', 'candidate', 'candidate', 'control', 'candidate', 'control')
ATOL, RTOL = 1e-4, 1e-5


def schedule_arms(mojo_arm):
    return [mojo_arm if role == 'candidate' else 1 - mojo_arm for role in SCHEDULE]


def validate_ready(row, policy, seed, steps, mojo_arm):
    if row.get('ready') is not True:
        raise ValueError('Missing ready handshake')
    expected = {'policy': policy, 'seed': seed, 'steps': steps, 'mojo_arm': mojo_arm}
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError('Worker capability mismatch: ' + key)
    for key in ('prompt_tokens', 'n_ctx', 'vocabulary'):
        if type(row.get(key)) is not int or row[key] <= 0:
            raise ValueError('Invalid worker metadata: ' + key)
    if row['n_ctx'] < row['prompt_tokens'] + steps or row['n_ctx'] > 8192:
        raise ValueError('Invalid context size')


def validate_trials(rows, mojo_arm, steps):
    arms = schedule_arms(mojo_arm)
    if len(rows) != len(arms):
        raise ValueError('Incomplete trial set')
    reference = None
    previous = 0
    for index, (row, arm) in enumerate(zip(rows, arms)):
        if row.get('trial') != index or row.get('arm') != arm or row.get('candidate') is not (arm == mojo_arm):
            raise ValueError('Unexpected trial order')
        tokens = row.get('tokens')
        if type(tokens) is not list or len(tokens) != steps or not all(type(t) is int and t >= 0 for t in tokens):
            raise ValueError('Invalid generated tokens')
        if type(row.get('first_eog')) is not int or not -1 <= row['first_eog'] < steps:
            raise ValueError('Invalid end-of-generation index')
        if type(row.get('text')) is not str:
            raise ValueError('Missing generated text')
        for key in ('replacement_calls', 'captured_calls', 'trial_replacement_calls', 'trial_captured_calls'):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError('Invalid counter: ' + key)
        if row['replacement_calls'] != previous + row['trial_replacement_calls']:
            raise ValueError('Inconsistent replacement counters')
        if row['captured_calls'] > row['replacement_calls'] or row['trial_captured_calls'] > row['trial_replacement_calls']:
            raise ValueError('Inconsistent capture counters')
        if arm != mojo_arm and row['trial_replacement_calls'] != 0:
            raise ValueError('CUDA context unexpectedly replaced kernels')
        previous = row['replacement_calls']
        if reference is None:
            reference = row
        elif tokens != reference['tokens'] or row['first_eog'] != reference['first_eog'] or row['text'] != reference['text']:
            raise ValueError('Generated output differs from CUDA reference at trial %d (first token index %d)'
                             % (index, next((i for i, (a, b) in enumerate(zip(tokens, reference['tokens'])) if a != b), steps)))
    if rows[-1]['captured_calls'] == 0:
        raise ValueError('Candidate graph capture missing')
    return {'tokens': reference['tokens'], 'first_eog': reference['first_eog'], 'text': reference['text'],
            'candidate_captured_calls': rows[-1]['captured_calls']}


def validate_report(stderr, steps, vocabulary):
    matches = re.findall(r'^mojo_logits (\{.*\})$', stderr, re.M)
    if len(matches) != 1:
        raise ValueError('Missing or ambiguous logit validation report')
    report = json.loads(matches[0])
    positions = steps + 1
    candidates = SCHEDULE.count('candidate')
    controls = SCHEDULE.count('control') - 1
    expected = {'vocabulary': vocabulary, 'positions': positions,
                'candidate_positions': candidates * positions, 'control_positions': controls * positions,
                'candidate_values': candidates * positions * vocabulary,
                'control_values': controls * positions * vocabulary}
    for name, value in expected.items():
        if type(report.get(name)) is not int or report[name] != value:
            raise ValueError('Incomplete logit validation: ' + name)
    if report.get('passed') is not True or report.get('atol') != ATOL or report.get('rtol') != RTOL:
        raise ValueError('Logit validation failed or tolerance changed')
    for role in ('candidate', 'control'):
        for suffix in ('violations', 'top_mismatches'):
            if report.get(role + '_' + suffix) != 0:
                raise ValueError('Numerical or top-token mismatch')
        for suffix in ('max_abs', 'max_tolerance_ratio'):
            value = report.get(role + '_' + suffix)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid numerical error')
        if report[role + '_max_tolerance_ratio'] > 1:
            raise ValueError('Numerical tolerance exceeded')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--ptx', type=Path, required=True)
    p.add_argument('--wait-library', type=Path, required=True)
    p.add_argument('--prompt', type=Path, required=True)
    p.add_argument('--policy', choices=['greedy', 'sampled'], required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, choices=[128], default=128)
    p.add_argument('--mojo-arm', type=int, choices=[0, 1], required=True)
    p.add_argument('--cpu-quota-us', type=int, choices=[1000, 2000], default=1000)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.policy == 'greedy' and a.seed:
        p.error('--seed applies only to sampled generation')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    versions = json.loads((ROOT / 'versions.json').read_text())
    entry = validate_model_ptx(a.ptx.read_text(), 'qwen35-2b')
    if S.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU already in use')
    worker = 'resident-coverage'
    sensor = ModelSensors(0); samples = []; rows = []
    read_sensor = sensor.read

    def recorded_read():
        sample = read_sensor(); samples.append(sample); return sample
    sensor.read = recorded_read
    name = 'mojo-coverage-' + uuid.uuid4().hex; token = uuid.uuid4().hex
    command = ['docker', 'run', '--rm', '-i', '--name', name, '--label', 'io.mojo-imrope.benchmark=' + token,
               '--device', 'nvidia.com/gpu=0', '--user', f'{os.getuid()}:{os.getgid()}',
               '--network', 'none', '--read-only', '--tmpfs', '/tmp:rw,size=512m', '--cap-drop', 'ALL',
               '--security-opt', 'no-new-privileges', '--cpu-period', '5000', '--cpu-quota', str(a.cpu_quota_us),
               '--memory', '12g', '--memory-swap', '12g', '--entrypoint', '/observer/' + worker,
               '-e', 'LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64',
               '-e', 'LD_PRELOAD=/observer/observe_launches.so:/wait.so', '-e', 'MOJO_SHAPE=qwen35-2b',
               '-e', 'LLAMA_CUDA_WAIT_TARGET=/observer/' + worker, '-e', 'LLAMA_CUDA_WAIT=blocking',
               '-e', 'MOJO_RESIDENT=1', '-e', 'MOJO_CANDIDATE_PTX=/candidate.ptx', '-e', 'MOJO_REPLACE_ENTRY=' + entry]
    for src, dst in [(ROOT / 'build/upstream/llama-b12110', '/app'), (ROOT / 'build/integration', '/observer'),
                     (a.model, '/model.gguf'), (a.ptx, '/candidate.ptx'), (a.wait_library, '/wait.so'),
                     (a.prompt, '/prompt.txt')]:
        command += ['--mount', f'type=bind,src={src.resolve()},dst={dst},readonly']
    command += [versions['cuda_image'], '/model.gguf', a.policy, str(a.seed), str(a.steps), str(a.mojo_arm)]
    result = {'status': 'failed_or_stopped', 'protocol': 'resident-logit-coverage-v1', 'validation_only': True,
              'versions': versions, 'policy': a.policy, 'seed': a.seed, 'steps': a.steps, 'mojo_arm': a.mojo_arm,
              'cpu_period_us': 5000, 'cpu_quota_us': a.cpu_quota_us, 'cuda_wait_requested': 'blocking',
              'prompt_name': a.prompt.name, 'prompt_sha256': digest(a.prompt),
              'model_name': a.model.name, 'model_bytes': a.model.stat().st_size, 'model_sha256': digest(a.model),
              'ptx_sha256': digest(a.ptx), 'worker_sha256': digest(ROOT / 'build/integration' / worker),
              'observer_sha256': digest(ROOT / 'build/integration/observe_launches.so'),
              'wait_sha256': digest(a.wait_library), 'trials': rows,
              'scope': 'Numerical-only: full-prompt prefill and free-running generation with full-logit comparison. '
                       'Synchronization and copies invalidate timing; not performance, PDF latency or quality evidence.'}
    process = None; start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        cold_gate(sensor)
        with (out / 'stderr.log').open('w') as err, (out / 'stdout.log').open('w') as raw:
            process = S.Popen(command, stdin=S.PIPE, stdout=S.PIPE, stderr=err, text=True, bufsize=1)
            sel = selectors.DefaultSelector(); sel.register(process.stdout, selectors.EVENT_READ)

            def receive(limit):
                deadline = time.monotonic() + limit
                while time.monotonic() < deadline and time.monotonic() - start < 420:
                    if not safe(sensor.read()):
                        raise RuntimeError('Temperature or AC guard')
                    if sel.select(.1):
                        line = process.stdout.readline()
                        if not line:
                            raise RuntimeError('Worker exited before result')
                        raw.write(line); raw.flush()
                        return json.loads(line)
                raise RuntimeError('Worker deadline')
            ready = receive(120)
            validate_ready(ready, a.policy, a.seed, a.steps, a.mojo_arm)
            result['worker_capabilities'] = ready
            for arm in schedule_arms(a.mojo_arm):
                process.stdin.write(f'{arm}\n'); process.stdin.flush()
                rows.append(receive(150))
            process.stdin.close(); process.wait(timeout=60)
        if process.returncode:
            raise RuntimeError('Worker failed')
        stderr = (out / 'stderr.log').read_text()
        result['cuda_wait_verified'] = verify_wait_policy(stderr, 'blocking')
        result['generation'] = validate_trials(rows, a.mojo_arm, a.steps)
        result['logit_validation'] = validate_report(stderr, a.steps, ready['vocabulary'])
        result['status'] = 'passed'
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        errors = cleanup(name, token, [process])
        if errors:
            result.update(status='failed_or_stopped', cleanup_errors=errors)
        result['peaks'] = {k: max(s[k] for s in samples) for k in ['cpu_C', 'gpu_C', 'platform_C']} if samples else {}
        (out / 'sensors.json').write_text(json.dumps(samples) + '\n')
        (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'error', 'peaks')}))
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
