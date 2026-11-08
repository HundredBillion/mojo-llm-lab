"""Guarded CUDA prompt-processing (prefill) measurement with pinned llama-bench, optionally under Nsight Systems.

Reports llama-bench prompt-processing throughput for one configuration. With --nsys, the run is a kernel-attribution
capture only: profiler overhead invalidates its throughput.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess as S
import time
import uuid

from run_benchmark import cold_gate, safe, digest
from run_launch_observer import ModelSensors
from run_pdf_smoke import cleanup
from cuda_wait_policy import verify_wait_policy

ROOT = Path(__file__).resolve().parents[1]


def bench_args(prompt_tokens, batch, ubatch, repetitions):
    return ['-m', '/model.gguf', '-p', str(prompt_tokens), '-n', '0', '-r', str(repetitions), '-t', '2', '--poll', '0',
            '-ngl', '99', '-b', str(batch), '-ub', str(ubatch), '-ctk', 'q4_0', '-ctv', 'q4_0', '-fa', 'on', '-o', 'json']


def parse_bench(stdout, prompt_tokens, batch, ubatch, repetitions):
    start = stdout.find('[')
    if start < 0:
        raise ValueError('Missing llama-bench JSON')
    rows = json.loads(stdout[start:])
    if len(rows) != 1:
        raise ValueError('Expected one llama-bench result')
    row = rows[0]
    expected = {'n_prompt': prompt_tokens, 'n_gen': 0, 'n_batch': batch, 'n_ubatch': ubatch,
                'n_gpu_layers': 99, 'type_k': 'q4_0', 'type_v': 'q4_0'}
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError('llama-bench setting mismatch: ' + key)
    samples = row.get('samples_ts')
    if type(samples) is not list or len(samples) != repetitions:
        raise ValueError('Unexpected repetition count')
    if not all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples):
        raise ValueError('Invalid throughput sample')
    return {'prompt_tokens': prompt_tokens, 'batch': batch, 'ubatch': ubatch, 'repetitions': repetitions,
            'samples_tokens_per_second': samples, 'mean_tokens_per_second': row['avg_ts'],
            'stddev_tokens_per_second': row['stddev_ts'], 'build_commit': row.get('build_commit')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--wait-library', type=Path, required=True)
    p.add_argument('--prompt-tokens', type=int, choices=[512, 2048, 6144], required=True)
    p.add_argument('--batch', type=int, choices=[32, 128, 512], default=32)
    p.add_argument('--ubatch', type=int, choices=[32, 128, 512], default=32)
    p.add_argument('--repetitions', type=int, choices=[1, 3], default=3)
    p.add_argument('--nsys', type=Path, help='Nsight Systems installation root (kernel attribution only)')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.batch < a.ubatch:
        p.error('--batch must be at least --ubatch')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    versions = json.loads((ROOT / 'versions.json').read_text())
    if S.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU already in use')
    sensor = ModelSensors(0); samples = []
    read_sensor = sensor.read

    def recorded_read():
        sample = read_sensor(); samples.append(sample); return sample
    sensor.read = recorded_read
    name = 'mojo-prefill-' + uuid.uuid4().hex; token = uuid.uuid4().hex
    entry = '/nsys/target-linux-x64/nsys' if a.nsys else '/app/llama-bench'
    command = ['docker', 'run', '--rm', '--name', name, '--label', 'io.mojo-imrope.benchmark=' + token,
               '--device', 'nvidia.com/gpu=0', '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'HOME=/tmp',
               '--network', 'none', '--read-only', '--tmpfs', '/tmp:rw,size=512m', '--cap-drop', 'ALL',
               '--security-opt', 'no-new-privileges', '--cpu-period', '5000', '--cpu-quota', '1000',
               '--memory', '12g', '--memory-swap', '12g', '--entrypoint', entry,
               '-e', 'LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64', '-e', 'LD_PRELOAD=/wait.so',
               '-e', 'LLAMA_CUDA_WAIT=blocking', '-e', 'LLAMA_CUDA_WAIT_TARGET=/app/llama-bench']
    mounts = [(ROOT / 'build/upstream/llama-b12110', '/app', True), (a.model, '/model.gguf', True),
              (a.wait_library, '/wait.so', True)]
    if a.nsys:
        mounts += [(a.nsys, '/nsys', True), (out, '/profile', False)]
    for src, dst, readonly in mounts:
        command += ['--mount', f'type=bind,src={src.resolve()},dst={dst}' + (',readonly' if readonly else '')]
    command.append(versions['cuda_image'])
    if a.nsys:
        command += ['profile', '--trace=cuda,nvtx', '--cuda-graph-trace=node', '--sample=none', '--cpuctxsw=none',
                    '--output=/profile/prefill', '/app/llama-bench']
    command += bench_args(a.prompt_tokens, a.batch, a.ubatch, a.repetitions)
    result = {'status': 'failed_or_stopped', 'protocol': 'prefill-bench-v1', 'versions': versions,
              'mode': 'nsys-attribution' if a.nsys else 'throughput',
              'model_name': a.model.name, 'model_sha256': digest(a.model), 'wait_sha256': digest(a.wait_library),
              'llama_bench_sha256': digest(ROOT / 'build/upstream/llama-b12110/llama-bench'),
              'cpu_period_us': 5000, 'cpu_quota_us': 1000, 'cuda_wait_requested': 'blocking',
              'scope': 'CUDA-only prompt processing (prefill). The Mojo candidate is not loaded. '
                       + ('Profiler overhead invalidates throughput.' if a.nsys else
                          'Synthetic llama-bench prompts; not PDF latency or sampled output.')}
    process = None; start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        cold_gate(sensor)
        with (out / 'stdout.log').open('w') as stdout, (out / 'stderr.log').open('w') as stderr:
            process = S.Popen(command, stdout=stdout, stderr=stderr)
            while process.poll() is None:
                if not safe(sensor.read()):
                    raise RuntimeError('Temperature or AC guard')
                if time.monotonic() - start > 240:
                    raise RuntimeError('Run deadline')
                time.sleep(.1)
        if process.returncode:
            raise RuntimeError('llama-bench exit %d' % process.returncode)
        result['cuda_wait_verified'] = verify_wait_policy((out / 'stderr.log').read_text(), 'blocking')
        if a.nsys:
            if not (out / 'prefill.nsys-rep').is_file():
                raise RuntimeError('Missing profiler trace artifact')
            result['nsys_report_sha256'] = digest(out / 'prefill.nsys-rep')
        else:
            result['prefill'] = parse_bench((out / 'stdout.log').read_text(), a.prompt_tokens, a.batch, a.ubatch,
                                            a.repetitions)
        result['status'] = 'passed'
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        errors = cleanup(name, token, [process])
        if errors:
            result.update(status='failed_or_stopped', cleanup_errors=errors)
        result['peaks'] = {k: max(s[k] for s in samples) for k in ['cpu_C', 'gpu_C', 'platform_C']} if samples else {}
        clocks = [s['sm_clock_MHz'] for s in samples if s.get('sm_clock_MHz') is not None]
        result['max_sm_clock_MHz'] = max(clocks) if clocks else None
        (out / 'sensors.json').write_text(json.dumps(samples) + '\n')
        (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'error', 'peaks', 'prefill')}))
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
