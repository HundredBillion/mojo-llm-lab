"""Run bounded, temperature-guarded GPU microbenchmarks on synthetic data."""
import argparse
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess as S
import time
import uuid

from validate_ptx import validate_ptx

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


class Sensors:
    def __init__(self, gpu):
        self.cpu = [t for hw in Path('/sys/class/hwmon').glob('hwmon*')
                    if (hw / 'name').read_text().strip() in {'coretemp', 'k10temp', 'zenpower'}
                    for t in hw.glob('temp*_input')]
        if not self.cpu:
            raise RuntimeError('No supported CPU temperature sensor; refusing an unguarded run')
        self.nv = C.CDLL('libnvidia-ml.so.1')
        if self.nv.nvmlInit_v2():
            raise RuntimeError('NVML initialization failed')
        self.device = C.c_void_p()
        self.nv.nvmlDeviceGetHandleByIndex_v2.argtypes = [C.c_uint, C.POINTER(C.c_void_p)]
        if self.nv.nvmlDeviceGetHandleByIndex_v2(gpu, C.byref(self.device)):
            raise RuntimeError('Requested GPU unavailable')
        self.nv.nvmlDeviceGetTemperature.argtypes = [C.c_void_p, C.c_uint, C.POINTER(C.c_uint)]

    def read(self):
        temperature = C.c_uint()
        if self.nv.nvmlDeviceGetTemperature(self.device, 0, C.byref(temperature)):
            raise RuntimeError('GPU temperature sensor unavailable')
        supplies = list(Path('/sys/class/power_supply').iterdir())
        battery = any((p / 'type').read_text().strip() == 'Battery' for p in supplies)
        ac = not battery or any((p / 'online').exists() and (p / 'online').read_text().strip() == '1'
                                for p in supplies if (p / 'type').read_text().strip() != 'Battery')
        platform = [int((z / 'temp').read_text()) / 1000 for z in Path('/sys/class/thermal').glob('thermal_zone*')
                    if (z / 'type').read_text().strip().startswith('SEN')]
        return {'cpu_C': max(int(t.read_text()) / 1000 for t in self.cpu),
                'gpu_C': temperature.value, 'platform_C': max(platform, default=0), 'ac': ac}


def safe(sample):
    return sample['ac'] and sample['cpu_C'] < 90 and sample['gpu_C'] < 82 and sample['platform_C'] < 75


def cold_gate(sensors):
    deadline, stable = time.monotonic() + 60, None
    while time.monotonic() < deadline:
        sample = sensors.read()
        if not safe(sample):
            raise RuntimeError('Power/temperature preflight failed: ' + json.dumps(sample))
        if sample['cpu_C'] < 70 and sample['gpu_C'] < 68:
            stable = stable or time.monotonic()
            if time.monotonic() - stable >= 5:
                return
        else:
            stable = None
        time.sleep(.25)
    raise RuntimeError('No cold window within 60 seconds')


def remove_owned(name, token):
    inspect = S.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=10)
    if inspect.returncode:
        return
    resource = json.loads(inspect.stdout)[0]
    if (resource['Config'].get('Labels') or {}).get('io.mojo-imrope.benchmark') != token:
        raise RuntimeError('Container ownership changed; refusing removal')
    S.run(['docker', 'rm', '-f', name], check=True, capture_output=True, timeout=15)


def run_case(args, sensors, versions, ptx, entry, label, case, control, *,
             benchmark_dir=None, fixture_dir=None, executable="bench", input_path=None):
    benchmark_dir = benchmark_dir or ROOT / "build/benchmark"
    fixture_dir = fixture_dir or ROOT / "build/fixtures"
    cold_gate(sensors)
    token = uuid.uuid4().hex
    name = 'mojo-imrope-' + token
    out = args.output
    command = ['docker', 'create', '--name', name, '--label', 'io.mojo-imrope.benchmark=' + token,
               '--device', f'nvidia.com/gpu={args.gpu}', '--network', 'none', '--read-only',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--user', f'{os.getuid()}:{os.getgid()}',
               '--cpu-period', '5000', '--cpu-quota', '1000', '--memory', '1g', '--memory-swap', '1g',
               '-e', 'LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64',
               '--mount', f'type=bind,src={benchmark_dir},dst=/benchmark,readonly',
               '--mount', f'type=bind,src={fixture_dir},dst=/fixtures,readonly',
               '--mount', f'type=bind,src={ptx.parent},dst=/mojo,readonly',
               '--mount', f'type=bind,src={ROOT / "build/upstream" / ("llama-" + versions["engine_release"])},dst=/app,readonly',
               '--entrypoint', '/benchmark/' + executable, versions['cuda_image'],
               input_path or f'/fixtures/{case}.input.bin', '/mojo/' + ptx.name, entry]
    if control:
        command.append('--same-graph-control')
    samples = []
    process = None
    try:
        S.run(command, check=True, capture_output=True, timeout=30)
        with (out / f'{label}.stdout').open('w') as stdout, (out / f'{label}.stderr').open('w') as stderr:
            process = S.Popen(['docker', 'start', '-a', name], stdout=stdout, stderr=stderr)
            deadline = time.monotonic() + 60
            while process.poll() is None:
                sample = sensors.read()
                samples.append(sample)
                if not safe(sample) or time.monotonic() >= deadline:
                    raise RuntimeError('Power/temperature/duration guard stopped owned work: ' + json.dumps(sample))
                time.sleep(.1)
        if process.returncode:
            raise RuntimeError(f'{label} failed; see {label}.stderr')
        result = json.loads((out / f'{label}.stdout').read_text())
        if result.get('correct') is not True or result.get('red_zones') != 'intact' or len(result['samples']) != 32:
            raise RuntimeError('Incomplete or incorrect benchmark result')
        (out / f'{label}.json').write_text(json.dumps(result, indent=2) + '\n')
        medians = {mode: statistics.median(s['gpu_us_per_operation'] for s in result['samples'] if s['mode'] == mode)
                   for mode in {s['mode'] for s in result['samples']}}
        ratio = medians['ggml'] / medians['ggml_control' if control else 'mojo']
        if control and not .95 <= ratio <= 1.05:
            raise RuntimeError('Same-graph control differs by over 5%; do not interpret a speedup')
        return {'case': case, 'control': control, 'gpu_medians_us': medians,
                'baseline_over_candidate_ratio': ratio,
                'kernel_nodes': [result['baseline_kernel_nodes'], result['mojo_kernel_nodes']],
                'observed_gpu_clocks_MHz': sorted({s['gpu_clock_mhz'] for s in result['samples']}),
                'peaks': {key: max(s[key] for s in samples) for key in ('cpu_C', 'gpu_C', 'platform_C')}}
    finally:
        remove_owned(name, token)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except S.TimeoutExpired:
                process.kill()
                process.wait()
        (out / f'{label}-sensors.json').write_text(json.dumps(samples, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/local')
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('GPU index must be nonnegative')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    versions = json.loads((ROOT / 'versions.json').read_text())
    ptxs = list((ROOT / 'build/mojo').glob('*.ptx'))
    if len(ptxs) != 1:
        raise RuntimeError('Build exactly one Mojo PTX artifact first')
    ptx = ptxs[0]
    entry = validate_ptx(ptx.read_text())
    busy = S.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if busy:
        raise RuntimeError('GPU compute is already active; stop it before benchmarking')
    sensors = Sensors(args.gpu)
    result = {'started_utc': datetime.now(timezone.utc).isoformat(), 'versions': versions,
              'ptx_sha256': digest(ptx), 'benchmark_sha256': digest(ROOT / 'build/benchmark/bench'),
              'fixture_sha256': {p.name: digest(p) for p in (ROOT / 'build/fixtures').glob('*.input.bin')},
              'environment': S.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip(),
              'comparisons': [], 'status': 'running', 'scope': 'Cached single-token Q branch microbenchmark; not whole-LLM speed or energy.'}
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        for label, case, control in [('control', 'long_position', True), ('distinct', 'distinct_axes', False),
                                     ('long', 'long_position', False), ('fallback', 'fallback_only', False)]:
            row = run_case(args, sensors, versions, ptx, entry, label, case, control)
            result['comparisons'].append(row)
            print(json.dumps(row), flush=True)
        result['status'] = 'passed'
    except BaseException:
        result['status'] = 'failed_or_stopped'
        raise
    finally:
        sensors.nv.nvmlShutdown()
        (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
