"""Run the experimental Q4 comparison with CPU, temperature and AC guards."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
from run_benchmark import Sensors, cold_gate, safe

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/local-q4-comparison')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU compute busy; stop other GPU work first')
    sensors = Sensors(0)
    process = None
    unit = 'mojo-q4-' + uuid.uuid4().hex
    samples = []
    status = 'failed_or_stopped'
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        cold_gate(sensors)
        with (args.output/'worker.log').open('w') as log:
            process = subprocess.Popen([
                'systemd-run', '--user', '--scope', '--quiet', '--unit='+unit,
                '-p', 'CPUQuota=20%', '-p', 'CPUQuotaPeriodSec=5ms', '-p', 'MemoryMax=1G',
                '-p', 'RuntimeMaxSec=120', sys.executable, str(ROOT/'tools/q4_comparison_worker.py'),
                '--output', str(args.output/'timing.json')], stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic()+120
            while process.poll() is None:
                sample = sensors.read()
                samples.append(sample)
                if not safe(sample) or time.monotonic() >= deadline:
                    raise RuntimeError('Temperature/power/time guard stopped owned comparison')
                time.sleep(.1)
        if process.returncode:
            raise RuntimeError('Comparison failed; see worker.log')
        result = json.loads((args.output/'timing.json').read_text())
        if result['status'] != 'passed':
            raise RuntimeError('Incomplete comparison')
        status = 'passed'
        print(json.dumps({k:v for k,v in result.items() if k != 'samples'}))
    finally:
        subprocess.run(['systemctl', '--user', 'stop', unit+'.scope'], capture_output=True, timeout=15)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        sensors.nv.nvmlShutdown()
        (args.output/'guard.json').write_text(json.dumps({'status':status,'samples':samples},indent=2)+'\n')


if __name__ == '__main__':
    main()
