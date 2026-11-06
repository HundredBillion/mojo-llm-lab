"""Observe a short model decode under guards; this is not a speed comparison."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess as S
import time
import uuid
from run_benchmark import ROOT, Sensors, cold_gate, safe, remove_owned, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--wait-library', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    observer = ROOT/'build/integration/observe_launches.so'
    versions = json.loads((ROOT/'versions.json').read_text())
    result = {'status':'failed_or_stopped', 'scope':'Host-launch observation only; includes capture/warmup and excludes graph replay counts. No throughput claim.',
              'versions':versions, 'observer_sha256':digest(observer), 'model_name':model.name,
              'model_bytes':model.stat().st_size, 'model_sha256':None,
              'cpu_quota_percent':20, 'generation_tokens':16, 'wait_library_sha256':None}
    if S.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU compute busy')
    sensors = Sensors(0)
    name = 'mojo-observe-'+uuid.uuid4().hex
    token = uuid.uuid4().hex
    command = ['docker','create','--name',name,'--label','io.mojo-imrope.benchmark='+token,
        '--device','nvidia.com/gpu=0','--user',f'{os.getuid()}:{os.getgid()}',
        '--network','none','--read-only','--tmpfs','/tmp:rw,size=512m',
        '--cpu-period','5000','--cpu-quota','1000','--memory','12g','--memory-swap','12g',
        '--cap-drop','ALL','--security-opt','no-new-privileges','-e','HOME=/tmp',
        '-e','LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64','--entrypoint','/app/llama-bench']
    mounts = [(ROOT/'build/upstream'/('llama-'+versions['engine_release']),'/app'),
              (observer.parent,'/observer'), (model,'/model.gguf')]
    preload = '/observer/observe_launches.so'
    if args.wait_library:
        wait = args.wait_library.resolve(strict=True)
        result['wait_library_sha256'] = digest(wait)
        mounts.append((wait,'/wait.so'))
        preload += ':/wait.so'
    for src, dst in mounts:
        command += ['--mount',f'type=bind,src={src},dst={dst},readonly']
    command += ['-e','LD_PRELOAD='+preload, versions['cuda_image'],
        '-m','/model.gguf','-p','0','-n','16','-r','1','-t','2','--poll','0',
        '-ngl','99','-b','32','-ub','32','-ctk','q4_0','-ctv','q4_0','-fa','on','-o','json']
    process = None
    samples = []
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        cold_gate(sensors)
        S.run(command,check=True,capture_output=True,timeout=30)
        with (out/'stdout.log').open('w') as stdout, (out/'stderr.log').open('w') as stderr:
            process = S.Popen(['docker','start','-a',name],stdout=stdout,stderr=stderr)
            deadline = time.monotonic()+120
            while process.poll() is None:
                sample = sensors.read(); samples.append(sample)
                if not safe(sample) or time.monotonic()>deadline:
                    raise RuntimeError('Power/temperature/duration guard stopped observation')
                time.sleep(.1)
        if process.returncode:
            raise RuntimeError('Observer process failed; inspect stderr.log')
        rows = json.loads((out/'stdout.log').read_text())
        if not rows or rows[0]['n_gen'] != 16:
            raise RuntimeError('Missing requested decode record')
        stderr = (out/'stderr.log').read_text()
        counts = re.findall(r'mojo_observer target_launch_calls=(\d+) mode=observe_only',stderr)
        if not counts or max(map(int,counts)) == 0:
            raise RuntimeError('No target launch observed')
        result['target_launch_calls'] = max(map(int,counts))
        result['observations'] = [line for line in stderr.splitlines() if line.startswith('mojo_observer api=')]
        result['status'] = 'passed'
    finally:
        remove_owned(name,token)
        if process and process.poll() is None:
            process.terminate(); process.wait(timeout=10)
        sensors.nv.nvmlShutdown()
        result['peaks'] = {k:max(s[k] for s in samples) for k in ['cpu_C','gpu_C','platform_C']} if samples else {}
        (out/'sensors.json').write_text(json.dumps(samples)+'\n')
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
