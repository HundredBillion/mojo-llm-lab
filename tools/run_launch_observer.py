"""Observe a short model decode under guards; this is not a speed comparison."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess as S
import time
import uuid
from validate_ptx import validate_ptx
from run_benchmark import ROOT, Sensors, cold_gate, safe, remove_owned, digest


def validation_result(stderr, expected, feedback=False):
    if expected==0:
        counts=re.findall(r'mojo_validation eligible_calls=(\d+) validated_calls=(\d+) outputs=(CUDA|Mojo)',stderr)
        if len(counts)!=1 or int(counts[0][0])<1 or counts[0][0]!=counts[0][1]:
            raise RuntimeError('All-call validation incomplete')
        expected=int(counts[0][0])
    validations=re.findall(r'mojo_validation passed_call=(\d+) rows=9728 max_abs=([^\n]+)',stderr)
    if len(validations)!=expected or [int(x[0]) for x in validations]!=list(range(1,expected+1)):
        raise RuntimeError('Missing requested validation calls')
    errors=[float(x[1]) for x in validations]
    if any(not math.isfinite(x) or x<0 for x in errors):
        raise RuntimeError('Invalid numerical evidence')
    summaries=re.findall(r'mojo_validation eligible_calls=(\d+) validated_calls=(\d+) outputs=(CUDA|Mojo)',stderr)
    if len(summaries)!=1 or summaries[0][2]!=('Mojo' if feedback else 'CUDA') or int(summaries[0][1])!=expected or int(summaries[0][0])<expected:
        raise RuntimeError('Missing consistent final validation summary')
    if feedback and int(summaries[0][0])!=expected:
        raise RuntimeError('Feedback requires validation of every eligible call')
    return {'feedback':feedback,'validated_calls':expected,'validated_elements':expected*9728,'max_abs':max(errors),
            'eligible_calls':int(summaries[0][0])}


def replacement_result(stderr, graphs):
    matches=re.findall(r'mojo_replacement calls=(\d+) captured_calls=(\d+)',stderr)
    if len(matches)!=1:
        raise RuntimeError('Missing replacement summary')
    calls,captured=map(int,matches[0])
    if calls<1 or captured>calls or (graphs and captured<1) or (not graphs and captured!=0):
        raise RuntimeError('Invalid replacement/capture evidence')
    return {'replacement_calls':calls,'captured_replacement_calls':captured}


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--disable-graphs', action='store_true')
    parser.add_argument('--prompt', help='Synthetic text for a greedy completion check instead of llama-bench')
    parser.add_argument('--tokens', type=int, default=16, choices=range(1,129), metavar='N')
    parser.add_argument('--repetitions', type=int, default=1, choices=range(1,9), metavar='N')
    parser.add_argument('--feedback', action='store_true')
    kernels=parser.add_mutually_exclusive_group()
    kernels.add_argument('--validate-ptx', type=Path)
    kernels.add_argument('--replace-ptx', type=Path)
    parser.add_argument('--validation-calls', type=int, default=None, choices=range(0,10001), metavar='N')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--wait-library', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    return parser


def parse_arguments(argv=None):
    parser=argument_parser()
    args=parser.parse_args(argv)
    if args.validation_calls is None:
        args.validation_calls=0 if args.feedback else 36
    if args.feedback and not args.validate_ptx:
        parser.error('--feedback requires --validate-ptx')
    return args


def main():
    args=parse_arguments()
    model = args.model.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    observer = ROOT/'build/integration/observe_launches.so'
    versions = json.loads((ROOT/'versions.json').read_text())
    result = {'status':'failed_or_stopped', 'scope':'Host-launch observation only; includes capture/warmup and excludes graph replay counts. No throughput claim.',
              'versions':versions, 'observer_sha256':digest(observer), 'model_name':model.name,
              'model_bytes':model.stat().st_size, 'model_sha256':None,
              'cpu_quota_percent':20, 'generation_tokens':args.tokens, 'wait_library_sha256':None}
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
        '-e','LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64','--entrypoint','/app/llama-completion' if args.prompt else '/app/llama-bench']
    if args.disable_graphs:
        command += ['-e','GGML_CUDA_DISABLE_GRAPHS=1']
    result['cuda_graphs_enabled']=not (args.disable_graphs or args.validate_ptx)
    mounts = [(ROOT/'build/upstream'/('llama-'+versions['engine_release']),'/app'),
              (observer.parent,'/observer'), (model,'/model.gguf')]
    if args.validate_ptx:
        ptx=args.validate_ptx.resolve(strict=True)
        entry=validate_ptx(ptx.read_text(),expected_parameters=4)
        mounts.append((ptx,'/candidate.ptx'))
        command += ['-e','MOJO_VALIDATE_PTX=/candidate.ptx','-e','MOJO_VALIDATE_ENTRY='+entry,
                    '-e','GGML_CUDA_DISABLE_GRAPHS=1','-e','MOJO_VALIDATE_CALLS='+str(args.validation_calls or 10000)]
        if args.feedback:
            command += ['-e','MOJO_VALIDATE_FEEDBACK=1']
        result['ptx_sha256']=digest(ptx)
        result['scope']=f'Validation only: {args.validation_calls or "all (maximum 10000)"} eligible launches, outputs={"Mojo" if args.feedback else "CUDA"}; graphs disabled. No throughput claim.'
    if args.replace_ptx:
        ptx=args.replace_ptx.resolve(strict=True)
        entry=validate_ptx(ptx.read_text(),expected_parameters=4)
        mounts.append((ptx,'/candidate.ptx'))
        command += ['-e','MOJO_REPLACE_PTX=/candidate.ptx','-e','MOJO_REPLACE_ENTRY='+entry]
        result['ptx_sha256']=digest(ptx)
        result['scope']=f'Experimental model replacement, CUDA graphs {"disabled" if args.disable_graphs else "enabled"}; short smoke test, not a qualified speed comparison.'
    preload = '/observer/observe_launches.so'
    if args.wait_library:
        wait = args.wait_library.resolve(strict=True)
        result['wait_library_sha256'] = digest(wait)
        mounts.append((wait,'/wait.so'))
        preload += ':/wait.so'
    for src, dst in mounts:
        command += ['--mount',f'type=bind,src={src},dst={dst},readonly']
    command += ['-e','LD_PRELOAD='+preload, versions['cuda_image'],
        '-m','/model.gguf','-n',str(args.tokens),'-t','2','--poll','0',
        '-ngl','99','-b','32','-ub','32','-ctk','q4_0','-ctv','q4_0','-fa','on']
    if args.prompt:
        command += ['-p',args.prompt,'--seed','42','--temp','0','--no-conversation',
                    '--no-display-prompt','--simple-io','-c','512']
        result['prompt']=args.prompt
    else:
        command += ['-p','0','-r',str(args.repetitions),'-o','json']
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
        if args.prompt:
            if not (out/'stdout.log').read_text().strip():
                raise RuntimeError('Empty generated completion')
            result['completion_sha256']=digest(out/'stdout.log')
        else:
            rows = json.loads((out/'stdout.log').read_text())
            if len(rows)!=1 or rows[0]['n_gen'] != args.tokens or len(rows[0]['samples_ns'])!=args.repetitions:
                raise RuntimeError('Missing requested decode records')
            result['samples_ns']=rows[0]['samples_ns']
            result['samples_ts']=rows[0]['samples_ts']
        stderr = (out/'stderr.log').read_text()
        counts = re.findall(r'mojo_observer target_launch_calls=(\d+) mode=host_launches',stderr)
        if not counts or max(map(int,counts)) == 0:
            raise RuntimeError('No target launch observed')
        result['target_launch_calls'] = max(map(int,counts))
        result['observations'] = [line for line in stderr.splitlines() if line.startswith('mojo_observer api=')]
        if args.validate_ptx:
            result.update(validation_result(stderr,args.validation_calls,args.feedback))
        if args.replace_ptx:
            result.update(replacement_result(stderr,not args.disable_graphs))
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
