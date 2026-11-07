"""Guarded resident fixed-token comparison; not sampled-output or PDF latency."""
import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess as S
import time
import uuid

from run_benchmark import cold_gate, safe, digest
from run_launch_observer import ModelSensors, validate_model_ptx
from run_pdf_smoke import cleanup

ROOT=Path(__file__).resolve().parents[1]


def validate_trial(row, arm, previous, mode, tokens=32):
    if row.get('arm')!=arm or row.get('tokens')!=tokens or row.get('decode_ns',0)<=0:
        raise ValueError('Unexpected trial workload or timing')
    for key in ('cpu_usage_usec','nr_throttled','throttled_usec','replacement_calls','captured_calls','timed_replacement_calls','timed_captured_calls'):
        if not isinstance(row.get(key),int) or row[key]<0:
            raise ValueError('Invalid counter: '+key)
    if row['captured_calls']>row['replacement_calls'] or row['replacement_calls']<previous:
        raise ValueError('Inconsistent replacement counters')
    if (row['timed_captured_calls']>row['timed_replacement_calls']
        or row['timed_replacement_calls']>row['replacement_calls']-previous
        or row['timed_captured_calls']>row['captured_calls']
        or row['timed_replacement_calls']-row['timed_captured_calls']
            >row['replacement_calls']-row['captured_calls']):
        raise ValueError('Inconsistent timed counters')
    if (mode=='control' or arm==0) and row['replacement_calls']!=previous:
        raise ValueError('CUDA arm unexpectedly replaced kernels')
    if mode=='mojo' and arm==1 and row['captured_calls']==0:
        raise ValueError('Candidate graph capture missing')
    return row['replacement_calls']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--ptx',type=Path,required=True)
    p.add_argument('--wait-library',type=Path,required=True)
    p.add_argument('--tokens',type=int,choices=[16,32],default=16)
    p.add_argument('--cycles',type=int,choices=[1,2],default=1)
    p.add_argument('--cpu-quota-us',type=int,choices=[1000,2000],default=1000,
                   help='Experimental container CPU quota per 5000us period; default unchanged')
    p.add_argument('--continuous',action='store_true',help='No inter-trial cooldown; all absolute guards still apply')
    p.add_argument('--mode',choices=['control','mojo'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    versions=json.loads((ROOT/'versions.json').read_text())
    entry=validate_model_ptx(a.ptx.read_text(),'qwen35-2b')
    if S.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU already in use')
    sensor=ModelSensors(0); samples=[]; rows=[]
    read_sensor=sensor.read
    def recorded_read():
        sample=read_sensor(); samples.append(sample); return sample
    sensor.read=recorded_read
    name='mojo-resident-'+uuid.uuid4().hex; token=uuid.uuid4().hex
    command=['docker','run','--rm','-i','--name',name,'--label','io.mojo-imrope.benchmark='+token,
        '--device','nvidia.com/gpu=0','--user',f'{os.getuid()}:{os.getgid()}',
        '--network','none','--read-only','--tmpfs','/tmp:rw,size=512m','--cap-drop','ALL',
        '--security-opt','no-new-privileges','--cpu-period','5000','--cpu-quota',str(a.cpu_quota_us),
        '--memory','12g','--memory-swap','12g','--entrypoint','/observer/resident-decode',
        '-e','LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64',
        '-e','LD_PRELOAD=/observer/observe_launches.so:/wait.so','-e','MOJO_SHAPE=qwen35-2b',
        '-e','MOJO_RESIDENT=1','-e','MOJO_CANDIDATE_PTX=/candidate.ptx','-e','MOJO_REPLACE_ENTRY='+entry]
    for src,dst in [(ROOT/'build/upstream/llama-b12110','/app'),(ROOT/'build/integration','/observer'),
                    (a.model,'/model.gguf'),(a.ptx,'/candidate.ptx'),(a.wait_library,'/wait.so')]:
        command+=['--mount',f'type=bind,src={src.resolve()},dst={dst},readonly']
    command += [versions['cuda_image'],'/model.gguf',a.mode]
    result={'status':'failed_or_stopped','mode':a.mode,'versions':versions,'protocol':'resident-block-v2' if a.continuous else 'resident-cooldown-v3','tokens':a.tokens,'cycles':a.cycles,
            'cpu_period_us':5000,'cpu_quota_us':a.cpu_quota_us,
            'model_name':a.model.name,'model_bytes':a.model.stat().st_size,
            'ptx_sha256':digest(a.ptx),'worker_sha256':digest(ROOT/'build/integration/resident-decode'),
            'observer_sha256':digest(ROOT/'build/integration/observe_launches.so'),
            'wait_sha256':digest(a.wait_library),'trials':rows,
            'scope':'Resident fixed-token decode, separate contexts, two warmups then ABBA cycles. Not sampled-output or PDF latency.'}
    process=None; start=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        cold_gate(sensor)
        with (out/'stderr.log').open('w') as err,(out/'stdout.log').open('w') as raw:
            process=S.Popen(command,stdin=S.PIPE,stdout=S.PIPE,stderr=err,text=True,bufsize=1)
            sel=selectors.DefaultSelector();sel.register(process.stdout,selectors.EVENT_READ)
            def receive():
                deadline=time.monotonic()+90
                while time.monotonic()<deadline and time.monotonic()-start<360:
                    sample=sensor.read()
                    if not safe(sample):raise RuntimeError('Temperature or AC guard')
                    if sel.select(.1):
                        line=process.stdout.readline()
                        if not line:raise RuntimeError('Worker exited before result')
                        raw.write(line);raw.flush()
                        return json.loads(line)
                raise RuntimeError('Worker deadline')
            ready=receive()
            if ready.get('ready') is not True:raise RuntimeError('Missing ready handshake')
            cold_gate(sensor)
            previous=0
            for index,arm in enumerate([0,1]+[0,1,1,0]*a.cycles):
                if time.monotonic()-start>300:raise RuntimeError('Experiment deadline')
                if not a.continuous:cold_gate(sensor)
                process.stdin.write(f'{arm} {a.tokens}\n');process.stdin.flush()
                row=receive()
                if row.get('trial')!=index:raise RuntimeError('Trial order mismatch')
                previous=validate_trial(row,arm,previous,a.mode,a.tokens)
                row['warmup']=index<2;rows.append(row)
                print(json.dumps(row),flush=True)
            process.stdin.close();process.wait(timeout=30)
            if process.returncode:raise RuntimeError('Worker failed')
            if len({r['top_token'] for r in rows})!=1:raise RuntimeError('Fixed-prefix top-token mismatch')
            result['status']='passed'
    except BaseException as error:
        result['error']=str(error)
    finally:
        errors=cleanup(name,token,[process])
        if errors:result.update(status='failed_or_stopped',cleanup_errors=errors)
        result['elapsed_seconds']=time.monotonic()-start
        result['peaks']={k:max(s[k] for s in samples) for k in ['cpu_C','gpu_C','platform_C']} if samples else {}
        (out/'sensors.json').write_text(json.dumps(samples)+'\n')
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'error':result.get('error'),'peaks':result['peaks']}))
    raise SystemExit(0 if result['status']=='passed' else 1)


if __name__=='__main__':main()
