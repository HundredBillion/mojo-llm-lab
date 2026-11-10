"""Guarded resident fixed-token comparison; not sampled-output or PDF latency."""
import argparse
import json
import math
import re
import os
from pathlib import Path
import selectors
import signal
import subprocess as S
import time
import uuid

from run_benchmark import cold_gate, safe, digest
from run_launch_observer import ModelSensors, validate_model_ptx
from run_benchmark import cleanup
from cuda_wait_policy import verify_wait_policy

ROOT=Path(__file__).resolve().parents[1]


def validate_ready(row, prime_tokens, mojo_arm, tokens):
    if row.get('ready') is not True:
        raise ValueError('Missing ready handshake')
    for key, expected in (('prime_tokens', prime_tokens), ('mojo_arm', mojo_arm)):
        if type(row.get(key)) is not int or row[key] != expected:
            raise ValueError('Worker capability mismatch: '+key)
    if type(row.get('max_timed_tokens')) is not int or row['max_timed_tokens'] < tokens:
        raise ValueError('Worker does not support requested timed tokens')
    if type(row.get('fixture_tokens')) is not int or row['fixture_tokens'] < 8+prime_tokens+tokens:
        raise ValueError('Worker fixture is too short')


def validate_trial(row, arm, previous, mode, tokens=32, prime_tokens=0, mojo_arm=1):
    if type(row.get('mojo_arm')) is not int or row['mojo_arm']!=mojo_arm:
        raise ValueError('Candidate context mismatch')
    if type(row.get('prime_tokens')) is not int or row['prime_tokens']!=prime_tokens:
        raise ValueError('Priming workload mismatch')
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
    if (mode=='control' or arm!=mojo_arm) and row['replacement_calls']!=previous:
        raise ValueError('CUDA arm unexpectedly replaced kernels')
    if mode=='mojo' and arm==mojo_arm and row['captured_calls']==0:
        raise ValueError('Candidate graph capture missing')
    return row['replacement_calls']



def validate_logits_report(stderr, tokens, prime_tokens, cycles):
    matches=re.findall(r'^mojo_logits (\{.*\})$',stderr,re.M)
    if len(matches)!=1:raise ValueError('Missing or ambiguous logit validation report')
    report=json.loads(matches[0])
    positions=1+prime_tokens+tokens
    vocabulary=report.get('vocabulary')
    if type(vocabulary) is not int or vocabulary<=0:raise ValueError('Invalid validation vocabulary')
    expected={'positions':positions,'candidate_positions':(1+2*cycles)*positions,
              'control_positions':2*cycles*positions}
    expected.update(candidate_values=expected['candidate_positions']*vocabulary,
                    control_values=expected['control_positions']*vocabulary)
    for name,value in expected.items():
        if type(report.get(name)) is not int or report[name]!=value:raise ValueError('Incomplete logit validation: '+name)
    if report.get('passed') is not True or report.get('atol')!=1e-4 or report.get('rtol')!=1e-5:
        raise ValueError('Logit validation failed or tolerance changed')
    for mode in ('candidate','control'):
        for suffix in ('violations','top_mismatches'):
            value=report.get(mode+'_'+suffix)
            if type(value) is not int or value!=0:raise ValueError('Numerical or top-token mismatch')
        for suffix in ('max_abs','max_tolerance_ratio'):
            value=report.get(mode+'_'+suffix)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Invalid numerical error')
        if report[mode+'_max_tolerance_ratio']>1:raise ValueError('Numerical tolerance exceeded')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--ptx',type=Path,required=True)
    p.add_argument('--wait-library',type=Path,required=True)
    p.add_argument('--validate-logits',action='store_true',help='Separate numerical-only worker; its timings are not performance evidence')
    p.add_argument('--tokens',type=int,choices=[16,32,128],default=16)
    p.add_argument('--cycles',type=int,choices=[1,2],default=1)
    p.add_argument('--cpu-quota-us',type=int,choices=[1000,2000],default=1000,
                   help='Experimental container CPU quota per 5000us period; default unchanged')
    p.add_argument('--cuda-wait',choices=['auto','blocking'],default='blocking')
    p.add_argument('--prime-tokens',type=int,choices=[0,1,2],default=0,help='Synchronized untimed decode tokens after prefill')
    p.add_argument('--continuous',action='store_true',help='No inter-trial cooldown; all absolute guards still apply')
    p.add_argument('--mojo-arm',type=int,choices=[0,1],default=1)
    p.add_argument('--mode',choices=['control','mojo'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.validate_logits and a.mode!='mojo':p.error('--validate-logits requires --mode mojo')
    worker='resident-logits' if a.validate_logits else 'resident-decode'
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
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
        '--memory','12g','--memory-swap','12g','--entrypoint','/observer/'+worker,
        '-e','LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64',
        '-e','LD_PRELOAD=/observer/observe_launches.so:/wait.so','-e','MOJO_SHAPE=qwen35-2b',
        '-e','LLAMA_CUDA_WAIT_TARGET=/observer/'+worker,'-e','LLAMA_CUDA_WAIT='+a.cuda_wait,
        '-e','MOJO_RESIDENT=1','-e','MOJO_CANDIDATE_PTX=/candidate.ptx','-e','MOJO_REPLACE_ENTRY='+entry]
    for src,dst in [(ROOT/'build/upstream/llama-b12110','/app'),(ROOT/'build/integration','/observer'),
                    (a.model,'/model.gguf'),(a.ptx,'/candidate.ptx'),(a.wait_library,'/wait.so')]:
        command+=['--mount',f'type=bind,src={src.resolve()},dst={dst},readonly']
    command += [versions['cuda_image'],'/model.gguf',a.mode,str(a.prime_tokens),str(a.mojo_arm)]
    result={'status':'failed_or_stopped','mode':a.mode,'versions':versions,'validation_only':a.validate_logits,'protocol':'resident-logit-validation-v1' if a.validate_logits else 'resident-block-v2' if a.continuous else 'resident-cooldown-v3','tokens':a.tokens,'cycles':a.cycles,
            'mojo_arm':a.mojo_arm,'cuda_wait_requested':a.cuda_wait,'prime_tokens':a.prime_tokens,'cpu_period_us':5000,'cpu_quota_us':a.cpu_quota_us,
            'model_name':a.model.name,'model_bytes':a.model.stat().st_size,
            'ptx_sha256':digest(a.ptx),'worker_sha256':digest(ROOT/'build/integration'/worker),
            'observer_sha256':digest(ROOT/'build/integration/observe_launches.so'),
            'wait_sha256':digest(a.wait_library),'trials':rows,
            'scope':'Numerical-only comparison at every fixed-fixture position; synchronization/copies invalidate timing.' if a.validate_logits else 'Resident fixed-token decode, separate contexts, two warmups then ABBA cycles. Not sampled-output or PDF latency.'}
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
            validate_ready(ready,a.prime_tokens,a.mojo_arm,a.tokens)
            result['worker_capabilities']=ready
            cold_gate(sensor)
            previous=0
            for index,arm in enumerate([0,1]+[0,1,1,0]*a.cycles):
                if time.monotonic()-start>300:raise RuntimeError('Experiment deadline')
                if not a.continuous:cold_gate(sensor)
                process.stdin.write(f'{arm} {a.tokens}\n');process.stdin.flush()
                row=receive()
                if row.get('trial')!=index:raise RuntimeError('Trial order mismatch')
                previous=validate_trial(row,arm,previous,a.mode,a.tokens,a.prime_tokens,a.mojo_arm)
                row['warmup']=index<2;rows.append(row)
                print(json.dumps(row),flush=True)
            process.stdin.close();process.wait(timeout=30)
            if process.returncode:raise RuntimeError('Worker failed')
            if len({r['top_token'] for r in rows})!=1:raise RuntimeError('Fixed-prefix top-token mismatch')
            result['cuda_wait_verified']=verify_wait_policy((out/'stderr.log').read_text(),a.cuda_wait)
            if a.validate_logits:result['logit_validation']=validate_logits_report((out/'stderr.log').read_text(),a.tokens,a.prime_tokens,a.cycles)
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
