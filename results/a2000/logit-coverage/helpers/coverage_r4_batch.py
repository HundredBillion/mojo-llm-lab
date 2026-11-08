import json,subprocess as S,sys,time
from pathlib import Path
sys.path.insert(0,'/path/to/mojo-llm-lab/tools')
from run_benchmark import Sensors,safe
sensors=Sensors(0)
def cooldown():
    # Attempt 4: idle gap plus pre-launch cold window, as in the fast-division r3 amendment.
    time.sleep(20)
    begin=time.monotonic();stable=None
    while time.monotonic()-begin<180:
        s=sensors.read()
        if not safe(s):return {'ok':False,'reason':'unsafe sample','sample':s}
        if s['cpu_C']<70 and s['gpu_C']<68:
            stable=stable or time.monotonic()
            if time.monotonic()-stable>=5:return {'ok':True,'waited_seconds':20+time.monotonic()-begin}
        else:stable=None
        time.sleep(.25)
    return {'ok':False,'reason':'no cool window within 180 s'}
P=Path(__file__).parent;repo=Path('/path/to/mojo-llm-lab')
plan=[('statement-short','greedy',0,1),('statement-medium','greedy',0,0),('statement-long','greedy',0,1),('prose','greedy',0,0),
      ('statement-short','sampled',20261107,1),('statement-long','sampled',20261107,0)]
FIRST=4  # runs 0-1 passed in attempt 2, runs 2-3 in attempt 3; not repeated
out=P/'coverage-r4-batch.json';result={'plan':plan,'runs':[],'status':'running','started_unix':time.time()};out.write_text(json.dumps(result,indent=2))
for index,(prompt,policy,seed,arm) in enumerate(plan):
    if index<FIRST:continue
    cool=cooldown();result.setdefault('cooldowns',[]).append(cool)
    if not cool['ok']:
        result.update(status='stopped_before_run',error='Cool-down failed');out.write_text(json.dumps(result,indent=2));print('STOPPED cooldown',flush=True);sys.exit(1)
    if S.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        result.update(status='stopped_before_run',error='GPU in use');out.write_text(json.dumps(result,indent=2));sys.exit(1)
    fan=json.loads((P/'coverage4-fan-window-status.json').read_text())
    if (fan.get('status')!='active' or fan.get('clock_lock_active') is not True or fan['started_unix']+1200-time.time()<300
        or time.time()-fan['samples'][-1]['time']>6 or not all(4500<=rpm<10000 for rpm in fan['samples'][-1]['fan_rpm'])):
        result.update(status='stopped_before_run',error='Fan/clock window not ready or under 300 s remaining');out.write_text(json.dumps(result,indent=2));print('STOPPED window',flush=True);sys.exit(1)
    label=f'local-coverage-r4-{index:02d}-{prompt}-{policy}-arm{arm}'
    print('Starting '+label,flush=True);started=time.time()
    cmd=[sys.executable,str(repo/'tools/run_logit_coverage.py'),'--model',str(repo/'build/models/Qwen3.5-2B-Q4_K_M.gguf'),
         '--ptx',str(repo/'build/q4-gate-order-2b-fastdiv/host_q4_k_q8_1_gate_order_2b_fastdiv6A_cb8e527a88629b4d.ptx'),
         '--wait-library',str(repo/'build/integration/libllama-cuda-wait.so'),'--prompt',str(repo/'experiments/coverage-prompts'/(prompt+'.txt')),
         '--policy',policy,'--trial-cooldown','--mojo-arm',str(arm),'--output',str(repo/'results'/label)]+(['--seed',str(seed)] if policy=='sampled' else [])
    r=S.run(cmd,cwd=repo/'tools',timeout=1300)
    try:
        clocks=[x['sm_clock_MHz'] for x in json.loads((repo/'results'/label/'sensors.json').read_text()) if x.get('sm_clock_MHz') is not None]
        max_clock=max(clocks) if clocks else None
    except (OSError,ValueError):max_clock=None
    result['runs'].append({'label':label,'returncode':r.returncode,'max_sm_clock_MHz':max_clock,'started_unix':started,'ended_unix':time.time()})
    if max_clock is None or max_clock>915:result['status']='stopped_clock_unverified';out.write_text(json.dumps(result,indent=2));print('STOPPED clock '+label,flush=True);sys.exit(1)
    out.write_text(json.dumps(result,indent=2))
    if r.returncode:result['status']='stopped_after_failure';out.write_text(json.dumps(result,indent=2));print('STOPPED '+label,flush=True);sys.exit(1)
result['status']='passed';out.write_text(json.dumps(result,indent=2));print('Coverage batch completed',flush=True)
