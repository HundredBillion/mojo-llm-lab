import json,subprocess as S,sys,time
from pathlib import Path
P=Path(__file__).parent;repo=Path('/path/to/mojo-llm-lab')
sys.path.insert(0,str(repo/'tools'))
from run_benchmark import Sensors,safe
sensors=Sensors(0)
NSYS='/home/user/.local/state/a2000-lab/cpu-thermal-investigation/sequential-20261103/profiler/usr/lib/nsight-systems'
plan=[(512,32,32,3,False),(2048,32,32,3,False),(6144,32,32,3,False),(2048,128,128,3,False),(2048,512,512,3,False),(2048,32,32,1,True)]
def cooldown():
    time.sleep(20);begin=time.monotonic();stable=None
    while time.monotonic()-begin<180:
        s=sensors.read()
        if not safe(s):return {'ok':False,'reason':'unsafe sample','sample':s}
        if s['cpu_C']<70 and s['gpu_C']<68:
            stable=stable or time.monotonic()
            if time.monotonic()-stable>=5:return {'ok':True,'waited_seconds':20+time.monotonic()-begin}
        else:stable=None
        time.sleep(.25)
    return {'ok':False,'reason':'no cool window within 180 s'}
out=P/'prefill-batch.json';result={'plan':plan,'runs':[],'status':'running'};out.write_text(json.dumps(result,indent=2))
for index,(ptok,b,ub,reps,nsys) in enumerate(plan):
    cool=cooldown()
    if not cool['ok']:result.update(status='stopped_before_run',error='Cool-down failed',cooldown=cool);out.write_text(json.dumps(result,indent=2));print('STOPPED cooldown',flush=True);sys.exit(1)
    fan=json.loads((P/'prefill-fan-window-status.json').read_text())
    if (fan.get('status')!='active' or fan['started_unix']+1200-time.time()<300 or time.time()-fan['samples'][-1]['time']>6
        or not all(4500<=rpm<10000 for rpm in fan['samples'][-1]['fan_rpm'])):
        result.update(status='stopped_before_run',error='Fan window not ready or under 300 s remaining');out.write_text(json.dumps(result,indent=2));print('STOPPED window',flush=True);sys.exit(1)
    label=f'local-prefill-{index:02d}-p{ptok}-ub{ub}'+('-nsys' if nsys else '')
    print('Starting '+label,flush=True);started=time.time()
    cmd=[sys.executable,str(repo/'tools/run_prefill_bench.py'),'--model',str(repo/'build/models/Qwen3.5-2B-Q4_K_M.gguf'),
         '--wait-library',str(repo/'build/integration/libllama-cuda-wait.so'),'--prompt-tokens',str(ptok),'--batch',str(b),'--ubatch',str(ub),
         '--repetitions',str(reps),'--output',str(repo/'results'/label)]+(['--nsys',NSYS] if nsys else [])
    r=S.run(cmd,cwd=repo/'tools',timeout=400)
    fan=json.loads((P/'prefill-fan-window-status.json').read_text())
    fan_valid=fan.get('status')=='active' and all(4500<=rpm<10000 for s in fan['samples'] if s['time']>=started for rpm in s['fan_rpm'])
    result['runs'].append({'label':label,'returncode':r.returncode,'cooldown':cool,'fan_window_valid':fan_valid,'started_unix':started,'ended_unix':time.time()})
    out.write_text(json.dumps(result,indent=2))
    if r.returncode or not fan_valid:result['status']='stopped_after_failure';out.write_text(json.dumps(result,indent=2));print('STOPPED '+label,flush=True);sys.exit(1)
result['status']='passed';out.write_text(json.dumps(result,indent=2));print('Prefill batch completed',flush=True)
