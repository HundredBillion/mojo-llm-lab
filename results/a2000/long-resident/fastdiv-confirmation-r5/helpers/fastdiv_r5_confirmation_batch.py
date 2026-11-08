import json,subprocess as S,sys,time
from pathlib import Path
P=Path(__file__).parent
sys.path.insert(0,'/path/to/mojo-llm-lab/tools')
from run_benchmark import Sensors,safe
plan=[('control','1'),('mojo','0'),('mojo','1'),('mojo','1'),('mojo','0'),('control','1')]
result={'schedule':plan,'stages':[],'status':'running',
        'cooldown':{'cpu_below_C':70,'gpu_below_C':68,'stable_seconds':5,'timeout_seconds':180,'stop_on_unsafe_sample':True},'gpu_clock_lock_MHz':[210,1200],'max_sm_clock_tolerance_MHz':1215}
output=P/'fastdiv-r5-confirmation-batch.json'
output.write_text(json.dumps(result,indent=2))
sensors=Sensors(0)
def cooldown():
    # Inter-stage wait (protocol r3): replaces the fixed 8 s gap of r2.
    begin=time.monotonic();stable=None;samples=0
    while time.monotonic()-begin<180:
        s=sensors.read();samples+=1
        if not safe(s):return {'ok':False,'reason':'unsafe sample','sample':s,'waited_seconds':time.monotonic()-begin}
        if s['cpu_C']<70 and s['gpu_C']<68:
            stable=stable or time.monotonic()
            if time.monotonic()-stable>=5:return {'ok':True,'waited_seconds':time.monotonic()-begin,'samples':samples,'last':s}
        else:stable=None
        time.sleep(.25)
    return {'ok':False,'reason':'no cool window within 180 s','waited_seconds':180}
for index,(mode,arm) in enumerate(plan):
    label=f'local-fastdiv-r5-confirmation-{index:02d}-{mode}-arm{arm}'
    cool=cooldown()
    if not cool['ok']:
        result.update(status='stopped_before_run',error='Cool-down failed before '+label,cooldown_result=cool)
        output.write_text(json.dumps(result,indent=2));sys.exit(1)
    print('Starting '+label,flush=True)
    fan=json.loads((P/'fastdiv-r5-fan-window-status.json').read_text())
    if (fan.get('status')!='active' or fan['started_unix']+1200-time.time()<90
        or time.time()-fan['samples'][-1]['time']>6
        or not all(4500<=rpm<10000 for rpm in fan['samples'][-1]['fan_rpm'])):
        result.update(status='stopped_before_run',error='Fan window not ready or insufficient time')
        output.write_text(json.dumps(result,indent=2));sys.exit(1)
    if fan.get('clock_lock_active') is not True:
        result.update(status='stopped_before_run',error='GPU clock lock not active');output.write_text(json.dumps(result,indent=2));sys.exit(1)
    started=time.time()
    r=S.run([sys.executable,str(P/'quiet_long_source_fastdiv.py'),mode,'1000',label,'blocking',arm])
    result['stages'].append({'label':label,'cooldown':cool,'returncode':r.returncode,'started_unix':started,'ended_unix':time.time()})
    fan=json.loads((P/'fastdiv-r5-fan-window-status.json').read_text())
    fan_valid=(fan.get('status')=='active' and time.time()-fan['samples'][-1]['time']<=6
               and all(4500<=rpm<10000 for sample in fan['samples'] if sample['time']>=started for rpm in sample['fan_rpm']))
    result['stages'][-1]['fan_window_valid']=fan_valid
    try:
        clocks=[x['sm_clock_MHz'] for x in json.loads((Path('/path/to/mojo-llm-lab/results')/label/'sensors.json').read_text()) if x.get('sm_clock_MHz') is not None]
        clock_valid=bool(clocks) and max(clocks)<=1215
        result['stages'][-1]['max_sm_clock_MHz']=max(clocks) if clocks else None
    except (OSError,ValueError,KeyError): clock_valid=False
    result['stages'][-1]['clock_lock_valid']=clock_valid
    if r.returncode or not fan_valid or not clock_valid:
        result['status']='stopped_after_failure';output.write_text(json.dumps(result,indent=2));sys.exit(1)
    output.write_text(json.dumps(result,indent=2))
result['status']='passed';output.write_text(json.dumps(result,indent=2));print('Scheduled confirmation batch completed',flush=True)
