import json,subprocess as S,sys,time
from pathlib import Path
P=Path(__file__).parent
plan=[('control','1'),('mojo','0'),('mojo','1'),('mojo','1'),('mojo','0'),('control','1')]
result={'schedule':plan,'stages':[],'status':'running'}
output=P/'fastdiv-r2-confirmation-batch.json'
output.write_text(json.dumps(result,indent=2))
for index,(mode,arm) in enumerate(plan):
    if index:time.sleep(8)
    label=f'local-fastdiv-r2-confirmation-{index:02d}-{mode}-arm{arm}'
    print('Starting '+label,flush=True)
    fan=json.loads((P/'fastdiv-r2-fan-window-status.json').read_text())
    if (fan.get('status')!='active' or fan['started_unix']+1200-time.time()<90
        or time.time()-fan['samples'][-1]['time']>6
        or not all(4500<=rpm<10000 for rpm in fan['samples'][-1]['fan_rpm'])):
        result.update(status='stopped_before_run',error='Fan window not ready or insufficient time')
        output.write_text(json.dumps(result,indent=2));sys.exit(1)
    started=time.time()
    r=S.run([sys.executable,str(P/'quiet_long_source_fastdiv.py'),mode,'1000',label,'blocking',arm])
    result['stages'].append({'label':label,'returncode':r.returncode,'started_unix':started,'ended_unix':time.time()})
    fan=json.loads((P/'fastdiv-r2-fan-window-status.json').read_text())
    fan_valid=(fan.get('status')=='active' and time.time()-fan['samples'][-1]['time']<=6
               and all(4500<=rpm<10000 for sample in fan['samples'] if sample['time']>=started for rpm in sample['fan_rpm']))
    result['stages'][-1]['fan_window_valid']=fan_valid
    if r.returncode or not fan_valid:
        result['status']='stopped_after_failure';output.write_text(json.dumps(result,indent=2));sys.exit(1)
    output.write_text(json.dumps(result,indent=2))
result['status']='passed';output.write_text(json.dumps(result,indent=2));print('Scheduled crossover batch completed',flush=True)
