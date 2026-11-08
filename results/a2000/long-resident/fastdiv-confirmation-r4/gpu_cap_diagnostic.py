import json,subprocess as S,time,os
from pathlib import Path
out=Path(__file__).parent/'gpu-cap-diagnostic.json'
def run(*a):
    r=S.run(['nvidia-smi',*a],capture_output=True,text=True,timeout=15);return {'cmd':' '.join(a),'rc':r.returncode,'out':(r.stdout+r.stderr).strip()[-600:]}
def q(): return run('--query-gpu=enforced.power.limit,power.default_limit,clocks.max.sm','--format=csv,noheader')
rec={'euid':os.geteuid(),'steps':[]}
assert not S.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
try:
    rec['steps'].append(q())
    rec['steps'].append(run('-i','0','-pl','25'))
    for d in (0.2,1,2): time.sleep(d); rec['steps'].append(q())
    rec['steps'].append(run('-q','-d','POWER'))
    rec['steps'].append(run('-i','0','-lgc','210,1200'))
    time.sleep(1); rec['steps'].append(run('-q','-d','PERFORMANCE,CLOCK'))
finally:
    rec['restore']=[run('-i','0','-rgc'),run('-i','0','-pl','35')]
    time.sleep(1); rec['final']=q()
    out.write_text(json.dumps(rec,indent=2)); print(json.dumps(rec['final']))
