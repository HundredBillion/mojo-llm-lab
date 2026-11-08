import ctypes as C,json,os,subprocess as S,time,signal,datetime
from pathlib import Path
P=Path(__file__).parent;out=P/'intermediate-captures';out.mkdir(exist_ok=True)
os.umask(0o077)
cores=[t for h in Path('/sys/class/hwmon').glob('hwmon*') if (h/'name').read_text().strip()=='coretemp' for t in h.glob('temp*_input')]
n=C.CDLL('libnvidia-ml.so.1');assert n.nvmlInit_v2()==0
h=C.c_void_p();n.nvmlDeviceGetHandleByPciBusId_v2.argtypes=[C.c_char_p,C.POINTER(C.c_void_p)];assert n.nvmlDeviceGetHandleByPciBusId_v2(b'00000000:01:00.0',C.byref(h))==0
n.nvmlDeviceGetTemperature.argtypes=[C.c_void_p,C.c_uint,C.POINTER(C.c_uint)]
results={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Compile stream-read roofline probe under CPU and thermal guards','stages':[]}
active_container=None;active_unit=None;proc=None
def save(): (P/'roofline-nov07-build-results.json').write_text(json.dumps(results,indent=2))
def sample():
 v=C.c_uint();assert n.nvmlDeviceGetTemperature(h,0,C.byref(v))==0
 return {'cpu_C':max(int(t.read_text())/1000 for t in cores),'gpu_C':v.value,'AC':Path('/sys/class/power_supply/AC/online').read_text().strip()=='1','platform_C':max(int((z/'temp').read_text())/1000 for z in Path('/sys/class/thermal').glob('thermal_zone*') if (z/'type').read_text().strip().startswith('SEN'))}
def cleanup():
 global active_container,active_unit
 if active_container:
  r=S.run(['docker','inspect',active_container],capture_output=True,text=True)
  if r.returncode==0:
   o=json.loads(r.stdout)[0];assert o['Config']['Labels'].get('io.local.benchmark')=='mojo-numeric-20261106'
   S.run(['docker','rm','-f',active_container],capture_output=True)
  active_container=None
 if active_unit:
  S.run(['systemctl','--user','stop',active_unit+'.scope'],capture_output=True);active_unit=None
 if proc and proc.poll() is None:
  proc.terminate()
  try:proc.wait(5)
  except S.TimeoutExpired:proc.kill();proc.wait()
def terminate(*_):cleanup();raise SystemExit('Own numerical work stopped')
signal.signal(signal.SIGTERM,terminate)
def guarded(command,name):
 global proc
 deadline=time.monotonic()+45;stable=None
 while time.monotonic()<deadline:
  r=sample()
  if r['AC'] and r['cpu_C']<70 and r['gpu_C']<68 and r['platform_C']<75:
   stable=stable or time.monotonic()
   if time.monotonic()-stable>=5:break
  else:stable=None
  time.sleep(.25)
 else:raise RuntimeError('No cold/AC window')
 peaks={k:r[k] for k in ('cpu_C','gpu_C','platform_C')};start=time.monotonic();aborted=False
 with (P/(name+'.log')).open('w') as log:
  proc=S.Popen(command,cwd=repo,stdout=log,stderr=S.STDOUT)
  while proc.poll() is None:
   r=sample();peaks={k:max(v,r[k]) for k,v in peaks.items()}
   if not r['AC'] or r['cpu_C']>=90 or r['gpu_C']>=82 or r['platform_C']>=75 or time.monotonic()-start>150:
    aborted=True;cleanup();break
   time.sleep(.25)
 row={'stage':name,'returncode':proc.returncode,'guard_stopped':aborted,'seconds_including_initialization':time.monotonic()-start,'peaks':peaks};results['stages'].append(row);save();print(json.dumps(row),flush=True)
 if aborted or proc.returncode:raise RuntimeError('Numerical stage failed: '+name)
try:
 repo=Path('/path/to/mojo-llm-lab')
 (repo/'build/q4-gate').mkdir(exist_ok=True)
 jobs=[('roofline-nov07-mojo',['env','MOJO_BIN=/home/user/.local/state/a2000-lab/cpu-thermal-investigation/sequential-20261103/mojo-prototype-20261106/.venv/bin/mojo','bash',str(repo/'tools/build_q4.sh'),'stream-read-roofline'])]
 for name,cmd in jobs:
  active_unit='mojo-'+name+'-'+str(os.getpid())
  guarded(['systemd-run','--user','--scope','--quiet','--unit='+active_unit,'-p','CPUQuota=10%','-p','CPUQuotaPeriodSec=5ms','-p','MemoryMax=1G','-p','RuntimeMaxSec=150',*cmd],name);cleanup()
 results['status']='passed';save()
except BaseException as e:
 results['status']='stopped_or_failed';results['error']=str(e);save();print(str(e),flush=True)
finally:cleanup();n.nvmlShutdown()
