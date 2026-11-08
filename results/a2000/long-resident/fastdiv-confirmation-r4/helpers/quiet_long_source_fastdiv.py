import json,os,signal,subprocess as S,sys,time,uuid
from pathlib import Path
P=Path(__file__).parent
repo=Path('/path/to/mojo-llm-lab')
mode,quota,label,*wait_args=sys.argv[1:]
wait_policy=wait_args[0] if wait_args else "blocking"
assert wait_policy in ("auto","blocking")
mojo_arm=wait_args[1] if len(wait_args)>1 else "1"
assert mojo_arm in ("0","1")
assert mode in ('control','mojo') and quota in ('1000','2000')
assert label.replace('-','').isalnum()
def identity(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text();a=s[s.rindex(')')+2:].split()
 return a[19],a[0]
targets=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  exe=(p/'exe').resolve();birth,state=identity(p.name)
  if exe in (Path('/usr/lib/chromium/chromium'),Path('/home/user/.local/bin/sprite')) and p.stat().st_uid==os.getuid() and state not in ('T','t','Z'):
   targets.append((int(p.name),birth,exe.name))
 except (OSError,ValueError):pass
assert any(x[2]=='sprite' for x in targets)
unit='mojo-desktop-resume-'+uuid.uuid4().hex
watch=P/(unit+'.py')
watch.write_text('import os,signal\nfrom pathlib import Path\nfor pid,birth,name in '+repr(targets)+':\n try:\n  s=Path(f"/proc/{pid}/stat").read_text()\n  if s[s.rindex(")")+2:].split()[19]==birth:os.kill(pid,signal.SIGCONT)\n except ProcessLookupError:pass\n except FileNotFoundError:pass\n')
S.run(['systemd-run','--user','--quiet','--unit='+unit,'--on-active=80s','--timer-property=AccuracySec=1s','--timer-property=RandomizedDelaySec=0','/usr/bin/python3',str(watch)],check=True)
S.run(['systemctl','--user','is-active','--quiet',unit+'.timer'],check=True)
def resume():
 for pid,birth,_ in targets:
  try:
   if identity(pid)[0]==birth:os.kill(pid,signal.SIGCONT)
  except (ProcessLookupError,FileNotFoundError):pass
signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
result={'watchdog_seconds':80,'deadline_seconds':75,'targets':targets,'started_unix':time.time()}
process=None;start=time.monotonic()
try:
 for pid,birth,_ in targets:
  if identity(pid)[0]==birth:os.kill(pid,signal.SIGSTOP)
 print('Present Chromium/Sprite processes paused; independent resume watchdog active',flush=True)
 command=[sys.executable,str(repo/'tools/run_resident_decode.py'),'--mode',mode,'--mojo-arm',mojo_arm,'--continuous','--tokens','128','--prime-tokens','2','--cuda-wait',wait_policy,'--cpu-quota-us',quota,'--model',str(repo/'build/models/Qwen3.5-2B-Q4_K_M.gguf'),'--ptx',str(repo/'build/q4-gate-order-2b-fastdiv/host_q4_k_q8_1_gate_order_2b_fastdiv6A_cb8e527a88629b4d.ptx'),'--wait-library',str(repo/'build/integration/libllama-cuda-wait.so'),'--output',str(repo/'results'/label)]
 process=S.Popen(command,cwd=repo)
 result['returncode']=process.wait(timeout=75)
except BaseException as e:result['error']=type(e).__name__+': '+str(e)
finally:
 resume();result['pause_seconds']=time.monotonic()-start
 if process and process.poll() is None:
  process.terminate()
  try:process.wait(timeout=20)
  except S.TimeoutExpired:result['cleanup_error']='Runner did not finish cleanup';process.kill();process.wait()
 time.sleep(.1)
 result['still_stopped']=[]
 for pid,birth,_ in targets:
  try:
   now,state=identity(pid)
   if now==birth and state in ('T','t'):result['still_stopped'].append(pid)
  except FileNotFoundError:pass
 if not result['still_stopped']:S.run(['systemctl','--user','stop',unit+'.timer'],capture_output=True)
 (P/(label+'-pause.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
sys.exit(0 if result.get('returncode')==0 and not result.get('cleanup_error') and not result['still_stopped'] else 1)
