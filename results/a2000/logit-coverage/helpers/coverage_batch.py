import json,subprocess as S,sys,time
from pathlib import Path
P=Path(__file__).parent;repo=Path('/path/to/mojo-llm-lab')
plan=[('statement-short','greedy',0,1),('statement-medium','greedy',0,0),('statement-long','greedy',0,1),('prose','greedy',0,0),
      ('statement-short','sampled',20261107,1),('statement-long','sampled',20261107,0)]
out=P/'coverage-batch.json';result={'plan':plan,'runs':[],'status':'running','started_unix':time.time()};out.write_text(json.dumps(result,indent=2))
for index,(prompt,policy,seed,arm) in enumerate(plan):
    if S.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        result.update(status='stopped_before_run',error='GPU in use');out.write_text(json.dumps(result,indent=2));sys.exit(1)
    label=f'local-coverage-{index:02d}-{prompt}-{policy}-arm{arm}'
    print('Starting '+label,flush=True);started=time.time()
    cmd=[sys.executable,str(repo/'tools/run_logit_coverage.py'),'--model',str(repo/'build/models/Qwen3.5-2B-Q4_K_M.gguf'),
         '--ptx',str(repo/'build/q4-gate-order-2b-fastdiv/host_q4_k_q8_1_gate_order_2b_fastdiv6A_cb8e527a88629b4d.ptx'),
         '--wait-library',str(repo/'build/integration/libllama-cuda-wait.so'),'--prompt',str(repo/'experiments/coverage-prompts'/(prompt+'.txt')),
         '--policy',policy,'--mojo-arm',str(arm),'--output',str(repo/'results'/label)]+(['--seed',str(seed)] if policy=='sampled' else [])
    r=S.run(cmd,cwd=repo/'tools',timeout=600)
    result['runs'].append({'label':label,'returncode':r.returncode,'started_unix':started,'ended_unix':time.time()})
    out.write_text(json.dumps(result,indent=2))
    if r.returncode:result['status']='stopped_after_failure';out.write_text(json.dumps(result,indent=2));print('STOPPED '+label,flush=True);sys.exit(1)
result['status']='passed';out.write_text(json.dumps(result,indent=2));print('Coverage batch completed',flush=True)
