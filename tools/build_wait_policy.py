"""Build the opt-in CUDA wait helper without accessing the GPU."""
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    versions=json.loads((ROOT/'versions.json').read_text())
    output=ROOT/'build/integration'
    output.mkdir(parents=True,exist_ok=True)
    subprocess.run([
        'docker','run','--rm','--network','none','--read-only',
        '--tmpfs','/tmp:rw,nosuid,size=64m','--cap-drop','ALL',
        '--security-opt','no-new-privileges','--user',f'{os.getuid()}:{os.getgid()}',
        '--cpu-period','10000','--cpu-quota','1000','--memory','256m','--memory-swap','256m',
        '--mount',f'type=bind,src={ROOT},dst=/src,readonly',
        '--mount',f'type=bind,src={output},dst=/output',
        '--entrypoint','gcc',versions['cuda_image'],
        '-O2','-shared','-fPIC','-Wall','-Wextra','-Werror',
        '/src/integration/cuda_wait.c','-ldl','-o','/output/libllama-cuda-wait.so',
    ],check=True)
    print('Built opt-in CUDA wait helper; no host settings changed')

if __name__=='__main__':main()
