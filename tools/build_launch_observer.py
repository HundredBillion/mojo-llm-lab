"""Compile the host harness in a pinned, CPU-limited CUDA container."""
import json
from pathlib import Path
import subprocess
import os
import argparse

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    worker=parser.add_mutually_exclusive_group()
    worker.add_argument('--resident',action='store_true')
    worker.add_argument('--resident-logits',action='store_true')
    worker.add_argument('--resident-coverage',action='store_true')
    args=parser.parse_args()
    versions = json.loads((ROOT / 'versions.json').read_text())
    output = ROOT / 'build/integration'
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'docker', 'run', '--rm', '--network', 'none', '--read-only',
        '--tmpfs', '/tmp:rw,nosuid,size=256m', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--user', f'{os.getuid()}:{os.getgid()}',
        '--cpu-period', '10000', '--cpu-quota', '1000', '--memory', '1g', '--memory-swap', '1g',
        '--mount', f'type=bind,src={ROOT},dst=/src,readonly',
        '--mount', f'type=bind,src={output},dst=/output', '--entrypoint', 'sh', versions['cuda_image'], '-c',
        'g++ -O2 -std=c++17 -Wall -Wextra -Werror /src/tests/fusion_eligibility.cpp -o /output/test-eligibility && /output/test-eligibility && exec g++ "$@"', 'build',
        *(['-O2', *(['-DMOJO_VALIDATE_LOGITS=1'] if args.resident_logits else []), '-std=c++17','-Wall','-Wextra','-Werror',
           '-I/src/build/upstream/headers','/src/integration/resident_coverage.cpp' if args.resident_coverage else '/src/integration/resident_decode.cpp',
           '-L/src/build/upstream/llama-b12110','-Wl,-rpath,/app',
           '-lllama','-lggml','-lggml-base','-ldl','-o','/output/resident-coverage' if args.resident_coverage else '/output/resident-logits' if args.resident_logits else '/output/resident-decode'] if args.resident or args.resident_logits or args.resident_coverage else [
        '-O2', '-shared', '-fPIC', '-std=c++17', '-Wall', '-Wextra', '-Werror',
        '-I/usr/local/cuda/include', '/src/integration/observe_launches.cpp',
        '-ldl', '-pthread', '-o', '/output/observe_launches.so']),
    ], check=True)
    print('Built resident coverage validator' if args.resident_coverage else 'Built resident logit validator' if args.resident_logits else 'Built resident worker' if args.resident else 'Built observer')


if __name__ == '__main__':
    main()
