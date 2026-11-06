"""Compile the host harness in a pinned, CPU-limited CUDA container."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import os

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, choices=[32,128], default=32)
    parser.add_argument('--repetitions', type=int, choices=[32,256], default=32)
    parser.add_argument('--warmup-pairs', type=int, choices=[2,16], default=2)
    parser.add_argument('--prepacked', action='store_true')
    parser.add_argument('--gate-up', action='store_true')
    args = parser.parse_args()
    if args.gate_up and (args.prepacked or args.threads != 128):
        parser.error('--gate-up requires 128 threads and native layout')
    executable = 'q4-compare' + ('-128' if args.threads == 128 else '') + ('-split' if args.prepacked else '') + ('-gate' if args.gate_up else '')
    versions = json.loads((ROOT / 'versions.json').read_text())
    for name, expected in versions['header_sha256'].items():
        if hashlib.sha256((ROOT / 'build/upstream/headers' / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Run fetch_upstream.py: mismatched header ' + name)
    output = ROOT / 'build/q4-matvec'
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'docker', 'run', '--rm', '--network', 'none', '--read-only',
        '--tmpfs', '/tmp:rw,nosuid,size=256m', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--user', f'{os.getuid()}:{os.getgid()}',
        '--cpu-period', '5000', '--cpu-quota', '1000', '--memory', '1g', '--memory-swap', '1g',
        '--mount', f'type=bind,src={ROOT},dst=/src,readonly',
        '--mount', f'type=bind,src={output},dst=/output', '--entrypoint', 'gcc', versions['cuda_image'],
        '-O2', *(['-DGATE_UP=1'] if args.gate_up else []), *(['-DPREPACKED=1'] if args.prepacked else []), f'-DMOJO_THREADS={args.threads}', f'-DREPS={args.repetitions}', f'-DWARMUP_PAIRS={args.warmup_pairs}', '-std=gnu11', '-Wall', '-Wextra', '-Werror',
        '-I/src/build/upstream/headers', '-I/usr/local/cuda/include', '/src/benchmark/q4_compare.c',
        '-L/usr/local/cuda/lib64/stubs', '-lcuda', '-ldl', '-lm', '-rdynamic', '-o', '/output/' + executable,
    ], check=True)
    print('Built build/q4-matvec/' + executable + '; no GPU was used.')


if __name__ == '__main__':
    main()
