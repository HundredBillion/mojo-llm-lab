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
    parser.add_argument('--shape', choices=['4b', '2b'], default='4b')
    parser.add_argument('--threads', type=int, choices=[32,64,128,256], default=32)
    parser.add_argument('--repetitions', type=int, choices=[32,256], default=32)
    parser.add_argument('--warmup-pairs', type=int, choices=[2,16], default=2)
    parser.add_argument('--prepacked', action='store_true')
    parser.add_argument('--gate-up', action='store_true')
    args = parser.parse_args()
    if args.gate_up and (args.prepacked or args.threads not in ((64,128,256) if args.shape == '2b' else (128,))):
        parser.error('--gate-up requires native layout and a supported thread count')
    if args.shape == '4b' and args.threads in (64,256):
        parser.error('64/256 threads are only qualified for 2B gate/up')
    if args.shape == '2b' and not args.gate_up:
        parser.error('--shape 2b requires --gate-up')
    executable = 'q4-compare' + (f'-{args.threads}' if args.threads != 32 else '') + ('-split' if args.prepacked else '') + ('-gate' if args.gate_up else '') + ('-2b' if args.shape == '2b' else '')
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
        '-O2', *(['-DINPUT_WIDTH=2048', '-DROWS=6144'] if args.shape == '2b' else []), *(['-DGATE_UP=1'] if args.gate_up else []), *(['-DPREPACKED=1'] if args.prepacked else []), f'-DMOJO_THREADS={args.threads}', f'-DREPS={args.repetitions}', f'-DWARMUP_PAIRS={args.warmup_pairs}', '-std=gnu11', '-Wall', '-Wextra', '-Werror',
        '-I/src/build/upstream/headers', '-I/usr/local/cuda/include', '/src/benchmark/q4_compare.c',
        '-L/usr/local/cuda/lib64/stubs', '-lcuda', '-ldl', '-lm', '-rdynamic', '-o', '/output/' + executable,
    ], check=True)
    print('Built build/q4-matvec/' + executable + '; no GPU was used.')


if __name__ == '__main__':
    main()
