"""Compile the host harness in a pinned, CPU-limited CUDA container."""
import hashlib
import json
from pathlib import Path
import subprocess
import os

ROOT = Path(__file__).resolve().parents[1]


def main():
    versions = json.loads((ROOT / 'versions.json').read_text())
    for name, expected in versions['header_sha256'].items():
        if hashlib.sha256((ROOT / 'build/upstream/headers' / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Run fetch_upstream.py: mismatched header ' + name)
    output = ROOT / 'build/benchmark'
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'docker', 'run', '--rm', '--network', 'none', '--read-only',
        '--tmpfs', '/tmp:rw,nosuid,size=256m', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--user', f'{os.getuid()}:{os.getgid()}',
        '--cpu-period', '5000', '--cpu-quota', '1000', '--memory', '1g', '--memory-swap', '1g',
        '--mount', f'type=bind,src={ROOT},dst=/src,readonly',
        '--mount', f'type=bind,src={output},dst=/output', '--entrypoint', 'gcc', versions['cuda_image'],
        '-O2', '-std=gnu11', '-Wall', '-Wextra', '-Werror',
        '-I/src/build/upstream/headers', '-I/usr/local/cuda/include', '/src/benchmark/bench.c',
        '-L/usr/local/cuda/lib64/stubs', '-lcuda', '-ldl', '-lm', '-rdynamic', '-o', '/output/bench',
    ], check=True)
    print('Built build/benchmark/bench; no GPU was used.')


if __name__ == '__main__':
    main()
