"""Compile the host harness in a pinned, CPU-limited CUDA container."""
import json
from pathlib import Path
import subprocess
import os

ROOT = Path(__file__).resolve().parents[1]


def main():
    versions = json.loads((ROOT / 'versions.json').read_text())
    output = ROOT / 'build/integration'
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'docker', 'run', '--rm', '--network', 'none', '--read-only',
        '--tmpfs', '/tmp:rw,nosuid,size=256m', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--user', f'{os.getuid()}:{os.getgid()}',
        '--cpu-period', '10000', '--cpu-quota', '1000', '--memory', '1g', '--memory-swap', '1g',
        '--mount', f'type=bind,src={ROOT},dst=/src,readonly',
        '--mount', f'type=bind,src={output},dst=/output', '--entrypoint', 'g++', versions['cuda_image'],
        '-O2', '-shared', '-fPIC', '-std=c++17', '-Wall', '-Wextra', '-Werror',
        '-I/usr/local/cuda/include', '/src/integration/observe_launches.cpp',
        '-ldl', '-pthread', '-o', '/output/observe_launches.so',
    ], check=True)
    print('Built build/integration/observe_launches.so; no GPU was used.')


if __name__ == '__main__':
    main()
