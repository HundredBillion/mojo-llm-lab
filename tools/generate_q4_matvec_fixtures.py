"""Generate synthetic, quantization-matched inputs for the fixed-width prototype."""
import hashlib
import json
from pathlib import Path
import random
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_q4_reference import packed_block


def fixtures():
    rng = random.Random(42)
    weights = []
    for _ in range(64 * 10):
        weights.append(packed_block([rng.randrange(64) for _ in range(8)],
                                    [rng.randrange(64) for _ in range(8)], 1/512, 1/1024))
    activations = []
    for _ in range(80):
        values = [rng.randrange(-127, 128) for _ in range(32)]
        values[0] = 127
        activations.append(struct.pack('<ee32b', 1/128, sum(values)/128, *values))
    return {'weights.bin': b''.join(weights), 'activations.bin': b''.join(activations)}


def main():
    directory = ROOT / 'build/q4-matvec'
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'rows': 64, 'input_width': 2560, 'seed': 42, 'files': {}}
    for name, payload in fixtures().items():
        (directory / name).write_bytes(payload)
        manifest['files'][name] = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    (directory / 'fixtures.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
