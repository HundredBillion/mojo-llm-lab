import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

HEAD_DIM = 128
HEADS = 32
CASES = {
    'distinct_axes': (3, 5, 11, 17),
    'long_position': (32768, 32768, 32768, 32768),
    'fallback_only': (0, 0, 0, 32768),
}


def synthetic_fixture(positions):
    gamma = [1 + i / 128 for i in range(HEAD_DIM)]
    q = [[(-1 if (i+h) % 2 else 1) * (1 + ((i+3*h) % 8)/8) * (1+h/8)
          for i in range(HEAD_DIM)] for h in range(HEADS)]
    return {'q': q, 'gamma': gamma, 'positions_axis_major': list(positions)}


def serialize_fixture(fixture):
    q, gamma, positions = (fixture[key] for key in ('q', 'gamma', 'positions_axis_major'))
    if (len(q) != HEADS or any(len(row) != HEAD_DIM for row in q)
            or len(gamma) != HEAD_DIM or len(positions) != 4):
        raise ValueError('Expected Q[32,128], gamma[128], positions[4]')
    values = [value for row in q for value in row] + gamma
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Nonfinite fixture input')
    return struct.pack('<4224f4i', *values, *positions)


def write_fixtures(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, positions in CASES.items():
        payload = serialize_fixture(synthetic_fixture(positions))
        filename = name + '.input.bin'
        (output_dir / filename).write_bytes(payload)
        manifest[name] = {'filename': filename, 'bytes': len(payload),
                          'sha256': hashlib.sha256(payload).hexdigest()}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('build/fixtures'))
    args = parser.parse_args()
    print(json.dumps(write_fixtures(args.output_dir), indent=2))


if __name__ == '__main__':
    main()
