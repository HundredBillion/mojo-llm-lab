"""Generate synthetic, quantization-matched inputs for the fixed-width prototype."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_q4_reference import packed_block


def vary_factors(payload, seed):
    if not payload or len(payload) % 144:
        raise ValueError('Require whole Q4_K blocks')
    rng = random.Random(seed)
    result = bytearray(payload)
    for offset in range(0, len(payload), 144):
        factors = struct.unpack_from('<ee', payload, offset)
        if not all(math.isfinite(value) for value in factors):
            raise ValueError('Require finite Q4_K factors')
        struct.pack_into('<ee', result, offset,
                         *(value * rng.uniform(0.75, 1.25) for value in factors))
    return bytes(result)


def vary_activation_scales(payload, seed):
    if not payload or len(payload) % 36:
        raise ValueError('Require whole Q8_1 blocks')
    rng = random.Random(seed)
    result = bytearray()
    for offset in range(0, len(payload), 36):
        delta, _, *values = struct.unpack_from('<ee32b', payload, offset)
        if not math.isfinite(delta) or delta <= 0 or max(map(abs, values)) != 127:
            raise ValueError('Require positive finite scale and matching quantization range')
        delta *= 2 ** rng.randrange(-3, 4)
        result.extend(struct.pack('<ee32b', delta, delta * sum(values), *values))
    return bytes(result)


def fixtures(random_quants=False, gate_up=False, rows=64, factor_seed=None, activation_scale_seed=None):
    if rows < 64 or rows > 9728 or rows % 64:
        raise ValueError('rows must be a multiple of 64 between 64 and 9728')
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
    for _ in range((rows - 64) * 10):
        weights.append(packed_block([rng.randrange(64) for _ in range(8)],
                                    [rng.randrange(64) for _ in range(8)], 1/512, 1/1024))
    if random_quants:
        quant_rng = random.Random(12345)
        weights = [block[:16] + (bytes(quant_rng.randrange(256) for _ in range(128))
                   if index < 640 else quant_rng.randbytes(128))
                   for index, block in enumerate(weights)]
    result = {'weights.bin': b''.join(weights), 'activations.bin': b''.join(activations)}
    if gate_up:
        gate_rng = random.Random(54321)
        gate = []
        for index, block in enumerate(weights):
            factor = (0, 1, -1, 1/16, -1/16, 8, -8, 1)[(index // 10) % 8]
            gate.append(struct.pack('<ee', factor/8192, factor/16384) + block[4:16] +
                        (bytes(gate_rng.randrange(256) for _ in range(128))
                         if index < 640 else gate_rng.randbytes(128)))
        result['gate.bin'] = b''.join(gate)
    if factor_seed is not None:
        result['weights.bin'] = vary_factors(result['weights.bin'], factor_seed)
        if gate_up:
            result['gate.bin'] = vary_factors(result['gate.bin'], factor_seed + 1)
    if activation_scale_seed is not None:
        result['activations.bin'] = vary_activation_scales(result['activations.bin'], activation_scale_seed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'build/q4-matvec')
    parser.add_argument('--random-quants', action='store_true')
    parser.add_argument('--gate-up', action='store_true')
    parser.add_argument('--rows', type=int, choices=[64,9728], default=64)
    parser.add_argument('--factor-seed', type=int)
    parser.add_argument('--activation-scale-seed', type=int)
    args = parser.parse_args()
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'activation_scale_seed': args.activation_scale_seed, 'factor_seed': args.factor_seed, 'rows': args.rows, 'input_width': 2560, 'seed': 42, 'random_quants': args.random_quants, 'gate_up': args.gate_up, 'gate_seed': 54321 if args.gate_up else None, 'quant_seed': 12345 if args.random_quants else None, 'files': {}}
    for name, payload in fixtures(random_quants=args.random_quants, gate_up=args.gate_up, rows=args.rows, factor_seed=args.factor_seed, activation_scale_seed=args.activation_scale_seed).items():
        (directory / name).write_bytes(payload)
        manifest['files'][name] = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    (directory / 'fixtures.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
