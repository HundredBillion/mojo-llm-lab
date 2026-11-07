import hashlib
from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from generate_q4_matvec_fixtures import fixtures, vary_factors, vary_activation_scales
from q4_reference import dot_q4_q8


class MatvecFixtureTests(unittest.TestCase):
    def test_qualified_fixture_hashes(self):
        expected = {'weights.bin': '66201e24c50fceb171ec011470f4703ae88bd984ecb012d0a3d9dfafd9b9b065',
                    'activations.bin': 'dbff591fcb974ae950824639629c928f29ac3c2c859ceb038e741373d5328c4e'}
        self.assertEqual({name:hashlib.sha256(data).hexdigest() for name,data in fixtures().items()},expected)

    def test_varied_quants_preserve_metadata_and_activation_contract(self):
        original=fixtures(); varied=fixtures(random_quants=True)
        self.assertEqual(original['activations.bin'],varied['activations.bin'])
        chunks=[]
        for start in range(0,len(original['weights.bin']),144):
            self.assertEqual(original['weights.bin'][start:start+16],varied['weights.bin'][start:start+16])
            chunks.append(varied['weights.bin'][start+16:start+144])
        self.assertEqual(len(set(chunks)),640)
        self.assertEqual(set(b for chunk in chunks for b in chunk),set(range(256)))

    def test_cuda_quantization_reproduces_input_blocks(self):
        payload=fixtures()['activations.bin']
        for start in range(0,len(payload),36):
            delta,total,*values=struct.unpack_from('<ee32b',payload,start)
            floats=[delta*v for v in values]
            self.assertEqual(max(map(abs,floats))/127,delta)
            self.assertEqual([round(v/delta) for v in floats],values)
            self.assertEqual(total,sum(floats))


class GateFixtureTests(unittest.TestCase):
    def test_distinct_gate_weights_cover_nonlinear_regions(self):
        baseline = fixtures(random_quants=True)
        fused = fixtures(random_quants=True, gate_up=True)
        self.assertEqual(fused['weights.bin'], baseline['weights.bin'])
        self.assertEqual(fused['activations.bin'], baseline['activations.bin'])
        self.assertEqual(len(fused['gate.bin']), len(fused['weights.bin']))
        self.assertNotEqual(fused['gate.bin'], fused['weights.bin'])
        values = [dot_q4_q8(fused['gate.bin'][i*1440:(i+1)*1440], fused['activations.bin']) for i in range(64)]
        self.assertEqual(values[0], 0)
        self.assertGreater(abs(dot_q4_q8(fused['weights.bin'][:1440], fused['activations.bin'])), 1)
        self.assertTrue(any(v == 0 for v in values))
        self.assertTrue(any(0 < v < 1 for v in values))
        self.assertTrue(any(-1 < v < 0 for v in values))
        self.assertTrue(any(v > 10 for v in values))
        self.assertTrue(any(v < -10 for v in values))


class FullRowFixtureTests(unittest.TestCase):
    def test_extension_preserves_inputs_and_adds_distinct_rows(self):
        small = fixtures(random_quants=True, gate_up=True)
        extended = fixtures(random_quants=True, gate_up=True, rows=128)
        self.assertEqual(small['activations.bin'], extended['activations.bin'])
        for name in ('weights.bin', 'gate.bin'):
            self.assertEqual(len(extended[name]), 128 * 1440)
            self.assertEqual(extended[name][:64*1440], small[name])
            rows = [extended[name][i*1440:(i+1)*1440] for i in range(128)]
            self.assertEqual(len(set(rows)), 128)

    def test_invalid_row_counts_are_rejected(self):
        for rows in (0, 1, 63, 65, 9792):
            with self.assertRaises(ValueError):
                fixtures(rows=rows)


class VariedFactorTests(unittest.TestCase):
    def test_only_factors_change_and_sign_zero_are_preserved(self):
        base = fixtures(random_quants=True, gate_up=True)
        varied = fixtures(random_quants=True, gate_up=True, factor_seed=739)
        self.assertEqual(base['activations.bin'], varied['activations.bin'])
        for name in ('weights.bin', 'gate.bin'):
            self.assertNotEqual(base[name], varied[name])
            for offset in range(0, len(base[name]), 144):
                self.assertEqual(base[name][offset+4:offset+144], varied[name][offset+4:offset+144])
                for before, after in zip(struct.unpack_from('<ee', base[name], offset), struct.unpack_from('<ee', varied[name], offset)):
                    if before == 0:
                        self.assertEqual(after, 0)
                    else:
                        self.assertGreaterEqual(after/before, 0.749)
                        self.assertLessEqual(after/before, 1.251)

    def test_seed_reproducibility_and_variation(self):
        payload = fixtures()['weights.bin']
        self.assertEqual(vary_factors(payload, 739), vary_factors(payload, 739))
        self.assertNotEqual(vary_factors(payload, 739), vary_factors(payload, 740))

    def test_bad_lengths_and_nonfinite_factors_rejected(self):
        for payload in (b'', bytes(143), bytes(145), struct.pack('<ee',float('nan'),1)+bytes(140)):
            with self.assertRaises(ValueError):
                vary_factors(payload, 739)


class ActivationScaleTests(unittest.TestCase):
    def test_scales_vary_while_quantized_values_and_cached_sums_match(self):
        base = fixtures()['activations.bin']
        varied = vary_activation_scales(base, 971)
        scales = set()
        for offset in range(0, len(base), 36):
            self.assertEqual(base[offset+4:offset+36], varied[offset+4:offset+36])
            delta, total, *values = struct.unpack_from('<ee32b', varied, offset)
            scales.add(delta)
            self.assertEqual(total, sum(values)*delta)
            self.assertEqual(max(abs(v*delta) for v in values)/127, delta)
        self.assertEqual(len(scales), 7)
        self.assertEqual(varied, vary_activation_scales(base, 971))

    def test_invalid_activation_blocks_rejected(self):
        for data in (b'', bytes(35), bytes(36), struct.pack('<ee32b',float('inf'),0,*([127]*32))):
            with self.assertRaises(ValueError):vary_activation_scales(data, 971)


class TwoBFixtureTests(unittest.TestCase):
    def test_width_and_extension_preserve_quantization_contract(self):
        small = fixtures(random_quants=True, gate_up=True, input_width=2048)
        extended = fixtures(random_quants=True, gate_up=True, rows=128, input_width=2048)
        self.assertEqual(len(small['activations.bin']), 2304)
        self.assertEqual(small['activations.bin'], extended['activations.bin'])
        for name in ('weights.bin', 'gate.bin'):
            self.assertEqual(len(extended[name]), 128 * 1152)
            self.assertEqual(extended[name][:64 * 1152], small[name])
            self.assertEqual(len({extended[name][i*1152:(i+1)*1152] for i in range(128)}), 128)
        for start in range(0, 2304, 36):
            delta, total, *values = struct.unpack_from('<ee32b', small['activations.bin'], start)
            self.assertEqual(max(abs(delta*v) for v in values)/127, delta)
            self.assertEqual(total, sum(values)*delta)
        self.assertEqual(dot_q4_q8(small['gate.bin'][:1152], small['activations.bin']), 0)
        self.assertNotEqual(dot_q4_q8(small['weights.bin'][:1152], small['activations.bin']), 0)

    def test_unsupported_width_rejected(self):
        for width in (0, 1024, 2049, 4096):
            with self.assertRaises(ValueError):
                fixtures(input_width=width)
