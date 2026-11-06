import hashlib
from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from generate_q4_matvec_fixtures import fixtures
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
