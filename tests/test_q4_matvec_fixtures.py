import hashlib
from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from generate_q4_matvec_fixtures import fixtures


class MatvecFixtureTests(unittest.TestCase):
    def test_qualified_fixture_hashes(self):
        expected = {'weights.bin': '66201e24c50fceb171ec011470f4703ae88bd984ecb012d0a3d9dfafd9b9b065',
                    'activations.bin': 'dbff591fcb974ae950824639629c928f29ac3c2c859ceb038e741373d5328c4e'}
        self.assertEqual({name:hashlib.sha256(data).hexdigest() for name,data in fixtures().items()},expected)

    def test_cuda_quantization_reproduces_input_blocks(self):
        payload=fixtures()['activations.bin']
        for start in range(0,len(payload),36):
            delta,total,*values=struct.unpack_from('<ee32b',payload,start)
            floats=[delta*v for v in values]
            self.assertEqual(max(map(abs,floats))/127,delta)
            self.assertEqual([round(v/delta) for v in floats],values)
            self.assertEqual(total,sum(floats))
