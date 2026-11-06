import struct
import unittest
from q4_reference import decode_q8_1, dot_q4_q8
from test_q4_reference import packed_block


class Q4DotTests(unittest.TestCase):
    def test_signed_activation_bytes_and_ignored_cached_sum(self):
        values = [-128,-127,-1,0,1,126,127,3]*4
        block = struct.pack('<ee32b',0.125,123,*values)
        self.assertEqual(decode_q8_1(block),[v/8 for v in values])

    def test_multiple_blocks_and_nonuniform_group_deltas(self):
        weights=packed_block([1]*8,[0]*8,d=1,minimum=0)*10
        groups=[struct.pack('<ee32b',(g+1)/8,0,*([1]*32)) for g in range(8)]
        activations=b''.join(groups)*10
        self.assertEqual(dot_q4_q8(weights,activations),10*240*sum((g+1)/8 for g in range(8)))

    def test_cancellation_and_minimum_term(self):
        weights=packed_block([0]*8,[1]*8,d=0,minimum=0.5)
        alternating=[1,-1]*16
        self.assertEqual(dot_q4_q8(weights,struct.pack('<ee32b',1,0,*alternating)*8),0)
        self.assertEqual(dot_q4_q8(weights,struct.pack('<ee32b',1,32,*([1]*32))*8),-128)

    def test_reject_wrong_sizes(self):
        for w,a in [(b'',b''),(bytes(144),bytes(36)),(bytes(145),bytes(288))]:
            with self.assertRaises(ValueError):dot_q4_q8(w,a)
        with self.assertRaises(ValueError):decode_q8_1(bytes(35))
