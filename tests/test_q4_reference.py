import struct
import unittest
from q4_reference import decode_block


def packed_block(scales, minima, d=0.5, minimum=0.25):
    meta = bytearray(12)
    for j in range(4):
        meta[j] = (scales[j] & 63) | ((scales[j+4] >> 4) << 6)
        meta[j+4] = (minima[j] & 63) | ((minima[j+4] >> 4) << 6)
        meta[j+8] = (scales[j+4] & 15) | ((minima[j+4] & 15) << 4)
    quants = bytes((i % 16) | (((15-i) % 16) << 4) for i in range(128))
    return struct.pack('<ee', d, minimum) + meta + quants


class Q4LayoutTests(unittest.TestCase):
    def test_all_six_bit_scales_minima_and_both_nibbles(self):
        for value in range(64):
            scales = [(value+j*7) % 64 for j in range(8)]
            minima = [(63-value+j*11) % 64 for j in range(8)]
            actual = decode_block(packed_block(scales,minima))
            expected = [0.5*scales[g]*(lane%16 if g%2==0 else 15-lane%16)-0.25*minima[g]
                        for g in range(8) for lane in range(32)]
            self.assertEqual(actual, expected)

    def test_zero_and_signed_factors(self):
        for d, m in [(0,0),(-0.5,0.25),(0.5,-0.25)]:
            actual=decode_block(packed_block([1]*8,[2]*8,d,m))
            self.assertEqual(actual[0], -2*m)
            self.assertEqual(actual[32], d*15-2*m)

    def test_invalid_layout_and_nonfinite_factors(self):
        for data in [bytes(143),bytes(145),struct.pack('<ee',float('inf'),0)+bytes(140)]:
            with self.assertRaises(ValueError):decode_block(data)
