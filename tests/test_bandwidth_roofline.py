from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

import run_bandwidth_roofline as R


class RooflineExpectationTests(unittest.TestCase):
    def test_buffer_matches_vocabulary_projection_size(self):
        self.assertEqual(R.BYTES, 417177600)
        self.assertEqual(R.WORDS * 8, R.BYTES)

    def test_parity_matches_explicit_count_for_small_strides(self):
        word = (R.PATTERN << 32) | R.PATTERN
        for grid in (640, 1280):
            expected = R.expected_accumulators(grid)
            stride = grid * 128
            for g in (0, 1, stride // 2, stride - 1):
                count = len(range(g, R.WORDS, stride))
                self.assertEqual(expected[g], word if count % 2 else 0)

    def test_entry_requires_single_visible_kernel(self):
        self.assertEqual(R.entry_name('.visible .entry k(\n.param .u64 a\n)'), 'k')
        with self.assertRaises(ValueError):
            R.entry_name('.visible .entry a(\n)\n.visible .entry b(\n)')


if __name__ == '__main__':
    unittest.main()
