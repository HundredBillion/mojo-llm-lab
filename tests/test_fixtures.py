import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools.generate_fixtures import CASES, serialize_fixture, synthetic_fixture, write_fixtures

HASHES = {
    'distinct_axes': '0d31be72f3191a3d3a09c0ae73c860771940dd02e5d58919a94cf62b763a0a4a',
    'long_position': '129c59e45d9d26bf15d88cbda9ed90a74bf1a6eaca28ce33a09e720340fa832c',
    'fallback_only': '592db0fc67720838ba054d71fc11455235a35a3181d0cbd0a8dfe8290d9836b4',
}


class FixtureTests(unittest.TestCase):
    def test_original_synthetic_payload_hashes_and_lengths(self):
        for name, positions in CASES.items():
            with self.subTest(case=name):
                payload = serialize_fixture(synthetic_fixture(positions))
                self.assertEqual(len(payload), 16912)
                self.assertEqual(hashlib.sha256(payload).hexdigest(), HASHES[name])
                self.assertEqual(struct.unpack_from('<4i', payload, 16896), positions)

    def test_layout_is_head_major_q_then_shared_gamma_then_positions(self):
        payload = serialize_fixture(synthetic_fixture(CASES['distinct_axes']))
        self.assertEqual(struct.unpack_from('<f', payload, 0)[0], 1.0)
        self.assertEqual(struct.unpack_from('<f', payload, 128*4)[0], -1.546875)
        self.assertEqual(struct.unpack_from('<f', payload, 4096*4)[0], 1.0)
        self.assertEqual(struct.unpack_from('<f', payload, 4223*4)[0], 1+127/128)

    def test_writer_manifest_matches_files(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_fixtures(directory)
            saved = json.loads((Path(directory)/'manifest.json').read_text())
            self.assertEqual(manifest, saved)
            self.assertEqual(set(manifest), set(HASHES))
            for name, entry in manifest.items():
                payload = (Path(directory)/entry['filename']).read_bytes()
                self.assertEqual(entry['sha256'], HASHES[name])
                self.assertEqual(entry['bytes'], len(payload))

    def test_wrong_shape_and_nonfinite_values_rejected(self):
        for field, replacement in [('q', [[1.0]*128]), ('gamma', [1.0]*4096),
                                   ('positions_axis_major', [0]*3)]:
            fixture = synthetic_fixture(CASES['distinct_axes'])
            fixture[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                serialize_fixture(fixture)
        fixture = synthetic_fixture(CASES['distinct_axes'])
        fixture['q'][0][0] = float('nan')
        with self.assertRaises(ValueError):
            serialize_fixture(fixture)


if __name__ == '__main__':
    unittest.main()
