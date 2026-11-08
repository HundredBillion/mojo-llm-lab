import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from run_q4_cuda_comparison import validate_2b_fixtures


class TwoBInputValidationTests(unittest.TestCase):
    def test_dimension_hash_and_distinctness_rejections(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = b''.join(i.to_bytes(4, 'little') + bytes(1148) for i in range(6144))
            manifest = {'rows': 6144, 'input_width': 2048, 'gate_up': True, 'files': {}}
            for name, payload in {'weights.bin': weights, 'gate.bin': weights, 'activations.bin': bytes(2304)}.items():
                (root / name).write_bytes(payload)
                manifest['files'][name] = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
            def save():
                (root / 'fixtures.json').write_text(json.dumps(manifest))
            save()
            validate_2b_fixtures(root)
            manifest['input_width'] = 2560
            save()
            with self.assertRaisesRegex(ValueError, 'width 2048'):
                validate_2b_fixtures(root)
            manifest['input_width'] = 2048
            save()
            (root / 'activations.bin').write_bytes(bytes(2303))
            with self.assertRaisesRegex(ValueError, 'size/hash'):
                validate_2b_fixtures(root)
            (root / 'activations.bin').write_bytes(b'x' + bytes(2303))
            with self.assertRaisesRegex(ValueError, 'size/hash'):
                validate_2b_fixtures(root)
            (root / 'activations.bin').write_bytes(bytes(2304))
            repeated = bytes(len(weights))
            (root / 'gate.bin').write_bytes(repeated)
            manifest['files']['gate.bin']['sha256'] = hashlib.sha256(repeated).hexdigest()
            save()
            with self.assertRaisesRegex(ValueError, 'distinct rows'):
                validate_2b_fixtures(root)
