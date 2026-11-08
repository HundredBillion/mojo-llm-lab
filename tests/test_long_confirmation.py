import sys
from pathlib import Path
import unittest
import json
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from verify_long_confirmation import decision, verify


class ConfirmationDecisionTests(unittest.TestCase):
    def test_recorded_evidence_and_reject_mislabeled_cuda(self):
        evidence = Path(__file__).resolve().parents[1] / 'results/a2000/long-resident'
        self.assertTrue(verify(evidence)['passed'])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for source in evidence.glob('confirmation-*/summary.json'):
                destination = target / source.parent.name / source.name
                destination.parent.mkdir()
                raw = json.loads(source.read_text())
                if raw['mode'] == 'mojo':
                    for row in raw['trials']:
                        row['captured_calls'] = row['replacement_calls'] = 0
                destination.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, 'capture missing'):
                verify(target)

    def test_positive_with_control_margin(self):
        self.assertTrue(decision([1.02] * 4, [1.001, .999])['passed'])

    def test_one_loss_or_large_control_difference_rejects(self):
        self.assertFalse(decision([1.02, 1.02, .999, 1.02], [1., 1.])['passed'])
        self.assertFalse(decision([1.02] * 4, [1.018, 1.])['passed'])

    def test_missing_or_nonfinite_results_reject(self):
        with self.assertRaises(ValueError):
            decision([1.02] * 3, [1., 1.])
        with self.assertRaises(ValueError):
            decision([float('nan')] * 4, [1., 1.])
