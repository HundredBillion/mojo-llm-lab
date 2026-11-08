from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from run_logit_coverage import schedule_arms, validate_ready, validate_report, validate_trials
from generate_coverage_prompts import prompts, write_prompts

STEPS = 4
VOCAB = 10


def trials(mojo_arm=1, mutate=None):
    rows, total, captured = [], 0, 0
    for index, arm in enumerate(schedule_arms(mojo_arm)):
        candidate = arm == mojo_arm
        replaced = 5 if candidate else 0
        caught = 2 if candidate else 0
        total += replaced; captured += caught
        rows.append({'trial': index, 'arm': arm, 'candidate': candidate, 'replacement_calls': total,
                     'captured_calls': captured, 'trial_replacement_calls': replaced,
                     'trial_captured_calls': caught, 'first_eog': -1, 'tokens': [1, 2, 3, 4], 'text': 'abcd'})
    if mutate:
        mutate(rows)
    return rows


def report(**changes):
    positions = STEPS + 1
    row = {'passed': True, 'vocabulary': VOCAB, 'positions': positions, 'atol': 1e-4, 'rtol': 1e-5,
           'candidate_values': 3 * positions * VOCAB, 'candidate_positions': 3 * positions,
           'candidate_violations': 0, 'candidate_top_mismatches': 0, 'candidate_max_abs': 0.0,
           'candidate_max_tolerance_ratio': 0.0,
           'control_values': 2 * positions * VOCAB, 'control_positions': 2 * positions,
           'control_violations': 0, 'control_top_mismatches': 0, 'control_max_abs': 0.0,
           'control_max_tolerance_ratio': 0.0}
    row.update(changes)
    return 'noise\nmojo_logits ' + json.dumps(row) + '\n'


class CoverageScheduleTests(unittest.TestCase):
    def test_schedule_starts_with_cuda_reference_and_balances_roles(self):
        self.assertEqual(schedule_arms(1), [0, 1, 1, 0, 1, 0])
        self.assertEqual(schedule_arms(0), [1, 0, 0, 1, 0, 1])


class CoverageTrialTests(unittest.TestCase):
    def test_accepts_identical_generation_with_capture(self):
        summary = validate_trials(trials(), 1, STEPS)
        self.assertEqual(summary['tokens'], [1, 2, 3, 4])
        self.assertEqual(summary['candidate_captured_calls'], 6)

    def test_rejects_generated_token_divergence(self):
        def change(rows): rows[2]['tokens'] = [1, 2, 9, 4]
        with self.assertRaisesRegex(ValueError, 'first token index 2'):
            validate_trials(trials(mutate=change), 1, STEPS)

    def test_rejects_text_divergence_with_equal_tokens(self):
        def change(rows): rows[4]['text'] = 'abce'
        with self.assertRaises(ValueError):
            validate_trials(trials(mutate=change), 1, STEPS)

    def test_rejects_replacement_in_cuda_context(self):
        def change(rows):
            rows[3]['trial_replacement_calls'] = 1
            for row in rows[3:]: row['replacement_calls'] += 1
        with self.assertRaisesRegex(ValueError, 'CUDA context'):
            validate_trials(trials(mutate=change), 1, STEPS)

    def test_rejects_missing_capture(self):
        def change(rows):
            for row in rows: row['captured_calls'] = row['trial_captured_calls'] = 0
        with self.assertRaisesRegex(ValueError, 'capture missing'):
            validate_trials(trials(mutate=change), 1, STEPS)

    def test_rejects_incomplete_or_reordered_trials(self):
        with self.assertRaises(ValueError):
            validate_trials(trials()[:-1], 1, STEPS)
        with self.assertRaises(ValueError):
            validate_trials(trials(mojo_arm=0), 1, STEPS)

    def test_rejects_short_generation(self):
        def change(rows):
            for row in rows: row['tokens'] = [1, 2, 3]
        with self.assertRaises(ValueError):
            validate_trials(trials(mutate=change), 1, STEPS)


class CoverageReportTests(unittest.TestCase):
    def test_accepts_exact_complete_report(self):
        self.assertTrue(validate_report(report(), STEPS, VOCAB)['passed'])

    def test_rejects_violation_top_mismatch_and_tolerance_change(self):
        for change in ({'candidate_violations': 1}, {'control_top_mismatches': 1},
                       {'atol': 1e-3}, {'passed': False}, {'candidate_max_tolerance_ratio': 1.5},
                       {'control_max_abs': float('nan')}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_report(report(**change), STEPS, VOCAB)

    def test_rejects_incomplete_counts(self):
        with self.assertRaises(ValueError):
            validate_report(report(candidate_positions=14), STEPS, VOCAB)
        with self.assertRaises(ValueError):
            validate_report(report(), STEPS + 1, VOCAB)
        with self.assertRaises(ValueError):
            validate_report(report() + report(), STEPS, VOCAB)


class CoverageReadyTests(unittest.TestCase):
    def ready(self, **changes):
        row = {'ready': True, 'prompt_tokens': 600, 'n_ctx': 768, 'steps': 128, 'policy': 'sampled',
               'seed': 7, 'mojo_arm': 0, 'vocabulary': VOCAB}
        row.update(changes)
        return row

    def test_accepts_matching_handshake(self):
        validate_ready(self.ready(), 'sampled', 7, 128, 0)

    def test_rejects_mismatches(self):
        for change in ({'policy': 'greedy'}, {'seed': 8}, {'mojo_arm': 1}, {'n_ctx': 700}, {'n_ctx': 8448},
                       {'ready': False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_ready(self.ready(**change), 'sampled', 7, 128, 0)


class CoveragePromptTests(unittest.TestCase):
    def test_prompts_are_deterministic_and_match_committed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_prompts(Path(directory))
        committed = json.loads((ROOT / 'experiments/coverage-prompts/manifest.json').read_text())
        self.assertEqual(manifest, committed)
        self.assertEqual(set(prompts()), {'statement-short', 'statement-medium', 'statement-long', 'prose'})

    def test_prompts_use_chat_format_without_thinking(self):
        for text in prompts().values():
            self.assertTrue(text.startswith('<|im_start|>system\n'))
            self.assertTrue(text.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'))


if __name__ == '__main__':
    unittest.main()
