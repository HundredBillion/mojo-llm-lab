from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

from run_prefill_bench import bench_args, parse_bench


def output(**changes):
    row = {'build_commit': 'cd959ba', 'n_prompt': 2048, 'n_gen': 0, 'n_batch': 32, 'n_ubatch': 32,
           'n_gpu_layers': 99, 'type_k': 'q4_0', 'type_v': 'q4_0', 'avg_ts': 1000.0, 'stddev_ts': 5.0,
           'samples_ts': [995.0, 1000.0, 1005.0]}
    row.update(changes)
    return 'log line\n' + json.dumps([row])


class PrefillBenchTests(unittest.TestCase):
    def test_arguments_match_server_settings(self):
        args = bench_args(2048, 32, 32, 3)
        for flag, value in (('-p', '2048'), ('-n', '0'), ('-b', '32'), ('-ub', '32'), ('-ctk', 'q4_0'),
                            ('-ctv', 'q4_0'), ('-fa', 'on'), ('-t', '2'), ('--poll', '0'), ('-ngl', '99')):
            self.assertEqual(args[args.index(flag) + 1], value)

    def test_parses_matching_result(self):
        result = parse_bench(output(), 2048, 32, 32, 3)
        self.assertEqual(result['mean_tokens_per_second'], 1000.0)
        self.assertEqual(result['samples_tokens_per_second'], [995.0, 1000.0, 1005.0])

    def test_rejects_setting_mismatch_and_bad_samples(self):
        for change in ({'n_ubatch': 512}, {'n_gen': 16}, {'type_k': 'f16'}, {'n_prompt': 512},
                       {'samples_ts': [1.0, 2.0]}, {'samples_ts': [1.0, 0.0, 2.0]},
                       {'samples_ts': [1.0, float('nan'), 2.0]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse_bench(output(**change), 2048, 32, 32, 3)

    def test_rejects_missing_or_multiple_results(self):
        with self.assertRaises(ValueError):
            parse_bench('no json here', 2048, 32, 32, 3)
        with self.assertRaises(ValueError):
            parse_bench('[]', 2048, 32, 32, 3)


if __name__ == '__main__':
    unittest.main()
