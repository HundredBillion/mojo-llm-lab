import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_benchmark


class GuardTests(unittest.TestCase):
    def test_stop_at_limits_or_ac_loss(self):
        normal = {'cpu_C': 69, 'gpu_C': 65, 'platform_C': 60, 'ac': True}
        self.assertTrue(run_benchmark.safe(normal))
        for key, value in [('cpu_C', 90), ('gpu_C', 82), ('platform_C', 75), ('ac', False)]:
            with self.subTest(key=key):
                self.assertFalse(run_benchmark.safe(dict(normal, **{key: value})))

    def test_foreign_container_never_removed(self):
        inspect = subprocess.CompletedProcess([], 0, json.dumps([
            {'Config': {'Labels': {'io.mojo-imrope.benchmark': 'someone-else'}}}
        ]))
        with patch.object(run_benchmark.S, 'run', return_value=inspect) as command:
            with self.assertRaisesRegex(RuntimeError, 'ownership'):
                run_benchmark.remove_owned('test-container', 'our-token')
        self.assertEqual(command.call_count, 1)

    def test_only_owned_container_removed(self):
        inspect = subprocess.CompletedProcess([], 0, json.dumps([
            {'Config': {'Labels': {'io.mojo-imrope.benchmark': 'our-token'}}}
        ]))
        with patch.object(run_benchmark.S, 'run', return_value=inspect) as command:
            run_benchmark.remove_owned('test-container', 'our-token')
        self.assertEqual(command.call_args_list[-1].args[0], ['docker', 'rm', '-f', 'test-container'])

    def test_missing_container_is_not_recreated_or_removed(self):
        with patch.object(run_benchmark.S, 'run', return_value=subprocess.CompletedProcess([], 1)) as command:
            run_benchmark.remove_owned('missing', 'our-token')
        self.assertEqual(command.call_count, 1)


if __name__ == '__main__':
    unittest.main()
