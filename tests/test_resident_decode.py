import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from run_resident_decode import validate_trial, verify_wait_policy

class ResidentEvidenceTests(unittest.TestCase):
    def row(self):
        return dict(mojo_arm=1,prime_tokens=0,arm=1,tokens=32,decode_ns=100,top_token=7,cpu_usage_usec=2,
                    nr_throttled=1,throttled_usec=4,replacement_calls=10,captured_calls=5,
                    timed_replacement_calls=0,timed_captured_calls=0)

    def test_cuda_cannot_claim_replacement(self):
        row=self.row();row['arm']=0
        with self.assertRaises(ValueError):validate_trial(row,0,0,'mojo')
        self.assertEqual(validate_trial(row,0,10,'mojo'),10)
        with self.assertRaises(ValueError):validate_trial(self.row(),1,0,'control')

    def test_short_trial_requires_explicit_workload(self):
        row={**self.row(),'tokens':16}
        self.assertEqual(validate_trial(row,1,0,'mojo',tokens=16),10)
        with self.assertRaises(ValueError):validate_trial(row,1,0,'mojo')

    def test_candidate_requires_capture_and_consistent_counters(self):
        self.assertEqual(validate_trial(self.row(),1,0,'mojo'),10)
        for change in [dict(captured_calls=0),dict(captured_calls=11),dict(decode_ns=0),
                       dict(tokens=16),dict(arm=0),dict(throttled_usec=-1)]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_trial({**self.row(),**change},1,0,'mojo')

    def test_timed_capture_must_fit_trial_and_cumulative_counters(self):
        row={**self.row(),'timed_replacement_calls':4,'timed_captured_calls':3}
        self.assertEqual(validate_trial(row,1,5,'mojo'),10)
        for change in [dict(timed_replacement_calls=-1),dict(timed_captured_calls=5),
                       dict(timed_replacement_calls=6),dict(timed_captured_calls=None)]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_trial({**row,**change},1,5,'mojo')
        del row['timed_captured_calls']
        with self.assertRaises(ValueError):validate_trial(row,1,5,'mojo')

    def test_timed_noncaptures_cannot_exceed_entire_history(self):
        row={**self.row(),'captured_calls':10,
             'timed_replacement_calls':4,'timed_captured_calls':3}
        with self.assertRaises(ValueError):validate_trial(row,1,5,'mojo')

    def test_priming_metadata_must_match_requested_workload(self):
        row={**self.row(),'prime_tokens':2}
        self.assertEqual(validate_trial(row,1,0,'mojo',prime_tokens=2),10)
        with self.assertRaises(ValueError):validate_trial(row,1,0,'mojo')
        del row['prime_tokens']
        with self.assertRaises(ValueError):validate_trial(row,1,0,'mojo',prime_tokens=2)
        with self.assertRaises(ValueError):validate_trial(row,1,0,'mojo')

    def test_candidate_can_be_context_zero_without_mislabeling_cuda(self):
        row={**self.row(),'arm':0,'mojo_arm':0}
        self.assertEqual(validate_trial(row,0,0,'mojo',mojo_arm=0),10)
        with self.assertRaises(ValueError):validate_trial(row,0,0,'mojo',mojo_arm=1)
        with self.assertRaises(ValueError):validate_trial({**row,'arm':1},1,0,'mojo',mojo_arm=0)
        with self.assertRaises(ValueError):validate_trial({**row,'captured_calls':0},0,0,'mojo',mojo_arm=0)


class WaitPolicyTests(unittest.TestCase):
    def test_verified_policy_preserves_unrelated_flag_bits(self):
        self.assertEqual(verify_wait_policy('LLAMA_CUDA_WAIT device=0 policy=blocking flags=12 verified=1\n','blocking')['flags'],12)
        self.assertEqual(verify_wait_policy('LLAMA_CUDA_WAIT device=0 policy=auto flags=0 verified=1\n','auto')['flags'],0)

    def test_missing_wrong_or_ambiguous_wait_evidence_fails(self):
        good='LLAMA_CUDA_WAIT device=0 policy=blocking flags=4 verified=1\n'
        for text in ['',good+good,good.replace('blocking','auto'),good.replace('flags=4','flags=0'),good.replace('device=0','device=1'),good.replace('verified=1','verified=0')]:
            with self.subTest(text=text),self.assertRaises(ValueError):verify_wait_policy(text,'blocking')


class CleanupTests(unittest.TestCase):
    def test_client_and_logger_are_reaped_when_removal_raises(self):
        from unittest.mock import Mock,patch
        from run_benchmark import cleanup
        client=Mock();client.poll.return_value=None
        logger=Mock();logger.poll.return_value=None
        with patch('run_benchmark.remove_owned',side_effect=RuntimeError('daemon unavailable')):
            errors=cleanup('owned','token',[client,logger])
        self.assertEqual(errors,['daemon unavailable'])
        client.terminate.assert_called_once();client.wait.assert_called_once()
        logger.terminate.assert_called_once();logger.wait.assert_called_once()


class ResidentCapabilityTests(unittest.TestCase):
    def test_long_trial_requires_full_fixture_and_worker_support(self):
        from run_resident_decode import validate_ready
        row=dict(ready=True,prime_tokens=2,mojo_arm=1,max_timed_tokens=128,fixture_tokens=138)
        validate_ready(row,2,1,128)
        for change in (dict(fixture_tokens=137),dict(max_timed_tokens=32),dict(fixture_tokens=True),
                       dict(prime_tokens=1),dict(mojo_arm=0),dict(ready=False)):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_ready({**row,**change},2,1,128)
        del row['max_timed_tokens']
        with self.assertRaises(ValueError):validate_ready(row,2,1,128)

    def test_short_trial_remains_supported_by_new_worker(self):
        from run_resident_decode import validate_ready
        validate_ready(dict(ready=True,prime_tokens=0,mojo_arm=0,max_timed_tokens=128,fixture_tokens=138),0,0,16)


class WholeLogitReportTests(unittest.TestCase):
    def report(self):
        return dict(passed=True,vocabulary=3,positions=19,atol=1e-4,rtol=1e-5,
                    candidate_values=171,candidate_positions=57,candidate_violations=0,candidate_top_mismatches=0,candidate_max_abs=0,candidate_max_tolerance_ratio=0,
                    control_values=114,control_positions=38,control_violations=0,control_top_mismatches=0,control_max_abs=0,control_max_tolerance_ratio=0)

    def test_requires_complete_finite_bounded_comparisons(self):
        import json
        from run_resident_decode import validate_logits_report
        report=self.report()
        text='mojo_logits '+json.dumps(report)+'\n'
        self.assertEqual(validate_logits_report(text,16,2,1),report)
        for change in (dict(candidate_values=168),dict(control_values=0),dict(atol=1),dict(candidate_max_tolerance_ratio=1.1),
                       dict(candidate_top_mismatches=1),dict(candidate_max_abs=float('nan')),dict(passed=False),dict(vocabulary=True)):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_logits_report('mojo_logits '+json.dumps({**report,**change})+'\n',16,2,1)
        for bad in ('',text+text):
            with self.assertRaises(ValueError):validate_logits_report(bad,16,2,1)

    def test_validation_output_cannot_be_summarized_as_speed(self):
        import json,tempfile
        from summarize_resident import summarize
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'summary.json'
            for metadata in ({'validation_only':True},{'protocol':'resident-logit-validation-v1'}):
                path.write_text(json.dumps(dict(status='passed',**metadata)))
                with self.assertRaisesRegex(ValueError,'not performance evidence'):summarize(path)
