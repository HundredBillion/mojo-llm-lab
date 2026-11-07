import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from summarize_resident import summarize


class CrossoverSummaryTests(unittest.TestCase):
    def evidence(self, arm=1, mode='mojo'):
        return dict(status='passed',mode=mode,cycles=1,tokens=16,mojo_arm=arm,
                    trials=[dict(arm=a,mojo_arm=arm,tokens=16,warmup=i<2,
                                 decode_ns=200 if a==0 else 100,throttled_usec=0)
                            for i,a in enumerate([0,1,0,1,1,0])])

    def summary(self, evidence):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'summary.json'
            path.write_text(json.dumps(evidence))
            return summarize(path)

    def test_reversing_candidate_context_reverses_cuda_over_mojo_ratio(self):
        self.assertEqual(self.summary(self.evidence(1))['cuda_over_mojo_latency_ratios'],[2])
        self.assertEqual(self.summary(self.evidence(0))['cuda_over_mojo_latency_ratios'],[0.5])
        self.assertIsNone(self.summary(self.evidence(0,'control'))['cuda_over_mojo_latency_ratios'])

    def test_workload_or_assignment_mismatch_cannot_produce_speed_evidence(self):
        for key,value in [('tokens',32),('decode_ns',0),('mojo_arm',0)]:
            data=self.evidence();data['trials'][3][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.summary(data)
        for arm in [2,None,True]:
            with self.subTest(arm=arm),self.assertRaises(ValueError):self.summary(self.evidence(arm))

    def test_partial_metadata_cannot_reverse_a_context_zero_result(self):
        data=self.evidence(0)
        del data['mojo_arm']
        with self.assertRaises(ValueError):self.summary(data)
        legacy=self.evidence(1)
        del legacy['mojo_arm']
        for row in legacy['trials']:del row['mojo_arm']
        self.assertEqual(self.summary(legacy)['cuda_over_mojo_latency_ratios'],[2])
