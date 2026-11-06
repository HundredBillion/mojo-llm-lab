import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_launch_observer import validation_result

class ValidationEvidenceTests(unittest.TestCase):
    def log(self):
        return ('mojo_validation passed_call=1 rows=9728 max_abs=0\n'
                'mojo_validation passed_call=2 rows=9728 max_abs=0.00006\n'
                'mojo_validation eligible_calls=4 validated_calls=2 outputs=CUDA\n')

    def test_complete(self):
        result=validation_result(self.log(),2)
        self.assertEqual(result['validated_elements'],19456)
        self.assertEqual(result['eligible_calls'],4)
        self.assertEqual(result['max_abs'],0.00006)

    def test_missing_or_duplicate_call(self):
        for text in (self.log().replace('passed_call=2','passed_call=1'),
                     self.log().splitlines()[0],self.log()+self.log()):
            with self.assertRaises(RuntimeError): validation_result(text,2)

    def test_nonfinite_or_negative(self):
        for bad in ('nan','inf','-1'):
            with self.assertRaises(RuntimeError):
                validation_result(self.log().replace('0.00006',bad),2)

    def test_missing_or_inconsistent_final_summary(self):
        for text in (self.log().replace('validated_calls=2','validated_calls=1'),
                     self.log().replace('eligible_calls=4','eligible_calls=1'),
                     self.log().replace('outputs=CUDA','outputs=Mojo')):
            with self.assertRaises(RuntimeError): validation_result(text,2)

    def test_wrong_rows(self):
        with self.assertRaises(RuntimeError):
            validation_result(self.log().replace('rows=9728','rows=64'),2)
