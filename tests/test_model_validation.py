import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_launch_observer import validation_result, replacement_result, parse_arguments

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

    def test_feedback_must_cover_every_eligible_call(self):
        text=self.log().replace('outputs=CUDA','outputs=Mojo')
        with self.assertRaises(RuntimeError): validation_result(text,2,feedback=True)
        result=validation_result(text.replace('eligible_calls=4','eligible_calls=2'),2,feedback=True)
        self.assertTrue(result['feedback'])
        with self.assertRaises(RuntimeError): validation_result(self.log(),2,feedback=True)

    def test_all_calls(self):
        with self.assertRaises(RuntimeError): validation_result(self.log(),0)
        self.assertEqual(validation_result(self.log().replace('eligible_calls=4','eligible_calls=2'),0)['validated_calls'],2)

    def test_replacement_capture_evidence(self):
        self.assertEqual(replacement_result('mojo_replacement calls=72 captured_calls=36',True)['replacement_calls'],72)
        self.assertEqual(replacement_result('mojo_replacement calls=72 captured_calls=0',False)['replacement_calls'],72)
        for text,graphs in [('',True),('mojo_replacement calls=72 captured_calls=0',True),
                            ('mojo_replacement calls=0 captured_calls=0',False),
                            ('mojo_replacement calls=2 captured_calls=3',True),
                            ('mojo_replacement calls=72 captured_calls=36',False)]:
            with self.assertRaises(RuntimeError): replacement_result(text,graphs)

    def test_feedback_defaults_to_all_calls(self):
        common=['--model','model.gguf','--output','results/local-test','--validate-ptx','candidate.ptx']
        self.assertEqual(parse_arguments(common).validation_calls,36)
        self.assertEqual(parse_arguments(common+['--feedback']).validation_calls,0)
        self.assertEqual(parse_arguments(common+['--feedback','--validation-calls','612']).validation_calls,612)

    def test_general_baseline_cannot_enable_replacement(self):
        from contextlib import redirect_stderr
        import io
        common=['--model','model.gguf','--output','results/local-test','--baseline-only']
        self.assertTrue(parse_arguments(common).baseline_only)
        for option in ['--replace-ptx','--validate-ptx']:
            with self.assertRaises(SystemExit),redirect_stderr(io.StringIO()):
                parse_arguments(common+[option,'candidate.ptx'])

    def test_2b_specialization_cannot_be_used_as_4b(self):
        from run_launch_observer import validate_model_ptx
        parameters=','.join('.param .u64 .ptr .align 1 p'+str(i) for i in range(4))
        ptx='.version 8.1\n.target sm_86\n.address_size 64\n.visible .entry q4_k_q8_1_gate_order_2b_probe('+parameters+') {}'
        self.assertEqual(validate_model_ptx(ptx,'qwen35-2b'),'q4_k_q8_1_gate_order_2b_probe')
        with self.assertRaises(ValueError): validate_model_ptx(ptx,'qwen3-4b')
        with self.assertRaises(ValueError): validate_model_ptx(ptx.replace('_2b_probe','_probe'),'qwen35-2b')
        for threads in ('64','256'):
            with self.assertRaises(ValueError): validate_model_ptx(ptx.replace('_2b_probe','_2b_'+threads+'_probe'),'qwen35-2b')

    def test_2b_validation_requires_correct_row_count(self):
        text=self.log().replace('rows=9728','rows=6144')
        self.assertEqual(validation_result(text,2,rows=6144)['validated_elements'],12288)
        with self.assertRaises(RuntimeError): validation_result(text,2)
