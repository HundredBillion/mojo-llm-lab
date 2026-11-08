import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from verify_gate_evidence import ROOT, report, verify_run

class GateEvidenceTests(unittest.TestCase):
    def test_retained_speed_evidence(self):
        data=report()
        self.assertEqual(data['paired_rounds'],24)
        self.assertEqual(data['rounds_favoring_mojo'],24)

    def test_reject_corrupted_qualification(self):
        cases=[('correct',False),('red_zones','broken'),('source_rows',64),
               ('mojo_kernel_nodes',1),('same_graph_control',True)]
        for key,value in cases:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)
                for name in ('summary','control','comparison'):
                    shutil.copyfile(ROOT/'results/a2000/q4-pr-proof'/(name+'.json'),out/(name+'.json'))
                data=json.loads((out/'comparison.json').read_text());data[key]=value
                (out/'comparison.json').write_text(json.dumps(data))
                with self.assertRaises(ValueError): verify_run(out)
