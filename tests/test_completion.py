import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from check_completion import check_completion

class CompletionTests(unittest.TestCase):
    def test_numeric_and_string_amounts(self):
        for amount in ('127.5','"127.50"'):
            self.assertTrue(check_completion('{"Business":"Oak Street Bakery","Total":'+amount+'} [end of text]')['valid'])

    def test_trailing_generation_fails(self):
        text='{"Business":"Oak Street Bakery","Total":"127.50"}\n\n{"Business":'
        self.assertFalse(check_completion(text)['valid'])

    def test_wrong_or_missing_fields_fail(self):
        for text in ('{}','[]','{"Business":"Wrong","Total":127.5}',
                     '{"Business":"Oak Street Bakery","Total":true}',
                     '{"Business":"Oak Street Bakery","Total":128}',
                     '{"Business":"Oak Street Bakery","Total":"NaN"}'):
            self.assertFalse(check_completion(text)['valid'])
