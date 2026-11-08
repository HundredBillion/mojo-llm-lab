from pathlib import Path
import subprocess
import tempfile
import unittest

class LogitValidationTests(unittest.TestCase):
    def test_numerical_and_pairing_contract(self):
        source=Path(__file__).with_name('logit_validation.cpp')
        with tempfile.TemporaryDirectory() as directory:
            binary=Path(directory)/'check'
            subprocess.run(['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror',str(source),'-o',str(binary)],check=True,capture_output=True)
            subprocess.run([str(binary)],check=True,capture_output=True)
