import unittest
from tools.validate_ptx import validate_ptx


def sample(count=5):
    parameters = ',\n'.join('.param .u64 .ptr .align 1 p'+str(i) for i in range(count))
    return '.version 8.1\n.target sm_86\n.address_size 64\n.visible .entry probe(\n'+parameters+'\n) {}'


class PtxValidationTests(unittest.TestCase):
    def test_expected_five_pointer_entry(self):
        self.assertEqual(validate_ptx(sample()), 'probe')

    def test_three_pointer_abi_requires_explicit_selection(self):
        self.assertEqual(validate_ptx(sample(3), expected_parameters=3), 'probe')
        self.assertEqual(validate_ptx(sample(4), expected_parameters=4), 'probe')
        for text in (sample(5), sample(3).replace('p2','p0')):
            with self.assertRaises(ValueError):
                validate_ptx(text, expected_parameters=3)

    def test_wrong_parameter_count_rejected(self):
        for count in (0, 3, 4, 6):
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_ptx(sample(count))

    def test_malformed_abi_rejected(self):
        for text in (sample().replace('.u64','.u32',1),
                     sample().replace('.ptr .align 1','',1),
                     sample().replace('.align 1','.align 4',1),
                     sample().replace('p4','p0'),
                     sample().replace('.param .u64 .ptr .align 1 p4','')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_ptx(text)

    def test_wrong_architecture_version_and_width_rejected(self):
        for text in (sample().replace('sm_86','sm_90'), sample().replace('8.1','8.7'),
                     sample().replace('size 64','size 32'),
                     sample()+'\n.target sm_90', sample()+'\n.version 8.1'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_ptx(text)

    def test_missing_multiple_entries_and_external_dependencies_rejected(self):
        for text in ('', sample()+sample(), sample()+'\n.extern .func external();'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_ptx(text)


if __name__ == '__main__':
    unittest.main()
