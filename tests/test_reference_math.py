import math
import unittest
from decimal import Decimal, localcontext
from tests.reference_math import (BASE, EPSILON, HEAD_DIM, HEADS, fixture,
                              position_axis, reference_head)


def independent_trig(angle):
    with localcontext() as context:
        context.prec = 70
        pi = Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164')
        x = angle % (2*pi)
        if x > pi:
            x -= 2*pi
        sine = sine_term = x
        cosine = cosine_term = Decimal(1)
        for k in range(1, 160):
            sine_term *= -x*x / Decimal((2*k)*(2*k+1))
            cosine_term *= -x*x / Decimal((2*k-1)*(2*k))
            sine += sine_term
            cosine += cosine_term
            if max(abs(sine_term), abs(cosine_term)) < Decimal('1e-65'):
                break
        return float(cosine), float(sine)


class ReferenceContractTests(unittest.TestCase):
    def sparse_expected(self, pair, positions, angle):
        q = [0.0]*HEAD_DIM
        q[pair] = 1.0
        actual = reference_head(q, [1.0]*HEAD_DIM, positions)
        c, s = independent_trig(angle)
        amplitude = 1 / math.sqrt(1/HEAD_DIM + EPSILON)
        expected = [0.0]*HEAD_DIM
        expected[pair], expected[pair+64] = amplitude*c, amplitude*s
        for i, value in enumerate(expected):
            self.assertAlmostEqual(actual[i], value, delta=2e-10)

    def test_known_axis_choices_include_fallback(self):
        wanted = {0:0, 1:1, 2:2, 23:2, 24:0, 43:1, 44:2, 59:2,
                  60:0, 61:3, 62:3, 63:0}
        for pair, axis in wanted.items():
            self.assertEqual(position_axis(pair), axis)

    def test_zero_input_remains_zero(self):
        self.assertEqual(reference_head([0.0]*HEAD_DIM, [1.0]*HEAD_DIM,
                                        [3,5,11,17]), [0.0]*HEAD_DIM)

    def test_actual_model_shape_and_finite_oracle(self):
        f = fixture()
        self.assertEqual(len(f['q']), HEADS)
        self.assertTrue(all(len(row)==HEAD_DIM for row in f['expected_float64']))
        self.assertTrue(all(math.isfinite(x) for row in f['expected_float64'] for x in row))

    def test_fixture_discriminates_global_normalization_and_gamma_order(self):
        f = fixture()
        energies = {sum(x*x for x in row) for row in f['q']}
        self.assertEqual(len(energies), HEADS)
        self.assertTrue(all(f['gamma'][i] != f['gamma'][i+64] for i in range(64)))

    def test_multi_axis_changes_result(self):
        self.assertNotEqual(fixture((3,5,11,17))['expected_float64'],
                            fixture((3,3,3,3))['expected_float64'])

    def test_split_half_pairing_and_rotation_sign(self):
        self.sparse_expected(0, [1,0,0,0], Decimal(1))

    def test_position_32768_frequency_one(self):
        self.sparse_expected(0, [32768]*4, Decimal(32768))

    def test_position_32768_half_power_frequency(self):
        with localcontext() as context:
            context.prec = 70
            angle = Decimal(32768) / Decimal(BASE).sqrt()
        self.sparse_expected(32, [32768]*4, angle)

    def test_fourth_axis_fallback_rotates_only_fallback_pairs(self):
        with localcontext() as context:
            context.prec = 70
            for pair in (60,61,62,63):
                angle = (Decimal(32768) *
                         (-Decimal(pair)/64 * Decimal(BASE).ln()).exp()
                         if pair in (61,62) else Decimal(0))
                self.sparse_expected(pair, [0,0,0,32768], angle)

    def test_distinct_axis_selection_numerically(self):
        with localcontext() as context:
            context.prec = 70
            for pair, position in ((0,3), (1,5), (2,11)):
                angle = Decimal(position) * (-Decimal(pair)/64 * Decimal(BASE).ln()).exp()
                self.sparse_expected(pair, [3,5,11,17], angle)

    def test_rms_is_per_head_and_gamma_is_shared_128(self):
        gamma = [1+i/128 for i in range(HEAD_DIM)]
        for head in range(HEADS):
            magnitude = head+1
            actual = reference_head([float(magnitude)]*HEAD_DIM, gamma, [0]*4)
            factor = magnitude / math.sqrt(magnitude*magnitude + EPSILON)
            for i in range(HEAD_DIM):
                self.assertAlmostEqual(actual[i], factor*gamma[i], delta=1e-14)
        with self.assertRaises(ValueError):
            reference_head([1.0]*HEAD_DIM, [1.0]*(HEAD_DIM*HEADS), [0]*4)

    def test_gamma_multiplies_before_rotation(self):
        q, gamma = [0.0]*HEAD_DIM, [1.0]*HEAD_DIM
        q[0] = q[64] = 1.0
        gamma[0], gamma[64] = 2.0, 3.0
        actual = reference_head(q, gamma, [1,0,0,0])
        c, s = independent_trig(Decimal(1))
        denominator = math.sqrt(2/HEAD_DIM + EPSILON)
        self.assertAlmostEqual(actual[0], (2*c-3*s)/denominator, delta=1e-13)
        self.assertAlmostEqual(actual[64], (2*s+3*c)/denominator, delta=1e-13)


if __name__ == '__main__':
    unittest.main()
