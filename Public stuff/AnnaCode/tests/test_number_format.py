"""Run from the AnnaCode folder:  python3 -m unittest discover tests -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "network"))
import number_format as nf  # noqa: E402


class FmtCoeff(unittest.TestCase):
    def test_ordinary_values_keep_two_decimals(self):
        self.assertEqual(nf.fmt_coeff(1.0), "1.00")
        self.assertEqual(nf.fmt_coeff(-0.456), "-0.46")
        self.assertEqual(nf.fmt_coeff(0.1), "0.10")
        self.assertEqual(nf.fmt_coeff(0.456), "0.46")
        self.assertEqual(nf.fmt_coeff(123.456), "123.46")

    def test_a_small_nonzero_weight_never_looks_like_zero(self):
        for x in (0.0174, -0.0174, 0.004, -0.004, 0.00105, 3e-5, -2.5e-7, 0.0996, 0.05):
            shown = nf.fmt_coeff(x)
            self.assertNotEqual(float(shown), 0.0, (x, shown))
            self.assertAlmostEqual(float(shown) / x, 1.0, delta=0.01)

    def test_only_a_true_zero_is_written_zero(self):
        self.assertEqual(nf.fmt_coeff(0.0), "0")
        self.assertEqual(nf.fmt_coeff(-0.0), "0")

    def test_three_significant_digits_for_small_values(self):
        self.assertEqual(nf.fmt_coeff(0.017432), "0.0174")
        self.assertEqual(nf.fmt_coeff(0.001053), "0.00105")

    def test_the_text_can_be_read_back_by_a_number_box(self):
        for x in (0.017432, 5e-6, -0.3, 42.0):
            float(nf.fmt_coeff(x))      # a number input only accepts text float() can parse

    def test_bad_values_do_not_raise(self):
        self.assertEqual(nf.fmt_coeff(None), "0")
        self.assertEqual(nf.fmt_coeff("x"), "0")
        self.assertEqual(nf.fmt_coeff(float("nan")), "0")
        self.assertEqual(nf.fmt_coeff(float("inf")), "inf")


if __name__ == "__main__":
    unittest.main()
