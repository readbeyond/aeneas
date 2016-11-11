#!/usr/bin/env python
# coding=utf-8

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2016, Alberto Pettarin (www.albertopettarin.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy

from aeneas.exacttiming import Decimal
from aeneas.exacttiming import TimeInterval
from aeneas.exacttiming import TimePoint


class TestExactTiming(unittest.TestCase):

    def check(self, value, expected=None):
        self.assertTrue(isinstance(value, TimePoint))
        if expected is not None:
            self.assertEqual(value, expected)

    def check_numpy(self, value, expected=None):
        self.assertTrue(isinstance(value[0], TimePoint))
        self.assertTrue((value == expected).all())

    def test_create_from_float(self):
        tv1 = TimePoint(1.234)
        self.check(tv1)

    def test_create_from_string(self):
        tv1 = TimePoint("1.234")
        self.check(tv1)

    def test_repr(self):
        tv1 = TimePoint("1.234")
        self.assertEqual(tv1.__repr__(), "TimePoint('1.234')")

    def test_add(self):
        tv1 = TimePoint("1.100")
        tv2 = TimePoint("2.200")
        tv3 = TimePoint("3.300")
        self.check(tv1 + tv2, tv3)
        self.check(tv2 + tv1, tv3)
        d = Decimal("2.200")
        self.check(tv1 + d, tv3)
        self.check(d + tv1, tv3)

    def test_div(self):
        tv1 = TimePoint("1.100")
        tv2 = TimePoint("2.200")
        tv3 = TimePoint("0.500")
        self.check(tv1 / tv2, tv3)
        d = Decimal("2.200")
        self.check(tv1 / d, tv3)

    def test_floordiv(self):
        tv1 = TimePoint("1.100")
        tv2 = TimePoint("2.200")
        tv3 = TimePoint("0.000")
        self.check(tv1 // tv2, tv3)
        d = Decimal("2.200")
        self.check(tv1 // d, tv3)

    def test_mod(self):
        tv1 = TimePoint("1.100")
        tv2 = TimePoint("2.200")
        tv3 = TimePoint("0.000")
        self.check(tv2 % tv1, tv3)
        d = Decimal("1.100")
        self.check(tv2 % d, tv3)

    def test_mul(self):
        tv1 = TimePoint("1.100")
        tv2 = TimePoint("2.200")
        tv3 = TimePoint("2.420")
        self.check(tv1 * tv2, tv3)
        self.check(tv2 * tv1, tv3)
        d = Decimal("2.200")
        self.check(tv1 * d, tv3)

    def test_sub(self):
        tv1 = TimePoint("1.100")
        tv2 = TimePoint("2.200")
        tv3 = TimePoint("-1.100")
        tv4 = TimePoint("1.100")
        self.check(tv1 - tv2, tv3)
        self.check(tv2 - tv1, tv4)
        d = Decimal("2.200")
        self.check(tv1 - d, tv3)

    def test_truediv(self):
        tv1 = TimePoint("1")
        tv2 = TimePoint("2")
        tv3 = TimePoint("0.5")
        self.check(tv1 / tv2, tv3)
        d = Decimal("2")
        self.check(tv1 / d, tv3)

    def test_op_float(self):
        tv1 = TimePoint("1.100")
        tv2 = 2.200
        tv3 = TimePoint("3.300")
        with self.assertRaises(TypeError):
            self.check(tv1 + tv2, tv3)
        with self.assertRaises(TypeError):
            self.check(tv1 / tv2, tv3)
        with self.assertRaises(TypeError):
            self.check(tv1 // tv2, tv3)
        with self.assertRaises(TypeError):
            self.check(tv1 * tv2, tv3)
        with self.assertRaises(TypeError):
            self.check(tv1 - tv2, tv3)

    def test_numpy_int(self):
        tv1 = TimePoint("1.000")
        n1 = numpy.array([0, 1, 2], dtype=int)
        n2 = numpy.array([TimePoint("1.000"), TimePoint("2.000"), TimePoint("3.000")], dtype=TimePoint)
        n3 = numpy.array([TimePoint("0.000"), TimePoint("1.000"), TimePoint("2.000")], dtype=TimePoint)
        self.check_numpy(n1 + tv1, n2)
        self.check_numpy(n1 * tv1, n3)

    def test_numpy_float(self):
        tv1 = TimePoint("1.000")
        n1 = numpy.array([0.0, 1.0, 2.0], dtype=float)
        n2 = numpy.array([TimePoint("1.000"), TimePoint("2.000"), TimePoint("3.000")], dtype=TimePoint)
        n3 = numpy.array([TimePoint("0.000"), TimePoint("1.000"), TimePoint("2.000")], dtype=TimePoint)
        with self.assertRaises(TypeError):
            self.check_numpy(n1 + tv1, n2)
        with self.assertRaises(TypeError):
            self.check_numpy(n1 * tv1, n3)

    def test_numpy_tv(self):
        tv1 = TimePoint("1.000")
        n1 = numpy.array([TimePoint("0.000"), TimePoint("1.000"), TimePoint("2.000")], dtype=TimePoint)
        n2 = numpy.array([TimePoint("1.000"), TimePoint("2.000"), TimePoint("3.000")], dtype=TimePoint)
        n3 = numpy.array([TimePoint("0.000"), TimePoint("1.000"), TimePoint("2.000")], dtype=TimePoint)
        self.check_numpy(n1 + tv1, n2)
        self.check_numpy(n1 * tv1, n3)

    def test_numpy_decimal(self):
        tv1 = TimePoint("1.000")
        n1 = numpy.array([Decimal("0.000"), Decimal("1.000"), Decimal("2.000")], dtype=Decimal)
        n2 = numpy.array([TimePoint("1.000"), TimePoint("2.000"), TimePoint("3.000")], dtype=TimePoint)
        n3 = numpy.array([TimePoint("0.000"), TimePoint("1.000"), TimePoint("2.000")], dtype=TimePoint)
        self.check_numpy(n1 + tv1, n2)
        self.check_numpy(n1 * tv1, n3)

    def test_product_is_integer(self):
        for m, s, e in [
            ("0.001", 16000, True),
            ("0.001", 22050, False),
            ("0.001", 44100, False),
            ("0.005", 16000, True),
            ("0.005", 22050, False),
            ("0.005", 44100, False),
            ("0.010", 16000, True),
            ("0.010", 22050, False),
            ("0.010", 44100, True),
            ("0.020", 16000, True),
            ("0.020", 22050, True),
            ("0.020", 44100, True),
            ("0.040", 16000, True),
            ("0.040", 22050, True),
            ("0.040", 44100, True),
            ("1.000", 16000, True),
            ("1.000", 22050, True),
            ("1.000", 44100, True),
        ]:
            prod = TimePoint(m) * s
            self.assertTrue(isinstance(prod, TimePoint))
            self.assertEqual(int(prod) == prod, e)

    def test_time_interval_bad_type(self):
        params = [
            (None, None),
            (0, 1),
            (TimePoint("0.000"), 1),
            (0, TimePoint("1.000")),
        ]
        for b, e in params:
            with self.assertRaises(TypeError):
                TimeInterval(begin=b, end=e)

    def test_time_interval_bad_value(self):
        params = [
            ("1.000", "0.000"),
            ("-1.000", "0.000"),
            ("0.000", "-1.000"),
            ("-2.000", "-1.000"),
        ]
        for b, e in params:
            with self.assertRaises(ValueError):
                TimeInterval(begin=TimePoint(b), end=TimePoint(e))

    def test_time_interval_constructor(self):
        params = [
            ("0.000", "0.000"),
            ("0.000", "1.000"),
            ("1.000", "1.000"),
            ("1.234", "1.235"),
        ]
        for b, e in params:
            TimeInterval(begin=TimePoint(b), end=TimePoint(e))

    def test_time_interval_length(self):
        params = [
            ("0.000", "0.000", "0.000"),
            ("0.000", "1.000", "1.000"),
            ("1.000", "1.000", "0.000"),
            ("1.234", "1.235", "0.001"),
        ]
        for b, e, l in params:
            ti = TimeInterval(begin=TimePoint(b), end=TimePoint(e))
            self.assertEqual(ti.length, TimePoint(l))

    def test_time_interval_has_zero_length(self):
        params = [
            ("0.000", "0.000", True),
            ("0.000", "1.000", False),
            ("1.000", "1.000", True),
            ("1.234", "1.235", False),
        ]
        for b, e, f in params:
            ti = TimeInterval(begin=TimePoint(b), end=TimePoint(e))
            self.assertEqual(ti.has_zero_length, f)

    def test_time_interval_starts_at(self):
        params = [
            ("1.234", "1.237", "0.000", False),
            ("1.234", "1.237", "1.233", False),
            ("1.234", "1.237", "1.234", True),
            ("1.234", "1.237", "1.235", False),
            ("1.234", "1.237", "1.236", False),
            ("1.234", "1.237", "1.237", False),
            ("1.234", "1.237", "1.238", False),
            ("1.234", "1.237", "2.000", False),
        ]
        for b, e, p, f in params:
            ti = TimeInterval(begin=TimePoint(b), end=TimePoint(e))
            self.assertEqual(ti.starts_at(TimePoint(p)), f)

    def test_time_interval_ends_at(self):
        params = [
            ("1.234", "1.237", "0.000", False),
            ("1.234", "1.237", "1.233", False),
            ("1.234", "1.237", "1.234", False),
            ("1.234", "1.237", "1.235", False),
            ("1.234", "1.237", "1.236", False),
            ("1.234", "1.237", "1.237", True),
            ("1.234", "1.237", "1.238", False),
            ("1.234", "1.237", "2.000", False),
        ]
        for b, e, p, f in params:
            ti = TimeInterval(begin=TimePoint(b), end=TimePoint(e))
            self.assertEqual(ti.ends_at(TimePoint(p)), f)

    def test_time_interval_translate_bad(self):
        params = [
            None,
            1,
            1.234,
            "1.234",
            Decimal("1.234"),
        ]
        ti1 = TimeInterval(begin=TimePoint("0.000"), end=TimePoint("1.000"))
        for p in params:
            with self.assertRaises(TypeError):
                ti1.translate(p)

    def test_time_interval_translate(self):
        params = [
            (("0.000", "0.000"), "-1.000", False, ("0.000", "0.000")),
            (("0.000", "0.000"), "-0.000", False, ("0.000", "0.000")),
            (("0.000", "0.000"), "0.000", False, ("0.000", "0.000")),
            (("0.000", "0.000"), "0.500", False, ("0.500", "0.500")),
            (("1.000", "2.000"), "-2.500", False, ("0.000", "0.000")),
            (("1.000", "2.000"), "-2.000", False, ("0.000", "0.000")),
            (("1.000", "2.000"), "-1.500", False, ("0.000", "0.500")),
            (("1.000", "2.000"), "-1.000", False, ("0.000", "1.000")),
            (("1.000", "2.000"), "-0.000", False, ("1.000", "2.000")),
            (("1.000", "2.000"), "0.000", False, ("1.000", "2.000")),
            (("1.000", "2.000"), "0.500", False, ("1.500", "2.500")),
            (("1.000", "2.000"), "1.000", False, ("2.000", "3.000")),

            (("0.000", "0.000"), "-1.000", True, ("-1.000", "-1.000")),
            (("0.000", "0.000"), "-0.000", True, ("0.000", "0.000")),
            (("0.000", "0.000"), "0.000", True, ("0.000", "0.000")),
            (("0.000", "0.000"), "0.500", True, ("0.500", "0.500")),
            (("1.000", "2.000"), "-2.500", True, ("-1.500", "-0.500")),
            (("1.000", "2.000"), "-2.000", True, ("-1.000", "0.000")),
            (("1.000", "2.000"), "-1.500", True, ("-0.500", "0.500")),
            (("1.000", "2.000"), "-1.000", True, ("0.000", "1.000")),
            (("1.000", "2.000"), "-0.000", True, ("1.000", "2.000")),
            (("1.000", "2.000"), "0.000", True, ("1.000", "2.000")),
            (("1.000", "2.000"), "0.500", True, ("1.500", "2.500")),
            (("1.000", "2.000"), "1.000", True, ("2.000", "3.000")),
        ]
        for ti1, d, a, exp in params:
            ti1 = TimeInterval(begin=TimePoint(ti1[0]), end=TimePoint(ti1[1]))
            d = TimePoint(d)
            ti1.translate(delta=d, allow_negative=a)
            self.assertEqual(ti1.begin, TimePoint(exp[0]))
            self.assertEqual(ti1.end, TimePoint(exp[1]))

    def test_time_interval_contains(self):
        params = [
            ("1.000", "1.000", "0.000", False),
            ("1.000", "1.000", "0.999", False),
            ("1.000", "1.000", "1.000", True),
            ("1.000", "1.000", "1.001", False),
            ("1.000", "1.000", "2.000", False),

            ("1.000", "1.001", "0.000", False),
            ("1.000", "1.001", "0.999", False),
            ("1.000", "1.001", "1.000", True),
            ("1.000", "1.001", "1.001", True),
            ("1.000", "1.001", "1.002", False),
            ("1.000", "1.001", "2.000", False),

            ("1.000", "1.002", "0.000", False),
            ("1.000", "1.002", "0.999", False),
            ("1.000", "1.002", "1.000", True),
            ("1.000", "1.002", "1.001", True),
            ("1.000", "1.002", "1.002", True),
            ("1.000", "1.002", "1.003", False),
            ("1.000", "1.002", "2.000", False),

            ("1.234", "1.237", "0.000", False),
            ("1.234", "1.237", "1.233", False),
            ("1.234", "1.237", "1.234", True),
            ("1.234", "1.237", "1.235", True),
            ("1.234", "1.237", "1.236", True),
            ("1.234", "1.237", "1.237", True),
            ("1.234", "1.237", "1.238", False),
            ("1.234", "1.237", "2.000", False),
        ]
        for b, e, p, f in params:
            ti = TimeInterval(begin=TimePoint(b), end=TimePoint(e))
            self.assertEqual(ti.contains(TimePoint(p)), f)

    def test_time_interval_inner_contains(self):
        params = [
            ("1.000", "1.000", "0.000", False),
            ("1.000", "1.000", "0.999", False),
            ("1.000", "1.000", "1.000", False),
            ("1.000", "1.000", "1.001", False),
            ("1.000", "1.000", "2.000", False),

            ("1.000", "1.001", "0.000", False),
            ("1.000", "1.001", "0.999", False),
            ("1.000", "1.001", "1.000", False),
            ("1.000", "1.001", "1.001", False),
            ("1.000", "1.001", "1.002", False),
            ("1.000", "1.001", "2.000", False),

            ("1.000", "1.002", "0.000", False),
            ("1.000", "1.002", "0.999", False),
            ("1.000", "1.002", "1.000", False),
            ("1.000", "1.002", "1.001", True),
            ("1.000", "1.002", "1.002", False),
            ("1.000", "1.002", "1.003", False),
            ("1.000", "1.002", "2.000", False),

            ("1.234", "1.237", "0.000", False),
            ("1.234", "1.237", "1.233", False),
            ("1.234", "1.237", "1.234", False),
            ("1.234", "1.237", "1.235", True),
            ("1.234", "1.237", "1.236", True),
            ("1.234", "1.237", "1.237", False),
            ("1.234", "1.237", "1.238", False),
            ("1.234", "1.237", "2.000", False),
        ]
        for b, e, p, f in params:
            ti = TimeInterval(begin=TimePoint(b), end=TimePoint(e))
            self.assertEqual(ti.inner_contains(TimePoint(p)), f)

    def test_time_interval_relative_position_of(self):
        params = [
            # TABLE 1
            (("1.000", "1.000"), ("0.000", "0.000"), TimeInterval.RELATIVE_POSITION_PP_L),
            (("1.000", "1.000"), ("1.000", "1.000"), TimeInterval.RELATIVE_POSITION_PP_C),
            (("1.000", "1.000"), ("2.000", "2.000"), TimeInterval.RELATIVE_POSITION_PP_G),
            # TABLE 2
            (("1.000", "1.000"), ("0.000", "0.500"), TimeInterval.RELATIVE_POSITION_PI_LL),
            (("1.000", "1.000"), ("0.500", "1.000"), TimeInterval.RELATIVE_POSITION_PI_LC),
            (("1.000", "1.000"), ("0.500", "1.500"), TimeInterval.RELATIVE_POSITION_PI_LG),
            (("1.000", "1.000"), ("1.000", "1.500"), TimeInterval.RELATIVE_POSITION_PI_CG),
            (("1.000", "1.000"), ("1.500", "2.000"), TimeInterval.RELATIVE_POSITION_PI_GG),
            # TABLE 3
            (("1.000", "2.000"), ("0.000", "0.000"), TimeInterval.RELATIVE_POSITION_IP_L),
            (("1.000", "2.000"), ("1.000", "1.000"), TimeInterval.RELATIVE_POSITION_IP_B),
            (("1.000", "2.000"), ("1.500", "1.500"), TimeInterval.RELATIVE_POSITION_IP_I),
            (("1.000", "2.000"), ("2.000", "2.000"), TimeInterval.RELATIVE_POSITION_IP_E),
            (("1.000", "2.000"), ("2.500", "2.500"), TimeInterval.RELATIVE_POSITION_IP_G),
            # TABLE 4
            (("1.000", "2.000"), ("0.000", "0.500"), TimeInterval.RELATIVE_POSITION_II_LL),
            (("1.000", "2.000"), ("0.000", "1.000"), TimeInterval.RELATIVE_POSITION_II_LB),
            (("1.000", "2.000"), ("0.000", "1.500"), TimeInterval.RELATIVE_POSITION_II_LI),
            (("1.000", "2.000"), ("0.000", "2.000"), TimeInterval.RELATIVE_POSITION_II_LE),
            (("1.000", "2.000"), ("0.000", "2.500"), TimeInterval.RELATIVE_POSITION_II_LG),
            # TABLE 5
            (("1.000", "2.000"), ("1.000", "1.500"), TimeInterval.RELATIVE_POSITION_II_BI),
            (("1.000", "2.000"), ("1.000", "2.000"), TimeInterval.RELATIVE_POSITION_II_BE),
            (("1.000", "2.000"), ("1.000", "2.500"), TimeInterval.RELATIVE_POSITION_II_BG),
            # TABLE 6
            (("1.000", "2.000"), ("1.100", "1.500"), TimeInterval.RELATIVE_POSITION_II_II),
            (("1.000", "2.000"), ("1.100", "2.000"), TimeInterval.RELATIVE_POSITION_II_IE),
            (("1.000", "2.000"), ("1.100", "2.500"), TimeInterval.RELATIVE_POSITION_II_IG),
            # TABLE 7
            (("1.000", "2.000"), ("2.000", "2.500"), TimeInterval.RELATIVE_POSITION_II_EG),
            # TABLE 8
            (("1.000", "2.000"), ("2.500", "3.000"), TimeInterval.RELATIVE_POSITION_II_GG),
        ]
        for ti1, ti2, exp in params:
            ti1 = TimeInterval(begin=TimePoint(ti1[0]), end=TimePoint(ti1[1]))
            ti2 = TimeInterval(begin=TimePoint(ti2[0]), end=TimePoint(ti2[1]))
            self.assertEqual(ti1.relative_position_of(ti2), exp)
            self.assertEqual(ti2.relative_position_wrt(ti1), exp)

    def test_time_interval_intersection(self):
        params = [
            # TABLE 1
            (("1.000", "1.000"), ("0.000", "0.000"), None),
            (("1.000", "1.000"), ("1.000", "1.000"), ("1.000", "1.000")),
            (("1.000", "1.000"), ("2.000", "2.000"), None),
            # TABLE 2
            (("1.000", "1.000"), ("0.000", "0.500"), None),
            (("1.000", "1.000"), ("0.500", "1.000"), ("1.000", "1.000")),
            (("1.000", "1.000"), ("0.500", "1.500"), ("1.000", "1.000")),
            (("1.000", "1.000"), ("1.000", "1.500"), ("1.000", "1.000")),
            (("1.000", "1.000"), ("1.500", "2.000"), None),
            # TABLE 3
            (("1.000", "2.000"), ("0.000", "0.000"), None),
            (("1.000", "2.000"), ("1.000", "1.000"), ("1.000", "1.000")),
            (("1.000", "2.000"), ("1.500", "1.500"), ("1.500", "1.500")),
            (("1.000", "2.000"), ("2.000", "2.000"), ("2.000", "2.000")),
            (("1.000", "2.000"), ("2.500", "2.500"), None),
            # TABLE 4
            (("1.000", "2.000"), ("0.000", "0.500"), None),
            (("1.000", "2.000"), ("0.000", "1.000"), ("1.000", "1.000")),
            (("1.000", "2.000"), ("0.000", "1.500"), ("1.000", "1.500")),
            (("1.000", "2.000"), ("0.000", "2.000"), ("1.000", "2.000")),
            (("1.000", "2.000"), ("0.000", "2.500"), ("1.000", "2.000")),
            # TABLE 5
            (("1.000", "2.000"), ("1.000", "1.500"), ("1.000", "1.500")),
            (("1.000", "2.000"), ("1.000", "2.000"), ("1.000", "2.000")),
            (("1.000", "2.000"), ("1.000", "2.500"), ("1.000", "2.000")),
            # TABLE 6
            (("1.000", "2.000"), ("1.100", "1.500"), ("1.100", "1.500")),
            (("1.000", "2.000"), ("1.100", "2.000"), ("1.100", "2.000")),
            (("1.000", "2.000"), ("1.100", "2.500"), ("1.100", "2.000")),
            # TABLE 7
            (("1.000", "2.000"), ("2.000", "2.500"), ("2.000", "2.000")),
            # TABLE 8
            (("1.000", "2.000"), ("2.500", "3.000"), None),
        ]
        for ti1, ti2, exp in params:
            ti1 = TimeInterval(begin=TimePoint(ti1[0]), end=TimePoint(ti1[1]))
            ti2 = TimeInterval(begin=TimePoint(ti2[0]), end=TimePoint(ti2[1]))
            if exp is not None:
                exp = TimeInterval(begin=TimePoint(exp[0]), end=TimePoint(exp[1]))
            self.assertEqual(ti1.intersection(ti2), exp)
            self.assertEqual(ti2.intersection(ti1), exp)
            self.assertEqual(ti1.overlaps(ti2), exp is not None)
            self.assertEqual(ti2.overlaps(ti1), exp is not None)

    def test_time_interval_adjacent(self):
        params = [
            (("1.000", "1.000"), ("0.000", "2.000"), False),
            (("1.000", "1.000"), ("0.999", "2.000"), False),
            (("1.000", "1.000"), ("1.000", "2.000"), True),
            (("1.000", "1.000"), ("1.001", "2.000"), False),
            (("1.000", "1.000"), ("2.000", "2.000"), False),
            (("0.000", "1.000"), ("0.000", "2.000"), False),
            (("0.000", "1.000"), ("0.999", "2.000"), False),
            (("0.000", "1.000"), ("1.000", "2.000"), True),
            (("0.000", "1.000"), ("1.001", "2.000"), False),
            (("0.000", "1.000"), ("2.000", "2.000"), False),
        ]
        for ti1, ti2, exp in params:
            ti1 = TimeInterval(begin=TimePoint(ti1[0]), end=TimePoint(ti1[1]))
            ti2 = TimeInterval(begin=TimePoint(ti2[0]), end=TimePoint(ti2[1]))
            self.assertEqual(ti1.adjacent_before(ti2), exp)
            self.assertEqual(ti2.adjacent_after(ti1), exp)


if __name__ == '__main__':
    unittest.main()
