#!/usr/bin/env python
# coding=utf-8

import unittest

from aeneas.globalfunctions import safe_float
from aeneas.globalfunctions import safe_int

class TestGlobalFunctions(unittest.TestCase):

    def test_safe_float_01(self):
        default = 1.23
        expected = 3.14
        value = "3.14"
        result = safe_float(value, default)
        self.assertEqual(result, expected)

    def test_safe_float_02(self):
        default = 1.23
        expected = 3.14
        value = " 3.14"
        result = safe_float(value, default)
        self.assertEqual(result, expected)

    def test_safe_float_03(self):
        default = 1.23
        expected = 3.14
        value = "3.14 "
        result = safe_float(value, default)
        self.assertEqual(result, expected)

    def test_safe_float_04(self):
        default = 1.23
        expected = 3.14
        value = " 3.14 "
        result = safe_float(value, default)
        self.assertEqual(result, expected)

    def test_safe_float_05(self):
        default = 1.23
        expected = 1.23
        value = "foo"
        result = safe_float(value, default)
        self.assertEqual(result, expected)

    def test_safe_float_06(self):
        default = 1.23
        expected = 1.23
        value = "3.14f"
        result = safe_float(value, default)
        self.assertEqual(result, expected)

    def test_safe_float_07(self):
        default = 1.23
        expected = 1.23
        value = "0x3.14"
        result = safe_float(value, default)
        self.assertEqual(result, expected)

    def test_safe_int_01(self):
        default = 1
        expected = 3
        value = "3.14"
        result = safe_int(value, default)
        self.assertEqual(result, expected)

    def test_safe_int_02(self):
        default = 1
        expected = 3
        value = " 3.14"
        result = safe_int(value, default)
        self.assertEqual(result, expected)

    def test_safe_int_03(self):
        default = 1
        expected = 3
        value = "3.14 "
        result = safe_int(value, default)
        self.assertEqual(result, expected)

    def test_safe_int_04(self):
        default = 1
        expected = 3
        value = " 3.14 "
        result = safe_int(value, default)
        self.assertEqual(result, expected)

    def test_safe_int_05(self):
        default = 1
        expected = 1
        value = "foo"
        result = safe_int(value, default)
        self.assertEqual(result, expected)

    def test_safe_int_06(self):
        default = 1
        expected = 1
        value = "3f"
        result = safe_int(value, default)
        self.assertEqual(result, expected)

    def test_safe_int_07(self):
        default = 1
        expected = 1
        value = "0x3"
        result = safe_int(value, default)
        self.assertEqual(result, expected)

    def test_safe_int_08(self):
        default = 1
        expected = 3
        value = "3"
        result = safe_int(value, default)
        self.assertEqual(result, expected)

    def test_safe_int_09(self):
        default = 1
        expected = 3
        value = "3.00"
        result = safe_int(value, default)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()



