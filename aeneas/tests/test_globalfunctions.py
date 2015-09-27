#!/usr/bin/env python
# coding=utf-8

import unittest

import aeneas.globalfunctions as gf

class TestGlobalFunctions(unittest.TestCase):

    def test_safe_float(self):
        tests = [
            ["3.14", 1.23, 3.14],
            [" 3.14", 1.23, 3.14],
            [" 3.14 ", 1.23, 3.14],
            ["3.14f", 1.23, 1.23],
            ["0x3.14", 1.23, 1.23],
            ["", 1.23, 1.23],
            ["foo", 1.23, 1.23],
            [None, 1.23, 1.23],
        ]
        for test in tests:
            self.assertEqual(gf.safe_float(test[0], test[1]), test[2])

    def test_safe_int(self):
        tests = [
            ["3.14", 1, 3],
            ["3.14 ", 1, 3],
            [" 3.14", 1, 3],
            [" 3.14 ", 1, 3],
            ["3.14f", 1, 1],
            ["0x3.14", 1, 1],
            ["3", 1, 3],
            ["3 ", 1, 3],
            [" 3", 1, 3],
            [" 3 ", 1, 3],
            ["3f", 1, 1],
            ["0x3", 1, 1],
            ["", 1, 1],
            ["foo", 1, 1],
            [None, 1, 1],
        ]
        for test in tests:
            self.assertEqual(gf.safe_int(test[0], test[1]), test[2])

    def test_file_extension(self):
        tests = [
            [None, None],
            ["", ""],
            ["/", ""],
            ["/foo", ""],
            ["/foo.", ""],
            ["/.foo", ""],
            ["/foo.bar", "bar"],
            ["/foo/bar/foo.baz", "baz"],
            ["/foo/bar/baz", ""],
            ["/foo/bar/.baz", ""],
            ["foo", ""],
            ["foo.", ""],
            [".foo", ""],
            ["foo.bar", "bar"],
            ["foo/bar/foo.baz", "baz"],
            ["foo/bar/baz", ""],
            ["foo/bar/.baz", ""],
        ]
        for test in tests:
            self.assertEqual(gf.file_extension(test[0]), test[1])

    def test_file_name_without_extension(self):
        tests = [
            [None, None],
            ["", ""],
            ["/", ""],
            ["/foo", "foo"],
            ["/foo.", "foo"],
            ["/.foo", ".foo"],
            ["/foo.bar", "foo"],
            ["/foo/bar/foo.baz", "foo"],
            ["/foo/bar/baz", "baz"],
            ["/foo/bar/.baz", ".baz"],
            ["foo", "foo"],
            ["foo.", "foo"],
            [".foo", ".foo"],
            ["foo.bar", "foo"],
            ["foo/bar/foo.baz", "foo"],
            ["foo/bar/baz", "baz"],
            ["foo/bar/.baz", ".baz"],
        ]
        for test in tests:
            self.assertEqual(gf.file_name_without_extension(test[0]), test[1])

    def test_norm_join(self):
        tests = [
            [None, None, "."],
            [None, "", "."],
            [None, "/foo", "/foo"],
            [None, "/foo.bar", "/foo.bar"],
            [None, "/foo/../bar", "/bar"],
            [None, "/foo/./bar", "/foo/bar"],
            [None, "/foo/bar/baz", "/foo/bar/baz"],
            [None, "/foo/bar/../../baz", "/baz"],
            [None, "/foo/bar/./baz", "/foo/bar/baz"],
            ["", None, "."],
            ["/foo", None, "/foo"],
            ["/foo.bar", None, "/foo.bar"],
            ["/foo/../bar", None, "/bar"],
            ["/foo/./bar", None, "/foo/bar"],
            ["/foo/bar/baz", None, "/foo/bar/baz"],
            ["/foo/bar/../../baz", None, "/baz"],
            ["/foo/bar/./baz", None, "/foo/bar/baz"],
            ["", "", "."],
            ["/", "", "/"],
            ["", "/", "/"],
            ["/", "/", "/"],
            ["/foo", "bar", "/foo/bar"],
            ["/foo", "bar/foo.baz", "/foo/bar/foo.baz"],
            ["/foo", "bar/../foo.baz", "/foo/foo.baz"],
            ["/foo", "bar/../../foo.baz", "/foo.baz"],
            ["/foo", "bar.baz", "/foo/bar.baz"],
            ["/foo/../", "bar.baz", "/bar.baz"],
            ["/foo/", "../bar.baz", "/bar.baz"],
            ["/foo/./", "bar.baz", "/foo/bar.baz"],
            ["/foo/", "./bar.baz", "/foo/bar.baz"],
            ["foo", "bar", "foo/bar"],
            ["foo", "bar/foo.baz", "foo/bar/foo.baz"],
            ["foo", "bar/../foo.baz", "foo/foo.baz"],
            ["foo", "bar/../../foo.baz", "foo.baz"],
            ["foo", "bar.baz", "foo/bar.baz"],
            ["foo/../", "bar.baz", "bar.baz"],
            ["foo/", "../bar.baz", "bar.baz"],
            ["foo/./", "bar.baz", "foo/bar.baz"],
            ["foo/", "./bar.baz", "foo/bar.baz"],
        ]
        for test in tests:
            self.assertEqual(gf.norm_join(test[0], test[1]), test[2])

    def test_time_from_ttml(self):
        tests = [
            [None, 0],
            ["", 0],
            ["s", 0],
            ["0s", 0],
            ["000s", 0],
            ["1s", 1],
            ["001s", 1],
            ["1s", 1],
            ["001.234s", 1.234],
        ]
        for test in tests:
            self.assertEqual(gf.time_from_ttml(test[0]), test[1])

    def test_time_to_ttml(self):
        tests = [
            [None, "0.000s"],
            [0, "0.000s"],
            [1, "1.000s"],
            [1.234, "1.234s"],
        ]
        for test in tests:
            self.assertEqual(gf.time_to_ttml(test[0]), test[1])

    def test_time_from_ssmmm(self):
        tests = [
            [None, 0],
            ["", 0],
            ["0", 0],
            ["000", 0],
            ["1", 1],
            ["001", 1],
            ["1.234", 1.234],
            ["001.234", 1.234],
        ]
        for test in tests:
            self.assertEqual(gf.time_from_ssmmm(test[0]), test[1])

    def test_time_to_ssmm(self):
        tests = [
            [None, "0.000"],
            [0, "0.000"],
            [1, "1.000"],
            [1.234, "1.234"],
        ]
        for test in tests:
            self.assertEqual(gf.time_to_ssmmm(test[0]), test[1])

    def test_time_from_hhmmssmmm(self):
        tests = [
            [None, 0.000],
            ["", 0.000],
            ["23:45.678", 0.000], # no 2 ":"
            ["3:45.678", 0.000], # no 2 ":"
            ["45.678", 0.000], # no 2 ":"
            ["5.678", 0.000], # no 2 ":"
            ["5", 0.000], # no 2 ":"
            ["00:00:01", 0.000], # no "."
            ["1:23:45.678", 5025.678], # tolerate this (?)
            ["1:2:45.678", 3765.678], # tolerate this (?)
            ["1:23:4.678", 4984.678], # tolerate this (?)
            ["1:23:4.", 4984.000], # tolerate this (?)
            ["00:00:00.000", 0.000],
            ["00:00:12.000", 12.000],
            ["00:00:12.345", 12.345],
            ["00:01:00.000", 60],
            ["00:01:23.000", 83.000],
            ["00:01:23.456", 83.456],
            ["01:00:00.000", 3600.000],
            ["01:00:12.000", 3612.000],
            ["01:00:12.345", 3612.345],
            ["01:23:00.000", 4980.000],
            ["01:23:45.000", 5025.000],
            ["01:23:45.678", 5025.678],
        ]
        for test in tests:
            self.assertEqual(gf.time_from_hhmmssmmm(test[0]), test[1])

    def test_time_to_hhmmssmmm(self):
        tests = [
            [None, "00:00:00.000"],
            [0.000, "00:00:00.000"],
            [12.000, "00:00:12.000"],
            [12.345, "00:00:12.345"],
            [60, "00:01:00.000"],
            [83.000, "00:01:23.000"],
            [83.456, "00:01:23.456"],
            [3600.000, "01:00:00.000"],
            [3612.000, "01:00:12.000"],
            [3612.340, "01:00:12.340"], # numerical issues
            [4980.000, "01:23:00.000"],
            [5025.000, "01:23:45.000"],
            [5025.670, "01:23:45.670"], # numerical issues
        ]
        for test in tests:
            self.assertEqual(gf.time_to_hhmmssmmm(test[0]), test[1])

    def test_time_to_srt(self):
        tests = [
            [None, "00:00:00,000"],
            [0.000, "00:00:00,000"],
            [12.000, "00:00:12,000"],
            [12.345, "00:00:12,345"],
            [60, "00:01:00,000"],
            [83.000, "00:01:23,000"],
            [83.456, "00:01:23,456"],
            [3600.000, "01:00:00,000"],
            [3612.000, "01:00:12,000"],
            [3612.340, "01:00:12,340"], # numerical issues
            [4980.000, "01:23:00,000"],
            [5025.000, "01:23:45,000"],
            [5025.670, "01:23:45,670"], # numerical issues
        ]
        for test in tests:
            self.assertEqual(gf.time_to_srt(test[0]), test[1])

    def test_split_url(self):
        tests = [
            [None, [None, None]],
            ["", ["", None]],
            ["foo", ["foo", None]],
            ["foo.html", ["foo.html", None]],
            ["foo.html#", ["foo.html", ""]],
            ["foo.html#id", ["foo.html", "id"]],
            ["foo.html#id#bad", ["foo.html", "id"]],
        ]
        for test in tests:
            self.assertEqual(gf.split_url(test[0]), test[1])

if __name__ == '__main__':
    unittest.main()



