#!/usr/bin/env python
# coding=utf-8

import os
import unittest

from aeneas.tools.espeak_wrapper import ESPEAKWrapperCLI
import aeneas.globalfunctions as gf

class TestConvertSyncMapCLI(unittest.TestCase):

    def execute(self, parameters, expected_exit_code):
        output_path = gf.tmp_directory()
        params = ["placeholder"]
        for p_type, p_value in parameters:
            if p_type == "in":
                params.append(gf.absolute_path(p_value, __file__))
            elif p_type == "out":
                params.append(os.path.join(output_path, p_value))
            else:
                params.append(p_value)
        exit_code = ESPEAKWrapperCLI(use_sys=False).run(arguments=params)
        gf.delete_directory(output_path)
        self.assertEqual(exit_code, expected_exit_code)

    def test_help(self):
        self.execute([], 2)
        self.execute([("", "-h")], 2)
        self.execute([("", "--help")], 2)
        self.execute([("", "--version")], 2)

    def test_synt(self):
        self.execute([
            ("", "From fairest creatures we desire increase"),
            ("", "en"),
            ("out", "sonnet.wav")
        ], 0)

    def test_synt_multiple(self):
        self.execute([
            ("", "From|fairest|creatures|we|desire|increase"),
            ("", "en"),
            ("out", "sonnet.wav"),
            ("", "-m")
        ], 0)

    def test_synt_pure(self):
        self.execute([
            ("", "From fairest creatures we desire increase"),
            ("", "en"),
            ("out", "sonnet.wav"),
            ("", "--pure")
        ], 0)

    def test_synt_missing_1(self):
        self.execute([
            ("", "From fairest creatures we desire increase"),
            ("", "en")
        ], 2)

    def test_synt_missing_2(self):
        self.execute([
            ("", "From fairest creatures we desire increase"),
            ("out", "sonnet.wav")
        ], 2)

    def test_synt_cannot_write(self):
        self.execute([
            ("", "From fairest creatures we desire increase"),
            ("", "en"),
            ("", "/foo/bar/baz.wav")
        ], 1)

    def test_synt_bad_language(self):
        self.execute([
            ("", "From fairest creatures we desire increase"),
            ("", "en-zz"),
            ("out", "sonnet.wav")
        ], 1)

    def test_synt_override_bad_language(self):
        self.execute([
            ("", "From fairest creatures we desire increase"),
            ("", "en-zz"),
            ("out", "sonnet.wav"),
            ("", "--allow-unlisted-language")
        ], 0)



if __name__ == '__main__':
    unittest.main()



