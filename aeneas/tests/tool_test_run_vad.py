#!/usr/bin/env python
# coding=utf-8

import os
import unittest

from aeneas.tools.run_vad import RunVADCLI
import aeneas.globalfunctions as gf

class TestRunVADCLI(unittest.TestCase):

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
        exit_code = RunVADCLI(use_sys=False).run(arguments=params)
        gf.delete_directory(output_path)
        self.assertEqual(exit_code, expected_exit_code)

    def test_help(self):
        self.execute([], 2)
        self.execute([("", "-h")], 2)
        self.execute([("", "--help")], 2)
        self.execute([("", "--version")], 2)

    def test_run_both(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "both"),
            ("out", "both.txt")
        ], 0)

    def test_run_speech(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "speech"),
            ("out", "speech.txt")
        ], 0)

    def test_run_nonspeech(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "nonspeech"),
            ("out", "nonspeech.txt")
        ], 0)

    def test_run_cannot_read(self):
        self.execute([
            ("", "/foo/bar/baz.wav"),
            ("", "both"),
            ("out", "both.txt")
        ], 1)

    def test_run_cannot_write(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "both"),
            ("", "/foo/bar/baz.txt")
        ], 1)

    def test_run_both_missing_1(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "both")
        ], 2)

    def test_run_both_missing_2(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("out", "both.txt")
        ], 2)

    def test_run_both_missing_3(self):
        self.execute([
            ("", "both"),
            ("out", "both.txt")
        ], 2)

    def test_run_bad(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "foo"),
            ("out", "both.txt")
        ], 2)



if __name__ == '__main__':
    unittest.main()



