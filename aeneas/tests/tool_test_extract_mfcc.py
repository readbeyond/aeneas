#!/usr/bin/env python
# coding=utf-8

import os
import unittest

from aeneas.tools.extract_mfcc import ExtractMFCCCLI
import aeneas.globalfunctions as gf

class TestExtractMFCCCLI(unittest.TestCase):

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
        exit_code = ExtractMFCCCLI(use_sys=False).run(arguments=params)
        gf.delete_directory(output_path)
        self.assertEqual(exit_code, expected_exit_code)

    def test_help(self):
        self.execute([], 2)
        self.execute([("", "-h")], 2)
        self.execute([("", "--help")], 2)
        self.execute([("", "--version")], 2)

    def test_extract(self):
        self.execute([
            ("in", "../tools/res/audio.wav"),
            ("out", "audio.wav.mfcc.txt")
        ], 0)

    def test_extract_mp3(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("out", "audio.mp3.mfcc.txt")
        ], 0)

    def test_extract_pure(self):
        self.execute([
            ("in", "../tools/res/audio.wav"),
            ("out", "audio.wav.mfcc.txt"),
            ("", "--pure")
        ], 0)

    def test_extract_missing_1(self):
        self.execute([
            ("in", "../tools/res/audio.wav")
        ], 2)

    def test_extract_missing_2(self):
        self.execute([
            ("out", "audio.wav.mfcc.txt")
        ], 2)

    def test_extract_cannot_read(self):
        self.execute([
            ("", "/foo/bar/baz.wav"),
            ("out", "audio.wav.mfcc.txt"),
        ], 1)

    def test_extract_cannot_write(self):
        self.execute([
            ("in", "../tools/res/audio.wav"),
            ("", "/foo/bar/baz.wav")
        ], 1)



if __name__ == '__main__':
    unittest.main()



