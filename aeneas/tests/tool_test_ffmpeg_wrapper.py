#!/usr/bin/env python
# coding=utf-8

import os
import unittest

from aeneas.tools.ffmpeg_wrapper import FFMPEGWrapperCLI
import aeneas.globalfunctions as gf

class TestFFMPEGWrapperCLI(unittest.TestCase):

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
        exit_code = FFMPEGWrapperCLI(use_sys=False).run(arguments=params)
        gf.delete_directory(output_path)
        self.assertEqual(exit_code, expected_exit_code)

    def test_help(self):
        self.execute([], 2)
        self.execute([("", "-h")], 2)
        self.execute([("", "--help")], 2)
        self.execute([("", "--version")], 2)

    def test_convert(self):
        self.execute([
            ("in", "../tools/res/audio.wav"),
            ("out", "audio.wav")
        ], 0)

    def test_convert_mp3(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("out", "audio.wav")
        ], 0)

    def test_convert_missing_1(self):
        self.execute([
            ("in", "../tools/res/audio.wav")
        ], 2)

    def test_convert_missing_2(self):
        self.execute([
            ("out", "audio.wav")
        ], 2)

    def test_convert_cannot_read(self):
        self.execute([
            ("", "/foo/bar/baz.wav"),
            ("out", "audio.wav"),
        ], 1)

    def test_convert_cannot_write(self):
        self.execute([
            ("in", "../tools/res/audio.wav"),
            ("", "/foo/bar/baz.wav")
        ], 1)



if __name__ == '__main__':
    unittest.main()



