#!/usr/bin/env python
# coding=utf-8

import os
import unittest

from aeneas.tools.ffprobe_wrapper import FFPROBEWrapperCLI
import aeneas.globalfunctions as gf

class TestFFPROBEWrapperCLI(unittest.TestCase):

    def execute(self, parameters, expected_exit_code):
        output_path = gf.tmp_directory()
        params = ["placeholder"]
        for p_type, p_value in parameters:
            if p_type == "in":
                params.append(gf.get_abs_path(p_value, __file__))
            elif p_type == "out":
                params.append(os.path.join(output_path, p_value))
            else:
                params.append(p_value)
        exit_code = FFPROBEWrapperCLI(use_sys=False).run(arguments=params)
        gf.delete_directory(output_path)
        self.assertEqual(exit_code, expected_exit_code)

    def test_help(self):
        self.execute([], 2)
        self.execute([("", "-h")], 2)
        self.execute([("", "--help")], 2)
        self.execute([("", "--version")], 2)

    def test_probe(self):
        self.execute([
            ("in", "../tools/res/audio.wav")
        ], 0)

    def test_probe_mp3(self):
        self.execute([
            ("in", "../tools/res/audio.mp3")
        ], 0)

    def test_probe_cannot_read(self):
        self.execute([
            ("", "/foo/bar/baz.wav")
        ], 1)



if __name__ == '__main__':
    unittest.main()



