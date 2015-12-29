#!/usr/bin/env python
# coding=utf-8

import os
import unittest

from aeneas.tools.convert_syncmap import ConvertSyncMapCLI
import aeneas.globalfunctions as gf

class TestConvertSyncMapCLI(unittest.TestCase):

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
        exit_code = ConvertSyncMapCLI(use_sys=False).run(arguments=params)
        gf.delete_directory(output_path)
        self.assertEqual(exit_code, expected_exit_code)

    def test_help(self):
        self.execute([], 2)
        self.execute([("", "-h")], 2)
        self.execute([("", "--help")], 2)
        self.execute([("", "--version")], 2)

    def test_convert(self):
        self.execute([
            ("in", "../tools/res/sonnet.json"),
            ("out", "syncmap.srt")
        ], 0)

    def test_convert_cannot_read(self):
        self.execute([
            ("", "/foo/bar/baz.json"),
            ("out", "syncmap.srt")
        ], 1)

    def test_convert_cannot_write(self):
        self.execute([
            ("in", "../tools/res/sonnet.json"),
            ("", "/foo/bar/baz.srt")
        ], 1)

    def test_convert_bad_format_1(self):
        self.execute([
            ("in", "../tools/res/sonnet.zzz"),
            ("out", "syncmap.txt")
        ], 1)

    def test_convert_bad_format_2(self):
        self.execute([
            ("in", "../tools/res/sonnet.json"),
            ("out", "syncmap.zzz")
        ], 1)

    def test_convert_output_format(self):
        self.execute([
            ("in", "../tools/res/sonnet.json"),
            ("out", "syncmap.dat"),
            ("", "--output-format=txt")
        ], 0)

    def test_convert_output_format_bad(self):
        self.execute([
            ("in", "../tools/res/sonnet.json"),
            ("out", "syncmap.dat"),
            ("", "--output-format=foo")
        ], 1)

    def test_convert_input_format(self):
        self.execute([
            ("in", "../tools/res/sonnet.zzz"),
            ("out", "syncmap.txt"),
            ("", "--input-format=csv")
        ], 0)

    def test_convert_input_format_bad(self):
        self.execute([
            ("in", "../tools/res/sonnet.zzz"),
            ("out", "syncmap.txt"),
            ("", "--input-format=foo")
        ], 1)

    def test_convert_language(self):
        self.execute([
            ("in", "../tools/res/sonnet.csv"),
            ("out", "syncmap.json"),
            ("", "--language=en")
        ], 0)

    def test_convert_smil(self):
        self.execute([
            ("in", "../tools/res/sonnet.json"),
            ("out", "syncmap.smil"),
            ("", "--audio-ref=audio/sonnet001.mp3"),
            ("", "--page-ref=text/sonnet001.xhtml")
        ], 0)

    def test_convert_smil_missing_1(self):
        self.execute([
            ("in", "../tools/res/sonnet.json"),
            ("out", "syncmap.smil"),
            ("", "--audio-ref=audio/sonnet001.mp3"),
        ], 1)

    def test_convert_smil_missing_2(self):
        self.execute([
            ("in", "../tools/res/sonnet.json"),
            ("out", "syncmap.smil"),
            ("", "--page-ref=text/sonnet001.xhtml")
        ], 1)

    def test_convert_html(self):
        self.execute([
            ("in", "../tools/res/sonnet.json"),
            ("out", "sonnet.html"),
            ("in", "../tools/res/audio.mp3"),
            ("", "--output-html"),
        ], 0)

    def test_convert_html_missing(self):
        self.execute([
            ("in", "../tools/res/sonnet.json"),
            ("out", "sonnet.html"),
            ("", "--output-html"),
        ], 1)



if __name__ == '__main__':
    unittest.main()



