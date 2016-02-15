#!/usr/bin/env python
# coding=utf-8

import os
import unittest

from aeneas.tools.synthesize_text import SynthesizeTextCLI
import aeneas.globalfunctions as gf

class TestSynthesizeTextCLI(unittest.TestCase):

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
        exit_code = SynthesizeTextCLI(use_sys=False).run(arguments=params)
        gf.delete_directory(output_path)
        self.assertEqual(exit_code, expected_exit_code)

    def test_help(self):
        self.execute([], 2)
        self.execute([("", "-h")], 2)
        self.execute([("", "--help")], 2)
        self.execute([("", "--version")], 2)

    def test_synt_list(self):
        self.execute([
            ("", "list"),
            ("", "From|fairest|creatures|we|desire|increase"),
            ("", "en"),
            ("out", "synthesized.wav")
        ], 0)

    def test_synt_parsed(self):
        self.execute([
            ("", "parsed"),
            ("in", "../tools/res/parsed.txt"),
            ("", "en"),
            ("out", "synthesized.wav")
        ], 0)

    def test_synt_plain(self):
        self.execute([
            ("", "plain"),
            ("in", "../tools/res/plain.txt"),
            ("", "en"),
            ("out", "synthesized.wav")
        ], 0)

    def test_synt_subtitles(self):
        self.execute([
            ("", "subtitles"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "en"),
            ("out", "synthesized.wav")
        ], 0)

    def test_synt_unparsed_id_regex(self):
        self.execute([
            ("", "unparsed"),
            ("in", "../tools/res/unparsed.xhtml"),
            ("", "en"),
            ("out", "synthesized.wav"),
            ("", "--id-regex=f[0-9]*")
        ], 0)

    def test_synt_unparsed_class_regex(self):
        self.execute([
            ("", "unparsed"),
            ("in", "../tools/res/unparsed.xhtml"),
            ("", "en"),
            ("out", "synthesized.wav"),
            ("", "--class-regex=ra"),
            ("", "--sort=unsorted"),
        ], 0)

    def test_synt_unparsed_sort_numeric(self):
        self.execute([
            ("", "unparsed"),
            ("in", "../tools/res/unparsed.xhtml"),
            ("", "en"),
            ("out", "synthesized.wav"),
            ("", "--id-regex=f[0-9]*"),
            ("", "--sort=numeric")
        ], 0)

    def test_synt_unparsed_sort_lexicographic(self):
        self.execute([
            ("", "unparsed"),
            ("in", "../tools/res/unparsed.xhtml"),
            ("", "en"),
            ("out", "synthesized.wav"),
            ("", "--id-regex=f[0-9]*"),
            ("", "--sort=lexicographic")
        ], 0)

    def test_synt_missing_1(self):
        self.execute([
            ("", "list"),
            ("", "en"),
            ("in", "../tools/res/audio.mp3")
        ], 2)

    def test_synt_missing_2(self):
        self.execute([
            ("", "From|fairest|creatures|we|desire|increase"),
            ("", "en"),
            ("out", "synthesized.wav")
        ], 2)

    def test_synt_missing_3(self):
        self.execute([
            ("", "list"),
            ("", "From|fairest|creatures|we|desire|increase"),
            ("out", "synthesized.wav")
        ], 2)

    def test_synt_missing_4(self):
        self.execute([
            ("", "list"),
            ("", "From|fairest|creatures|we|desire|increase"),
            ("", "en")
        ], 2)

    def test_synt_cannot_read(self):
        self.execute([
            ("", "plain"),
            ("", "/foo/bar/baz.wav"),
            ("", "en"),
            ("out", "synthesized.wav")
        ], 1)

    def test_synt_unparsed_missing(self):
        self.execute([
            ("", "unparsed"),
            ("in", "../tools/res/unparsed.xhtml"),
            ("", "en"),
            ("out", "synthesized.wav")
        ], 1)

    def test_synt_plain_start(self):
        self.execute([
            ("", "plain"),
            ("in", "../tools/res/plain.txt"),
            ("", "en"),
            ("out", "synthesized.wav"),
            ("", "--start=5")
        ], 0)

    def test_synt_plain_end(self):
        self.execute([
            ("", "plain"),
            ("in", "../tools/res/plain.txt"),
            ("", "en"),
            ("out", "synthesized.wav"),
            ("", "--end=10")
        ], 0)

    def test_synt_plain_start_end(self):
        self.execute([
            ("", "plain"),
            ("in", "../tools/res/plain.txt"),
            ("", "en"),
            ("out", "synthesized.wav"),
            ("", "--start=5"),
            ("", "--end=10")
        ], 0)

    def test_synt_plain_backwards(self):
        self.execute([
            ("", "plain"),
            ("in", "../tools/res/plain.txt"),
            ("", "en"),
            ("out", "synthesized.wav"),
            ("", "--backwards")
        ], 0)

    def test_synt_plain_quit(self):
        self.execute([
            ("", "plain"),
            ("in", "../tools/res/plain.txt"),
            ("", "en"),
            ("out", "synthesized.wav"),
            ("", "--quit-after=10.0")
        ], 0)

    def test_synt_plain_backwards_quit(self):
        self.execute([
            ("", "plain"),
            ("in", "../tools/res/plain.txt"),
            ("", "en"),
            ("out", "synthesized.wav"),
            ("", "--backwards"),
            ("", "--quit-after=10.0")
        ], 0)

    def test_synt_plain_pure(self):
        self.execute([
            ("", "plain"),
            ("in", "../tools/res/plain.txt"),
            ("", "en"),
            ("out", "synthesized.wav"),
            ("", "-r=\"c_extensions=False\"")
        ], 0)

    def test_synt_plain_cew_subprocess(self):
        self.execute([
            ("", "plain"),
            ("in", "../tools/res/plain.txt"),
            ("", "en"),
            ("out", "synthesized.wav"),
            ("", "-r=\"cew_subprocess_enabled=True\"")
        ], 0)

    def test_synt_path(self):
        path = os.path.expanduser("~")
        path = os.path.join(path, ".bin/myespeak")
        if gf.file_exists(path):
            self.execute([
                ("", "plain"),
                ("in", "../tools/res/plain.txt"),
                ("", "en"),
                ("out", "synthesized.wav"),
                ("", "-r=\"c_extensions=False|espeak_path=%s\"" % path)
            ], 0)

    def test_synt_path_bad(self):
        path = "/foo/bar/espeak"
        self.execute([
            ("", "plain"),
            ("in", "../tools/res/plain.txt"),
            ("", "en"),
            ("out", "synthesized.wav"),
            ("", "-r=\"c_extensions=False|espeak_path=%s\"" % path)
        ], 1)



if __name__ == '__main__':
    unittest.main()



