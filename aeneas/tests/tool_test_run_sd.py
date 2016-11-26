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

import os
import unittest

from aeneas.tools.run_sd import RunSDCLI
import aeneas.globalfunctions as gf


class TestRunSDCLI(unittest.TestCase):

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
        exit_code = RunSDCLI(use_sys=False).run(arguments=params)
        gf.delete_directory(output_path)
        self.assertEqual(exit_code, expected_exit_code)

    def test_help(self):
        self.execute([], 2)
        self.execute([("", "-h")], 2)
        self.execute([("", "--help")], 2)
        self.execute([("", "--help-rconf")], 2)
        self.execute([("", "--version")], 2)

    def test_sd_list(self):
        self.execute([
            ("", "list"),
            ("", "From|fairest|creatures|we|desire|increase"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3")
        ], 0)

    def test_sd_parsed(self):
        self.execute([
            ("", "parsed"),
            ("in", "../tools/res/parsed.txt"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3")
        ], 0)

    def test_sd_plain(self):
        self.execute([
            ("", "plain"),
            ("in", "../tools/res/plain.txt"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3")

        ], 0)

    def test_sd_subtitles(self):
        self.execute([
            ("", "subtitles"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3")
        ], 0)

    def test_sd_unparsed_id_regex(self):
        self.execute([
            ("", "unparsed"),
            ("in", "../tools/res/unparsed.xhtml"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3"),
            ("", "--id-regex=f[0-9]*")
        ], 0)

    def test_sd_unparsed_class_regex(self):
        self.execute([
            ("", "unparsed"),
            ("in", "../tools/res/unparsed.xhtml"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3"),
            ("", "--class-regex=ra"),
            ("", "--sort=unsorted"),
        ], 0)

    def test_sd_unparsed_sort_numeric(self):
        self.execute([
            ("", "unparsed"),
            ("in", "../tools/res/unparsed.xhtml"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3"),
            ("", "--id-regex=f[0-9]*"),
            ("", "--sort=numeric")
        ], 0)

    def test_sd_unparsed_sort_lexicographic(self):
        self.execute([
            ("", "unparsed"),
            ("in", "../tools/res/unparsed.xhtml"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3"),
            ("", "--id-regex=f[0-9]*"),
            ("", "--sort=lexicographic")
        ], 0)

    def test_sd_missing_1(self):
        self.execute([
            ("", "list"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3")
        ], 2)

    def test_sd_missing_2(self):
        self.execute([
            ("", "From|fairest|creatures|we|desire|increase"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3")
        ], 2)

    def test_sd_missing_3(self):
        self.execute([
            ("", "list"),
            ("", "From|fairest|creatures|we|desire|increase"),
            ("in", "../tools/res/audio.mp3")
        ], 2)

    def test_sd_missing_4(self):
        self.execute([
            ("", "list"),
            ("", "From|fairest|creatures|we|desire|increase"),
            ("", "eng")
        ], 2)

    def test_sd_cannot_read(self):
        self.execute([
            ("", "plain"),
            ("", "/foo/bar/baz.wav"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3")
        ], 1)

    def test_sd_unparsed_missing(self):
        self.execute([
            ("", "unparsed"),
            ("in", "../tools/res/unparsed.xhtml"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3")
        ], 1)

    def test_sd_parsed_head(self):
        self.execute([
            ("", "parsed"),
            ("in", "../tools/res/parsed.txt"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3"),
            ("", "--min-head=0.0"),
            ("", "--max-head=5.0")
        ], 0)

    def test_sd_parsed_tail(self):
        self.execute([
            ("", "parsed"),
            ("in", "../tools/res/parsed.txt"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3"),
            ("", "--min-tail=1.0"),
            ("", "--max-tail=5.0")
        ], 0)

    def test_sd_parsed_head_tail(self):
        self.execute([
            ("", "parsed"),
            ("in", "../tools/res/parsed.txt"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3"),
            ("", "--min-head=0.0"),
            ("", "--max-head=5.0"),
            ("", "--min-tail=1.0"),
            ("", "--max-tail=5.0")
        ], 0)

    def test_sd_pure(self):
        self.execute([
            ("", "list"),
            ("", "From|fairest|creatures|we|desire|increase"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3"),
            ("", "-r=\"c_extensions=False\"")
        ], 0)

    def test_sd_no_cew(self):
        self.execute([
            ("", "list"),
            ("", "From|fairest|creatures|we|desire|increase"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3"),
            ("", "-r=\"cew=False\"")
        ], 0)

    def test_sd_no_cmfcc(self):
        self.execute([
            ("", "list"),
            ("", "From|fairest|creatures|we|desire|increase"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3"),
            ("", "-r=\"cmfcc=False\"")
        ], 0)

    def test_sd_no_cdtw(self):
        self.execute([
            ("", "list"),
            ("", "From|fairest|creatures|we|desire|increase"),
            ("", "eng"),
            ("in", "../tools/res/audio.mp3"),
            ("", "-r=\"cdtw=False\"")
        ], 0)


if __name__ == "__main__":
    unittest.main()
