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
        self.execute([("", "--help-rconf")], 2)
        self.execute([("", "--version")], 2)

    def test_run_both(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "both"),
            ("out", "both.txt"),
        ], 0)

    def test_run_both_stdout(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "both"),
        ], 0)

    def test_run_speech(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "speech"),
            ("out", "speech.txt"),
        ], 0)

    def test_run_speech_stdout(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "speech"),
        ], 0)

    def test_run_nonspeech(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "nonspeech"),
            ("out", "nonspeech.txt"),
        ], 0)

    def test_run_nonspeech_stdout(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "nonspeech"),
        ], 0)

    def test_run_pure(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "both"),
            ("out", "both.txt"),
            ("", "-r=\"c_extensions=False\""),
        ], 0)

    def test_run_no_cmfcc(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "both"),
            ("out", "both.txt"),
            ("", "-r=\"cmfcc=False\""),
        ], 0)

    def test_run_extend_after(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "both"),
            ("out", "both.txt"),
            ("", "-r=\"vad_extend_speech_after=0.100\""),
        ], 0)

    def test_run_extend_before(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "both"),
            ("out", "both.txt"),
            ("", "-r=\"vad_extend_speech_before=0.100\""),
        ], 0)

    def test_run_energy_threshold(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "both"),
            ("out", "both.txt"),
            ("", "-r=\"vad_log_energy_threshold=0.8\""),
        ], 0)

    def test_run_min_nonspeech(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "both"),
            ("out", "both.txt"),
            ("", "-r=\"vad_min_nonspeech_length=0.5\""),
        ], 0)

    def test_run_cannot_read(self):
        self.execute([
            ("", "/foo/bar/baz.wav"),
            ("", "both"),
            ("out", "both.txt"),
        ], 1)

    def test_run_cannot_write(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "both"),
            ("", "/foo/bar/baz.txt"),
        ], 1)

    def test_run_missing_1(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("out", "both.txt"),
        ], 2)

    def test_run_missing_2(self):
        self.execute([
            ("", "both"),
            ("out", "both.txt"),
        ], 2)

    def test_run_bad(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "foo"),
            ("out", "both.txt"),
        ], 2)


if __name__ == "__main__":
    unittest.main()
