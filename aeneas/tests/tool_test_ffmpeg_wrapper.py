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
        self.execute([("", "--help-rconf")], 2)
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

    def test_convert_16000(self):
        self.execute([
            ("in", "../tools/res/audio.wav"),
            ("out", "audio.wav"),
            ("", "-r=\"ffmpeg_sample_rate=16000\"")
        ], 0)

    def test_convert_22050(self):
        self.execute([
            ("in", "../tools/res/audio.wav"),
            ("out", "audio.wav"),
            ("", "-r=\"ffmpeg_sample_rate=22050\"")
        ], 0)

    def test_convert_44100(self):
        self.execute([
            ("in", "../tools/res/audio.wav"),
            ("out", "audio.wav"),
            ("", "-r=\"ffmpeg_sample_rate=44100\"")
        ], 0)

    def test_convert_path(self):
        path = os.path.expanduser("~")
        path = os.path.join(path, ".bin/myffmpeg")
        if gf.file_exists(path):
            self.execute([
                ("in", "../tools/res/audio.wav"),
                ("out", "audio.wav"),
                ("", "-r=\"ffmpeg_path=%s\"" % path)
            ], 0)

    def test_convert_path_bad(self):
        path = "/foo/bar/ffmpeg"
        self.execute([
            ("in", "../tools/res/audio.wav"),
            ("out", "audio.wav"),
            ("", "-r=\"ffmpeg_path=%s\"" % path)
        ], 1)

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


if __name__ == "__main__":
    unittest.main()
