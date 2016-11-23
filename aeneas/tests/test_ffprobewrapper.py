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

import unittest

from aeneas.ffprobewrapper import FFPROBEUnsupportedFormatError
from aeneas.ffprobewrapper import FFPROBEWrapper
import aeneas.globalfunctions as gf


class TestFFPROBEWrapper(unittest.TestCase):

    FILES = [
        {
            "path": "res/audioformats/p001.aac",
        },
        {
            "path": "res/audioformats/p001.aiff",
        },
        {
            "path": "res/audioformats/p001.flac",
        },
        {
            "path": "res/audioformats/p001.mp3",
        },
        {
            "path": "res/audioformats/p001.mp4",
        },
        {
            "path": "res/audioformats/p001.ogg",
        },
        {
            "path": "res/audioformats/p001.wav",
        },
        {
            "path": "res/audioformats/p001.webm",
        },
    ]

    NOT_EXISTING_PATH = "this_file_does_not_exist.mp3"
    EMPTY_FILE_PATH = "res/audioformats/p001.empty"

    def load(self, input_file_path):
        prober = FFPROBEWrapper()
        return prober.read_properties(
            gf.absolute_path(input_file_path, __file__)
        )

    def test_mp3_properties(self):
        properties = self.load("res/audioformats/p001.mp3")
        self.assertIsNotNone(properties["bit_rate"])
        self.assertIsNotNone(properties["channels"])
        self.assertIsNotNone(properties["codec_name"])
        self.assertIsNotNone(properties["duration"])
        self.assertIsNotNone(properties["sample_rate"])

    def test_path_none(self):
        with self.assertRaises(TypeError):
            self.load(None)

    def test_path_not_existing(self):
        with self.assertRaises(OSError):
            self.load(self.NOT_EXISTING_PATH)

    def test_file_empty(self):
        with self.assertRaises(FFPROBEUnsupportedFormatError):
            self.load(self.EMPTY_FILE_PATH)

    def test_formats(self):
        for f in self.FILES:
            properties = self.load(f["path"])
            self.assertIsNotNone(properties["duration"])


if __name__ == "__main__":
    unittest.main()
