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

from aeneas.audiofilemfcc import AudioFileMFCC
from aeneas.language import Language
from aeneas.sd import SD
from aeneas.textfile import TextFile
from aeneas.textfile import TextFileFormat
import aeneas.globalfunctions as gf


class TestSD(unittest.TestCase):

    AUDIO_FILE = gf.absolute_path("res/audioformats/mono.16000.wav", __file__)
    TEXT_FILE = gf.absolute_path("res/inputtext/sonnet_plain.txt", __file__)

    def load(self):
        audio_file_mfcc = AudioFileMFCC(self.AUDIO_FILE)
        text_file = TextFile(self.TEXT_FILE, file_format=TextFileFormat.PLAIN)
        text_file.set_language(Language.ENG)
        return SD(audio_file_mfcc, text_file)

    def test_create_sd(self):
        sd = self.load()

    def test_detect_interval(self):
        begin, end = self.load().detect_interval()
        self.assertNotEqual(begin, 0.0)
        self.assertNotEqual(end, 0.0)

    def test_detect_head(self):
        head = self.load().detect_head()
        self.assertNotEqual(head, 0.0)

    def test_detect_head_min_max(self):
        head = self.load().detect_head(min_head_length=2.0, max_head_length=10.0)
        self.assertNotEqual(head, 0.0)
        self.assertGreaterEqual(head, 2.0)
        self.assertLessEqual(head, 10.0)

    def test_detect_tail(self):
        tail = self.load().detect_tail()
        self.assertNotEqual(tail, 0.0)

    def test_detect_tail_min_max(self):
        tail = self.load().detect_tail(min_tail_length=2.0, max_tail_length=10.0)
        self.assertNotEqual(tail, 0.0)
        self.assertGreaterEqual(tail, 2.0)
        self.assertLessEqual(tail, 10.0)

    def test_detect_bad(self):
        sd = self.load()
        with self.assertRaises(TypeError):
            begin, end = sd.detect_interval(min_head_length="foo")
        with self.assertRaises(ValueError):
            begin, end = sd.detect_interval(min_head_length=-10.0)
        with self.assertRaises(TypeError):
            begin, end = sd.detect_interval(max_head_length="foo")
        with self.assertRaises(ValueError):
            begin, end = sd.detect_interval(max_head_length=-10.0)
        with self.assertRaises(TypeError):
            begin, end = sd.detect_interval(min_tail_length="foo")
        with self.assertRaises(ValueError):
            begin, end = sd.detect_interval(min_tail_length=-10.0)
        with self.assertRaises(TypeError):
            begin, end = sd.detect_interval(max_tail_length="foo")
        with self.assertRaises(ValueError):
            begin, end = sd.detect_interval(max_tail_length=-10.0)


if __name__ == "__main__":
    unittest.main()
