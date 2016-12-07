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

from aeneas.audiofile import AudioFileUnsupportedFormatError
from aeneas.audiofilemfcc import AudioFileMFCC
import aeneas.globalfunctions as gf


class TestVAD(unittest.TestCase):

    FILES = [
        {
            "path": "res/vad/nsn.wav",
            "speech_length": 1,
            "nonspeech_length": 2,
        },
        {
            "path": "res/vad/ns.wav",
            "speech_length": 1,
            "nonspeech_length": 1,
        },
        {
            "path": "res/vad/n.wav",
            "speech_length": 0,
            "nonspeech_length": 1,
        },
        {
            "path": "res/vad/sns.wav",
            "speech_length": 2,
            "nonspeech_length": 1,
        },
        {
            "path": "res/vad/sn.wav",
            "speech_length": 1,
            "nonspeech_length": 1,
        },
        {
            "path": "res/vad/s.wav",
            "speech_length": 1,
            "nonspeech_length": 0,
        },
        {
            "path": "res/vad/zero.wav",
            "speech_length": 0,
            "nonspeech_length": 1,
        },
    ]

    NOT_EXISTING_PATH = "this_file_does_not_exist.mp3"
    EMPTY_FILE_PATH = "res/audioformats/p001.empty"

    def perform(self, input_file_path, speech_length, nonspeech_length):
        audiofile = AudioFileMFCC(gf.absolute_path(input_file_path, __file__))
        audiofile.run_vad()
        self.assertEqual(len(audiofile.intervals(speech=True)), speech_length)
        self.assertEqual(len(audiofile.intervals(speech=False)), nonspeech_length)

    def test_compute_vad(self):
        for f in self.FILES:
            self.perform(f["path"], f["speech_length"], f["nonspeech_length"])

    def test_not_existing(self):
        with self.assertRaises(OSError):
            self.perform(self.NOT_EXISTING_PATH, 0, 0)

    def test_empty(self):
        with self.assertRaises(AudioFileUnsupportedFormatError):
            self.perform(self.EMPTY_FILE_PATH, 0, 0)


if __name__ == "__main__":
    unittest.main()
