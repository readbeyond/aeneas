#!/usr/bin/env python
# coding=utf-8

import unittest

from aeneas.ffprobewrapper import FFPROBEUnsupportedFormatError
from aeneas.ffprobewrapper import FFPROBEWrapper
import aeneas.tests as at

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
        return prober.read_properties(at.get_abs_path(input_file_path))

    def test_mp3_properties(self):
        properties = self.load("res/audioformats/p001.mp3")
        self.assertNotEqual(properties['bit_rate'], None)
        self.assertNotEqual(properties['channels'], None)
        self.assertNotEqual(properties['codec_name'], None)
        self.assertNotEqual(properties['duration'], None)
        self.assertNotEqual(properties['sample_rate'], None)

    def test_path_none(self):
        with self.assertRaises(TypeError):
            self.load(None)

    def test_path_not_existing(self):
        with self.assertRaises(IOError):
            self.load(self.NOT_EXISTING_PATH)

    def test_file_empty(self):
        with self.assertRaises(FFPROBEUnsupportedFormatError):
            self.load(self.EMPTY_FILE_PATH)

    def test_formats(self):
        for f in self.FILES:
            properties = self.load(f["path"])
            self.assertNotEqual(properties['duration'], None)

if __name__ == '__main__':
    unittest.main()



