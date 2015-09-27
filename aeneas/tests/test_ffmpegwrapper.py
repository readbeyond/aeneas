#!/usr/bin/env python
# coding=utf-8

import os
import tempfile
import unittest

from . import get_abs_path, delete_directory

from aeneas.ffmpegwrapper import FFMPEGWrapper

class TestFFMPEGWrapper(unittest.TestCase):

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

    def convert(self, input_file_path):
        output_path = tempfile.mkdtemp()
        output_file_path = os.path.join(output_path, "audio.wav")
        converter = FFMPEGWrapper()
        result = converter.convert(get_abs_path(input_file_path), output_file_path)
        self.assertEqual(result, output_file_path)
        delete_directory(output_path)

    def test_convert(self):
        for f in self.FILES:
            self.convert(f["path"])

    def test_not_existing(self):
        with self.assertRaises(OSError):
            self.convert(self.NOT_EXISTING_PATH)

    def test_empty(self):
        with self.assertRaises(OSError):
            self.convert(self.EMPTY_FILE_PATH)

if __name__ == '__main__':
    unittest.main()



