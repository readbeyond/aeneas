#!/usr/bin/env python
# coding=utf-8

import os
import tempfile
import unittest

from aeneas.ffmpegwrapper import FFMPEGWrapper
import aeneas.globalfunctions as gf
import aeneas.tests as at

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

    CANNOT_BE_WRITTEN_PATH = "x/y/z/cannot_be_written.wav"
    NOT_EXISTING_PATH = "this_file_does_not_exist.mp3"
    EMPTY_FILE_PATH = "res/audioformats/p001.empty"

    def convert(self, input_file_path, ofp=None):
        if ofp is None:
            output_path = tempfile.mkdtemp()
            output_file_path = os.path.join(output_path, "audio.wav")
        else:
            output_file_path = ofp
        try:
            converter = FFMPEGWrapper()
            result = converter.convert(at.get_abs_path(input_file_path), output_file_path)
            self.assertEqual(result, output_file_path)
            gf.delete_directory(output_path)
        except IOError as exc:
            if ofp is None:
                gf.delete_directory(output_path)
            else:
                gf.delete_file(None, ofp)
            raise exc

    def test_convert(self):
        for f in self.FILES:
            self.convert(f["path"])

    def test_not_existing(self):
        with self.assertRaises(IOError):
            self.convert(self.NOT_EXISTING_PATH)

    def test_empty(self):
        with self.assertRaises(IOError):
            self.convert(self.EMPTY_FILE_PATH)

    def test_cannot_be_written(self):
        with self.assertRaises(IOError):
            self.convert(self.FILES[0]["path"], self.CANNOT_BE_WRITTEN_PATH)

if __name__ == '__main__':
    unittest.main()



