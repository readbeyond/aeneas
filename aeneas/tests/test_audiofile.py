#!/usr/bin/env python
# coding=utf-8

import os
import sys
import unittest

from . import get_abs_path

from aeneas.audiofile import AudioFile

class TestAudioFile(unittest.TestCase):

    def test_read(self):
        file_path = get_abs_path("res/container/job/assets/p001.mp3")
        audiofile = AudioFile(file_path)
        self.assertEqual(audiofile.file_size, 426735)
        self.assertEqual(audiofile.audio_sample_rate, 44100)
        self.assertEqual(audiofile.audio_channels, 2)
        self.assertEqual(audiofile.audio_format, 'mp3')
        self.assertEqual(int(audiofile.audio_length), 53) # 53.315918

    def test_precision(self):
        file_path = get_abs_path("res/container/job/assets/p001.mp3")
        audiofile = AudioFile(file_path)
        self.assertEqual(audiofile.audio_length, 53.315918) # might fail?

    def test_cannotload(self):
        file_path = get_abs_path("res/this_file_does_not_exist.mp3")
        with self.assertRaises(OSError):
            audiofile = AudioFile(file_path)

if __name__ == '__main__':
    unittest.main()



