#!/usr/bin/env python
# coding=utf-8

import os
import sys
import unittest

from . import get_abs_path

from aeneas.ffprobewrapper import FFPROBEWrapper

class TestFFPROBEWrapper(unittest.TestCase):

    def test_read(self):
        file_path = get_abs_path("res/container/job/assets/p001.mp3")
        prober = FFPROBEWrapper()
        properties = prober.read_properties(file_path)
        self.assertEqual(properties['bit_rate'], '64000')
        self.assertEqual(properties['channels'], '2')
        self.assertEqual(properties['codec_name'], 'mp3')
        self.assertEqual(int(properties['duration']), 53) # '53.315918'
        self.assertEqual(properties['sample_rate'], '44100')

    def test_precision(self):
        file_path = get_abs_path("res/container/job/assets/p001.mp3")
        prober = FFPROBEWrapper()
        properties = prober.read_properties(file_path)
        self.assertEqual(properties['duration'], 53.315918) # might fail?

    def test_cannotload(self):
        file_path = get_abs_path("res/this_file_does_not_exist.mp3")
        prober = FFPROBEWrapper()
        with self.assertRaises(OSError):
            properties = prober.read_properties(file_path)

    def test_format_wav(self):
        file_path = get_abs_path("res/audioformats/p001.wav")
        prober = FFPROBEWrapper()
        properties = prober.read_properties(file_path)
        self.assertNotEqual(properties['duration'], None)

    def test_format_mp3(self):
        file_path = get_abs_path("res/audioformats/p001.mp3")
        prober = FFPROBEWrapper()
        properties = prober.read_properties(file_path)
        self.assertNotEqual(properties['duration'], None)

    def test_format_mp4(self):
        file_path = get_abs_path("res/audioformats/p001.mp4")
        prober = FFPROBEWrapper()
        properties = prober.read_properties(file_path)
        self.assertNotEqual(properties['duration'], None)

    def test_format_flac(self):
        file_path = get_abs_path("res/audioformats/p001.flac")
        prober = FFPROBEWrapper()
        properties = prober.read_properties(file_path)
        self.assertNotEqual(properties['duration'], None)

    def test_format_ogg(self):
        file_path = get_abs_path("res/audioformats/p001.ogg")
        prober = FFPROBEWrapper()
        properties = prober.read_properties(file_path)
        self.assertNotEqual(properties['duration'], None)

    def test_format_aac(self):
        file_path = get_abs_path("res/audioformats/p001.aac")
        prober = FFPROBEWrapper()
        properties = prober.read_properties(file_path)
        self.assertNotEqual(properties['duration'], None)

    def test_format_webm(self):
        file_path = get_abs_path("res/audioformats/p001.webm")
        prober = FFPROBEWrapper()
        properties = prober.read_properties(file_path)
        self.assertNotEqual(properties['duration'], None)

    def test_format_aiff(self):
        file_path = get_abs_path("res/audioformats/p001.aiff")
        prober = FFPROBEWrapper()
        properties = prober.read_properties(file_path)
        self.assertNotEqual(properties['duration'], None)

if __name__ == '__main__':
    unittest.main()



