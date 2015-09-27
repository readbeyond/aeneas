#!/usr/bin/env python
# coding=utf-8

import os
import tempfile
import unittest

from . import get_abs_path, delete_file

from aeneas.audiofile import AudioFile

class TestAudioFile(unittest.TestCase):

    AUDIO_FILE_PATH_MFCC = "res/cmfcc/audio.wav"

    FILES = [
        {
            "path": "res/audioformats/p001.aac",
            "size": 72196,
            "rate": 44100,
            "channels": 2,
            "format": "aac",
            "length": 7.9, # 7.907558 Estimating duration from bitrate, this may be inaccurate
        },
        {
            "path": "res/audioformats/p001.aiff",
            "size": 1586770,
            "rate": 44100,
            "channels": 2,
            "format": "pcm_s16be",
            "length": 9.0, # 8.994989
        },
        {
            "path": "res/audioformats/p001.flac",
            "size": 569729,
            "rate": 44100,
            "channels": 2,
            "format": "flac",
            "length": 9.0, # 8.994989
        },
        {
            "path": "res/audioformats/p001.mp3",
            "size": 72559,
            "rate": 44100,
            "channels": 2,
            "format": "mp3",
            "length": 9.0, # 9.038367
        },
        {
            "path": "res/audioformats/p001.mp4",
            "size": 74579,
            "rate": 44100,
            "channels": 2,
            "format": "aac",
            "length": 9.0, # 9.018209
        },
        {
            "path": "res/audioformats/p001.ogg",
            "size": 56658,
            "rate": 44100,
            "channels": 2,
            "format": "vorbis",
            "length": 9.0, # 8.994989
        },
        {
            "path": "res/audioformats/p001.wav",
            "size": 1586760,
            "rate": 44100,
            "channels": 2,
            "format": "pcm_s16le",
            "length": 9.0, # 8.994989
        },
        {
            "path": "res/audioformats/p001.webm",
            "size": 59404,
            "rate": 44100,
            "channels": 2,
            "format": "vorbis",
            "length": 9.0, # 9.0
        },
    ]

    def load(self, path):
        return AudioFile(get_abs_path(path))

    def test_cannot_load(self):
        audiofile = self.load("this_file_does_not_exist.mp3")
        with self.assertRaises(OSError):
            audiofile.read_properties()

    def test_read(self):
        for f in self.FILES:
            audiofile = self.load(f["path"])
            audiofile.read_properties()
            self.assertEqual(audiofile.file_size, f["size"])
            self.assertEqual(audiofile.audio_sample_rate, f["rate"])
            self.assertEqual(audiofile.audio_channels, f["channels"])
            self.assertEqual(audiofile.audio_format, f["format"])
            self.assertAlmostEqual(audiofile.audio_length, f["length"], places=1)

    def test_load_data(self):
        audiofile = self.load(self.AUDIO_FILE_PATH_MFCC)
        audiofile.load_data()
        self.assertNotEqual(audiofile.audio_data, None)
        audiofile.clear_data()

    def test_clear_data(self):
        audiofile = self.load(self.AUDIO_FILE_PATH_MFCC)
        audiofile.load_data()
        audiofile.clear_data()
        self.assertEqual(audiofile.audio_data, None)

    def test_extract_mfcc(self):
        audiofile = self.load(self.AUDIO_FILE_PATH_MFCC)
        audiofile.load_data()
        audiofile.extract_mfcc()
        audiofile.clear_data()
        self.assertNotEqual(audiofile.audio_mfcc, None)
        self.assertEqual(audiofile.audio_mfcc.shape[0], 13)
        self.assertEqual(audiofile.audio_mfcc.shape[1], 1332)

    def test_length(self):
        audiofile = self.load(self.AUDIO_FILE_PATH_MFCC)
        audiofile.load_data()
        audiofile.clear_data()
        self.assertAlmostEqual(audiofile.audio_length, 53.3, places=1) # 53.315918

    def test_reverse(self):
        audiofile = self.load(self.AUDIO_FILE_PATH_MFCC)
        audiofile.load_data()
        audiofile.reverse()
        self.assertAlmostEqual(audiofile.audio_length, 53.3, places=1) # 53.315918
        audiofile.clear_data()

    def test_trim(self):
        intervals = [
            [None, None, 53.3],
            [1.0, None, 52.3],
            [None, 52.3, 52.3],
            [1.0, 51.3, 51.3],
            [0.0, None, 53.3],
            [None, 60.0, 53.3],
            [-1.0, None, 53.3],
            [0.0, -60.0, 0.0],
            [10.0, 50.0, 43.3]
        ]

        for interval in intervals:
            audiofile = self.load(self.AUDIO_FILE_PATH_MFCC)
            audiofile.load_data()
            audiofile.trim(interval[0], interval[1])
            self.assertAlmostEqual(audiofile.audio_length, interval[2], places=1) # 53.315918
            audiofile.clear_data()

    def test_write(self):
        audiofile = self.load(self.AUDIO_FILE_PATH_MFCC)
        audiofile.load_data()
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        audiofile.write(output_file_path)
        self.assertTrue(os.path.isfile(output_file_path))
        audiocopy = self.load(output_file_path)
        audiocopy.load_data()
        self.assertEqual(audiofile.audio_length, audiocopy.audio_length)
        delete_file(handler, output_file_path)

if __name__ == '__main__':
    unittest.main()



