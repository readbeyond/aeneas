#!/usr/bin/env python
# coding=utf-8

import numpy
import os
import unittest

from aeneas.audiofile import AudioFile
from aeneas.audiofile import AudioFileMonoWAVE
from aeneas.audiofile import AudioFileMonoWAVENotInitialized
from aeneas.audiofile import AudioFileUnsupportedFormatError
import aeneas.globalfunctions as gf

class TestAudioFile(unittest.TestCase):

    AUDIO_FILE_EMPTY = "res/audioformats/p001.empty"
    AUDIO_FILE_NOT_WAVE = "res/audioformats/p001.mp3"
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
        return AudioFile(gf.absolute_path(path, __file__))

    def test_read_on_none(self):
        audiofile = self.load(None)
        with self.assertRaises(OSError):
            audiofile.read_properties()

    def test_read_on_non_existing_path(self):
        audiofile = self.load("not_existing.mp3")
        with self.assertRaises(OSError):
            audiofile.read_properties()

    def test_read_on_empty(self):
        audiofile = self.load(self.AUDIO_FILE_EMPTY)
        with self.assertRaises(AudioFileUnsupportedFormatError):
            audiofile.read_properties()

    def test_str(self):
        audiofile = self.load(self.FILES[0]["path"])
        audiofile.read_properties()
        ignored = str(audiofile)

    def test_read(self):
        for f in self.FILES:
            audiofile = self.load(f["path"])
            audiofile.read_properties()
            self.assertEqual(audiofile.file_size, f["size"])
            self.assertEqual(audiofile.audio_sample_rate, f["rate"])
            self.assertEqual(audiofile.audio_channels, f["channels"])
            self.assertEqual(audiofile.audio_format, f["format"])
            self.assertAlmostEqual(audiofile.audio_length, f["length"], places=1)



class TestAudioFileMonoWAVE(unittest.TestCase):

    AUDIO_FILE_WAVE = "res/audioformats/mono.16000.wav"
    AUDIO_FILE_EMPTY = "res/audioformats/p001.empty"
    AUDIO_FILE_NOT_WAVE = "res/audioformats/p001.mp3"
    NOT_EXISTING_FILE = "res/audioformats/x/y/z/not_existing.wav"

    def load(self, path):
        return AudioFileMonoWAVE(gf.absolute_path(path, __file__))

    def test_load_on_none(self):
        audiofile = self.load(None)
        with self.assertRaises(OSError):
            audiofile.read_samples_from_file()

    def test_load_on_non_existing_path(self):
        with self.assertRaises(OSError):
            audiofile = self.load(self.NOT_EXISTING_FILE)

    def test_load_on_empty(self):
        with self.assertRaises(AudioFileUnsupportedFormatError):
            audiofile = self.load(self.AUDIO_FILE_EMPTY)

    def test_load_not_wave_file(self):
        with self.assertRaises(AudioFileUnsupportedFormatError):
            audiofile = self.load(self.AUDIO_FILE_NOT_WAVE)

    def test_read_samples_from_file(self):
        audiofile = self.load(self.AUDIO_FILE_WAVE)
        self.assertIsNotNone(audiofile.audio_samples)
        audiofile.clear_data()

    def test_clear_data(self):
        audiofile = self.load(self.AUDIO_FILE_WAVE)
        audiofile.clear_data()
        with self.assertRaises(AudioFileMonoWAVENotInitialized):
            audiofile.audio_samples

    def test_length(self):
        audiofile = self.load(self.AUDIO_FILE_WAVE)
        audiofile.clear_data()
        self.assertAlmostEqual(audiofile.audio_length, 53.3, places=1) # 53.266

    def test_append_file(self):
        audiofile = self.load(self.AUDIO_FILE_WAVE)
        data = audiofile.audio_samples
        old_length = audiofile.audio_length
        audiofile.append(data)
        new_length = audiofile.audio_length
        self.assertAlmostEqual(new_length, 2 * old_length, places=1)

    def test_append_reverse_file(self):
        audiofile = self.load(self.AUDIO_FILE_WAVE)
        data = audiofile.audio_samples
        old_length = audiofile.audio_length
        audiofile.append(data, reverse=True)
        new_length = audiofile.audio_length
        self.assertAlmostEqual(new_length, 2 * old_length, places=1)

    def test_prepend_file(self):
        audiofile = self.load(self.AUDIO_FILE_WAVE)
        data = audiofile.audio_samples
        old_length = audiofile.audio_length
        audiofile.prepend(data)
        new_length = audiofile.audio_length
        self.assertAlmostEqual(new_length, 2 * old_length, places=1)

    def test_reverse(self):
        audiofile = self.load(self.AUDIO_FILE_WAVE)
        data = numpy.array(audiofile.audio_samples)
        audiofile.reverse()
        rev1 = numpy.array(audiofile.audio_samples)
        self.assertFalse((data == rev1).all())
        audiofile.reverse()
        rev2 = numpy.array(audiofile.audio_samples)
        self.assertTrue((data == rev2).all())
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
            audiofile = self.load(self.AUDIO_FILE_WAVE)
            audiofile.trim(interval[0], interval[1])
            self.assertAlmostEqual(audiofile.audio_length, interval[2], places=1) # 53.315918
            audiofile.clear_data()

    def test_write_not_existing_path(self):
        output_file_path = gf.absolute_path(self.NOT_EXISTING_FILE, __file__)
        audiofile = self.load(self.AUDIO_FILE_WAVE)
        with self.assertRaises(OSError):
            audiofile.write(output_file_path)

    def test_write(self):
        audiofile = self.load(self.AUDIO_FILE_WAVE)
        data = audiofile.audio_samples
        handler, output_file_path = gf.tmp_file(suffix=".wav")
        audiofile.write(output_file_path)
        audiocopy = self.load(output_file_path)
        datacopy = audiocopy.audio_samples
        self.assertTrue((datacopy == data).all())
        gf.delete_file(handler, output_file_path)

    def test_create_none(self):
        audiofile = AudioFileMonoWAVE()

    def test_preallocate(self):
        audiofile = AudioFileMonoWAVE()
        with self.assertRaises(AudioFileMonoWAVENotInitialized):
            audiofile.audio_samples
        audiofile.preallocate_memory(100)
        self.assertEqual(len(audiofile.audio_samples), 0)

    def test_preallocate_bigger(self):
        audiofile = AudioFileMonoWAVE()
        audiofile.preallocate_memory(100)
        self.assertEqual(len(audiofile.audio_samples), 0)
        audiofile.append(numpy.array([1, 2, 3, 4, 5]))
        self.assertEqual(len(audiofile.audio_samples), 5)
        audiofile.preallocate_memory(500)
        self.assertEqual(len(audiofile.audio_samples), 5)

    def test_preallocate_smaller(self):
        audiofile = AudioFileMonoWAVE()
        audiofile.preallocate_memory(100)
        self.assertEqual(len(audiofile.audio_samples), 0)
        audiofile.append(numpy.array([1, 2, 3, 4, 5]))
        self.assertEqual(len(audiofile.audio_samples), 5)
        audiofile.preallocate_memory(2)
        self.assertEqual(len(audiofile.audio_samples), 2)

    def test_append_memory(self):
        audiofile = AudioFileMonoWAVE()
        audiofile.append(numpy.array([1, 2, 3, 4, 5]))
        audiofile.append(numpy.array([6, 7, 8, 9, 10]))
        self.assertEqual(len(audiofile.audio_samples), 10)
        self.assertEqual(audiofile.audio_samples[0], 1)
        self.assertEqual(audiofile.audio_samples[1], 2)
        self.assertEqual(audiofile.audio_samples[4], 5)
        self.assertEqual(audiofile.audio_samples[5], 6)
        self.assertEqual(audiofile.audio_samples[6], 7)
        self.assertEqual(audiofile.audio_samples[9], 10)

    def test_append_reverse_memory(self):
        audiofile = AudioFileMonoWAVE()
        audiofile.append(numpy.array([1, 2, 3, 4, 5]), reverse=True)
        audiofile.append(numpy.array([6, 7, 8, 9, 10]), reverse=True)
        self.assertEqual(len(audiofile.audio_samples), 10)
        self.assertEqual(audiofile.audio_samples[0], 5)
        self.assertEqual(audiofile.audio_samples[1], 4)
        self.assertEqual(audiofile.audio_samples[4], 1)
        self.assertEqual(audiofile.audio_samples[5], 10)
        self.assertEqual(audiofile.audio_samples[6], 9)
        self.assertEqual(audiofile.audio_samples[9], 6)

    def test_prepend_memory(self):
        audiofile = AudioFileMonoWAVE()
        audiofile.prepend(numpy.array([1, 2, 3, 4, 5]))
        audiofile.prepend(numpy.array([6, 7, 8, 9, 10]))
        self.assertEqual(len(audiofile.audio_samples), 10)
        self.assertEqual(audiofile.audio_samples[0], 6)
        self.assertEqual(audiofile.audio_samples[1], 7)
        self.assertEqual(audiofile.audio_samples[4], 10)
        self.assertEqual(audiofile.audio_samples[5], 1)
        self.assertEqual(audiofile.audio_samples[6], 2)
        self.assertEqual(audiofile.audio_samples[9], 5)



if __name__ == '__main__':
    unittest.main()



