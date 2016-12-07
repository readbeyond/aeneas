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

import numpy
import unittest

from aeneas.audiofile import AudioFile
from aeneas.audiofile import AudioFileNotInitializedError
from aeneas.audiofile import AudioFileUnsupportedFormatError
from aeneas.exacttiming import TimeValue
import aeneas.globalfunctions as gf


class TestAudioFile(unittest.TestCase):

    AUDIO_FILE_WAVE = "res/audioformats/mono.16000.wav"
    AUDIO_FILE_EMPTY = "res/audioformats/p001.empty"
    AUDIO_FILE_NOT_WAVE = "res/audioformats/p001.mp3"
    NOT_EXISTING_FILE = "res/audioformats/x/y/z/not_existing.wav"
    FILES = [
        {
            "path": "res/audioformats/p001.aac",
            "size": 72196,
            "rate": 44100,
            "channels": 2,
            "format": "aac",
            "length": TimeValue("7.9"),     # 7.907558 Estimating duration from bitrate, this may be inaccurate
        },
        {
            "path": "res/audioformats/p001.aiff",
            "size": 1586770,
            "rate": 44100,
            "channels": 2,
            "format": "pcm_s16be",
            "length": TimeValue("9.0"),     # 8.994989
        },
        {
            "path": "res/audioformats/p001.flac",
            "size": 569729,
            "rate": 44100,
            "channels": 2,
            "format": "flac",
            "length": TimeValue("9.0"),     # 8.994989
        },
        {
            "path": "res/audioformats/p001.mp3",
            "size": 72559,
            "rate": 44100,
            "channels": 2,
            "format": "mp3",
            "length": TimeValue("9.0"),     # 9.038367
        },
        {
            "path": "res/audioformats/p001.mp4",
            "size": 74579,
            "rate": 44100,
            "channels": 2,
            "format": "aac",
            "length": TimeValue("9.0"),     # 9.018209
        },
        {
            "path": "res/audioformats/p001.ogg",
            "size": 56658,
            "rate": 44100,
            "channels": 2,
            "format": "vorbis",
            "length": TimeValue("9.0"),     # 8.994989
        },
        {
            "path": "res/audioformats/p001.wav",
            "size": 1586760,
            "rate": 44100,
            "channels": 2,
            "format": "pcm_s16le",
            "length": TimeValue("9.0"),     # 8.994989
        },
        {
            "path": "res/audioformats/p001.webm",
            "size": 59404,
            "rate": 44100,
            "channels": 2,
            "format": "vorbis",
            "length": TimeValue("9.0"),     # 9.0
        },
    ]

    def load(self, path, rp=False, rs=False):
        af = AudioFile(gf.absolute_path(path, __file__))
        if rp:
            af.read_properties()
        if rs:
            af.read_samples_from_file()
        return af

    def test_read_properties_from_none(self):
        with self.assertRaises(OSError):
            audiofile = self.load(None, rp=True)

    def test_read_properties_from_non_existing_path(self):
        with self.assertRaises(OSError):
            audiofile = self.load("not_existing.mp3", rp=True)

    def test_read_properties_from_empty(self):
        with self.assertRaises(AudioFileUnsupportedFormatError):
            audiofile = self.load(self.AUDIO_FILE_EMPTY, rp=True)

    def test_str(self):
        audiofile = self.load(self.AUDIO_FILE_WAVE, rp=True)
        ignored = str(audiofile)

    def test_read_properties_formats(self):
        for f in self.FILES:
            audiofile = self.load(f["path"], rp=True)
            self.assertEqual(audiofile.file_size, f["size"])
            self.assertEqual(audiofile.audio_sample_rate, f["rate"])
            self.assertEqual(audiofile.audio_channels, f["channels"])
            self.assertEqual(audiofile.audio_format, f["format"])
            self.assertAlmostEqual(audiofile.audio_length, f["length"], places=1)

    def test_read_samples_from_none(self):
        with self.assertRaises(OSError):
            audiofile = self.load(None, rs=True)

    def test_read_samples_from_non_existing_path(self):
        with self.assertRaises(OSError):
            audiofile = self.load(self.NOT_EXISTING_FILE, rs=True)

    def test_read_samples_from_empty(self):
        with self.assertRaises(AudioFileUnsupportedFormatError):
            audiofile = self.load(self.AUDIO_FILE_EMPTY, rs=True)

    def test_read_samples_from_file(self):
        audiofile = self.load(self.AUDIO_FILE_WAVE, rs=True)
        self.assertIsNotNone(audiofile.audio_samples)
        audiofile.clear_data()

    def test_clear_data(self):
        audiofile = self.load(self.AUDIO_FILE_WAVE, rs=True)
        audiofile.clear_data()

    def test_length(self):
        audiofile = self.load(self.AUDIO_FILE_WAVE, rs=True)
        audiofile.clear_data()
        self.assertAlmostEqual(audiofile.audio_length, TimeValue("53.3"), places=1)     # 53.266

    def test_add_samples_file(self):
        audiofile = self.load(self.AUDIO_FILE_WAVE, rs=True)
        data = audiofile.audio_samples
        old_length = audiofile.audio_length
        audiofile.add_samples(data)
        new_length = audiofile.audio_length
        self.assertAlmostEqual(new_length, 2 * old_length, places=1)

    def test_add_samples_reverse_file(self):
        audiofile = self.load(self.AUDIO_FILE_WAVE, rs=True)
        data = audiofile.audio_samples
        old_length = audiofile.audio_length
        audiofile.add_samples(data, reverse=True)
        new_length = audiofile.audio_length
        self.assertAlmostEqual(new_length, 2 * old_length, places=1)

    def test_reverse(self):
        audiofile = self.load(self.AUDIO_FILE_WAVE, rs=True)
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
            [None, None, TimeValue("53.3")],
            [TimeValue("1.0"), None, TimeValue("52.3")],
            [None, TimeValue("52.3"), TimeValue("52.3")],
            [TimeValue("1.0"), TimeValue("51.3"), TimeValue("51.3")],
            [TimeValue("0.0"), None, TimeValue("53.3")],
            [None, TimeValue("60.0"), TimeValue("53.3")],
            [TimeValue("-1.0"), None, TimeValue("53.3")],
            [TimeValue("0.0"), TimeValue("-60.0"), TimeValue("0.0")],
            [TimeValue("10.0"), TimeValue("50.0"), TimeValue("43.3")]
        ]

        for interval in intervals:
            audiofile = self.load(self.AUDIO_FILE_WAVE, rs=True)
            audiofile.trim(interval[0], interval[1])
            self.assertAlmostEqual(audiofile.audio_length, interval[2], places=1)   # 53.315918
            audiofile.clear_data()

    def test_write_not_existing_path(self):
        output_file_path = gf.absolute_path(self.NOT_EXISTING_FILE, __file__)
        audiofile = self.load(self.AUDIO_FILE_WAVE, rs=True)
        with self.assertRaises(OSError):
            audiofile.write(output_file_path)

    def test_write(self):
        audiofile = self.load(self.AUDIO_FILE_WAVE, rs=True)
        data = audiofile.audio_samples
        handler, output_file_path = gf.tmp_file(suffix=".wav")
        audiofile.write(output_file_path)
        audiocopy = self.load(output_file_path)
        datacopy = audiocopy.audio_samples
        self.assertTrue((datacopy == data).all())
        gf.delete_file(handler, output_file_path)

    def test_create_none(self):
        audiofile = AudioFile()

    def test_preallocate(self):
        audiofile = AudioFile()
        with self.assertRaises(AudioFileNotInitializedError):
            audiofile.audio_samples
        audiofile.preallocate_memory(100)
        self.assertEqual(len(audiofile.audio_samples), 0)

    def test_preallocate_bigger(self):
        audiofile = AudioFile()
        audiofile.preallocate_memory(100)
        self.assertEqual(len(audiofile.audio_samples), 0)
        audiofile.add_samples(numpy.array([1, 2, 3, 4, 5]))
        self.assertEqual(len(audiofile.audio_samples), 5)
        audiofile.preallocate_memory(500)
        self.assertEqual(len(audiofile.audio_samples), 5)

    def test_preallocate_smaller(self):
        audiofile = AudioFile()
        audiofile.preallocate_memory(100)
        self.assertEqual(len(audiofile.audio_samples), 0)
        audiofile.add_samples(numpy.array([1, 2, 3, 4, 5]))
        self.assertEqual(len(audiofile.audio_samples), 5)
        audiofile.preallocate_memory(2)
        self.assertEqual(len(audiofile.audio_samples), 2)

    def test_add_samples_memory(self):
        audiofile = AudioFile()
        audiofile.add_samples(numpy.array([1, 2, 3, 4, 5]))
        audiofile.add_samples(numpy.array([6, 7, 8, 9, 10]))
        self.assertEqual(len(audiofile.audio_samples), 10)
        self.assertEqual(audiofile.audio_samples[0], 1)
        self.assertEqual(audiofile.audio_samples[1], 2)
        self.assertEqual(audiofile.audio_samples[4], 5)
        self.assertEqual(audiofile.audio_samples[5], 6)
        self.assertEqual(audiofile.audio_samples[6], 7)
        self.assertEqual(audiofile.audio_samples[9], 10)

    def test_add_samples_reverse_memory(self):
        audiofile = AudioFile()
        audiofile.add_samples(numpy.array([1, 2, 3, 4, 5]), reverse=True)
        audiofile.add_samples(numpy.array([6, 7, 8, 9, 10]), reverse=True)
        self.assertEqual(len(audiofile.audio_samples), 10)
        self.assertEqual(audiofile.audio_samples[0], 5)
        self.assertEqual(audiofile.audio_samples[1], 4)
        self.assertEqual(audiofile.audio_samples[4], 1)
        self.assertEqual(audiofile.audio_samples[5], 10)
        self.assertEqual(audiofile.audio_samples[6], 9)
        self.assertEqual(audiofile.audio_samples[9], 6)


if __name__ == "__main__":
    unittest.main()
