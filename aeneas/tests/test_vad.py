#!/usr/bin/env python
# coding=utf-8

import unittest

from aeneas.audiofile import AudioFileMonoWAVE
from aeneas.audiofile import AudioFileUnsupportedFormatError
from aeneas.vad import VAD
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
        audiofile = AudioFileMonoWAVE(gf.get_abs_path(input_file_path, __file__))
        audiofile.extract_mfcc()
        vad = VAD(audiofile.audio_mfcc, audiofile.audio_length)
        vad.compute_vad()
        self.assertEqual(len(vad.speech), speech_length)
        self.assertEqual(len(vad.nonspeech), nonspeech_length)

    def test_compute_vad(self):
        for f in self.FILES:
            self.perform(f["path"], f["speech_length"], f["nonspeech_length"])

    def test_not_existing(self):
        with self.assertRaises(OSError):
            self.perform(self.NOT_EXISTING_PATH, 0, 0)

    def test_empty(self):
        with self.assertRaises(AudioFileUnsupportedFormatError):
            self.perform(self.EMPTY_FILE_PATH, 0, 0)

if __name__ == '__main__':
    unittest.main()



