#!/usr/bin/env python
# coding=utf-8

import os
import unittest

from aeneas.audiofilemfcc import AudioFileMFCC
from aeneas.language import Language
from aeneas.sd import SD
from aeneas.textfile import TextFile
from aeneas.textfile import TextFileFormat
import aeneas.globalfunctions as gf

class TestSD(unittest.TestCase):

    AUDIO_FILE = gf.absolute_path("res/audioformats/mono.16000.wav", __file__)
    TEXT_FILE = gf.absolute_path("res/inputtext/sonnet_plain.txt", __file__)

    def load(self):
        audio_file_mfcc = AudioFileMFCC(self.AUDIO_FILE)
        text_file = TextFile(self.TEXT_FILE, file_format=TextFileFormat.PLAIN)
        text_file.set_language(Language.EN)
        return SD(audio_file_mfcc, text_file)

    def test_create_sd(self):
        sd = self.load()

    def test_detect_interval(self):
        begin, end = self.load().detect_interval()
        self.assertNotEqual(begin, 0.0)
        self.assertNotEqual(end, 0.0)

    def test_detect_head(self):
        head = self.load().detect_head()
        self.assertNotEqual(head, 0.0)

    def test_detect_head_min_max(self):
        head = self.load().detect_head(min_head_length=2.0, max_head_length=10.0)
        self.assertNotEqual(head, 0.0)
        self.assertGreaterEqual(head, 2.0)
        self.assertLessEqual(head, 10.0)

    def test_detect_tail(self):
        tail = self.load().detect_tail()
        self.assertNotEqual(tail, 0.0)

    def test_detect_tail_min_max(self):
        tail = self.load().detect_tail(min_tail_length=2.0, max_tail_length=10.0)
        self.assertNotEqual(tail, 0.0)
        self.assertGreaterEqual(tail, 2.0)
        self.assertLessEqual(tail, 10.0)

    def test_detect_bad(self):
        sd = self.load()
        with self.assertRaises(TypeError):
            begin, end = sd.detect_interval(min_head_length="foo")
        with self.assertRaises(ValueError):
            begin, end = sd.detect_interval(min_head_length=-10.0)
        with self.assertRaises(TypeError):
            begin, end = sd.detect_interval(max_head_length="foo")
        with self.assertRaises(ValueError):
            begin, end = sd.detect_interval(max_head_length=-10.0)
        with self.assertRaises(TypeError):
            begin, end = sd.detect_interval(min_tail_length="foo")
        with self.assertRaises(ValueError):
            begin, end = sd.detect_interval(min_tail_length=-10.0)
        with self.assertRaises(TypeError):
            begin, end = sd.detect_interval(max_tail_length="foo")
        with self.assertRaises(ValueError):
            begin, end = sd.detect_interval(max_tail_length=-10.0)



if __name__ == '__main__':
    unittest.main()



