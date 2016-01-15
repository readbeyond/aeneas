#!/usr/bin/env python
# coding=utf-8

import os
import unittest

from aeneas.audiofile import AudioFileMonoWAVE
from aeneas.sd import SD
from aeneas.sd import SDMetric
from aeneas.textfile import TextFile
from aeneas.textfile import TextFileFormat
import aeneas.globalfunctions as gf

class TestSD(unittest.TestCase):

    AUDIO_FILE = gf.absolute_path("res/cmfcc/audio.wav", __file__)
    TEXT_FILE = gf.absolute_path("res/inputtext/sonnet_plain.txt", __file__)

    def load(self):
        audio_file = AudioFileMonoWAVE(file_path=self.AUDIO_FILE)
        text_file = TextFile(file_path=self.TEXT_FILE, file_format=TextFileFormat.PLAIN)
        return SD(audio_file, text_file)

    def test_create_sd(self):
        sd = self.load()

    def test_detect_interval(self):
        sd = self.load()
        begin, end = sd.detect_interval()

    def test_detect_interval_head_min(self):
        sd = self.load()
        begin, end = sd.detect_interval(min_head_length=0.0)

    def test_detect_interval_head_max(self):
        sd = self.load()
        begin, end = sd.detect_interval(max_head_length=10.0)

    def test_detect_interval_head_min_max(self):
        sd = self.load()
        begin, end = sd.detect_interval(min_head_length=0.0, max_head_length=10.0)

    def test_detect_interval_tail_min(self):
        sd = self.load()
        begin, end = sd.detect_interval(min_tail_length=0.0)

    def test_detect_interval_tail_max(self):
        sd = self.load()
        begin, end = sd.detect_interval(max_tail_length=10.0)

    def test_detect_interval_tail_min_max(self):
        sd = self.load()
        begin, end = sd.detect_interval(min_tail_length=0.0, max_tail_length=10.0)

    def test_detect_interval_head_tail(self):
        sd = self.load()
        begin, end = sd.detect_interval(min_head_length=0.0, max_head_length=10.0, min_tail_length=0.0, max_tail_length=10.0)

    def test_detect_interval_metric_value(self):
        sd = self.load()
        begin, end = sd.detect_interval(metric=SDMetric.VALUE)

    def test_detect_interval_metric_distortion(self):
        sd = self.load()
        begin, end = sd.detect_interval(metric=SDMetric.DISTORTION)

    def test_detect_interval_metric_bad(self):
        sd = self.load()
        begin, end = sd.detect_interval(metric="foo")

    def test_detect_head(self):
        sd = self.load()
        begin = sd.detect_head()

    def test_detect_head_min(self):
        sd = self.load()
        begin = sd.detect_head(min_head_length=0.0)

    def test_detect_head_min_bad_1(self):
        sd = self.load()
        begin = sd.detect_head(min_head_length=-10.0)

    def test_detect_head_min_bad_2(self):
        sd = self.load()
        begin = sd.detect_head(min_head_length=1000.0)

    def test_detect_head_min_bad_3(self):
        sd = self.load()
        begin = sd.detect_head(min_head_length="foo")

    def test_detect_head_max(self):
        sd = self.load()
        begin = sd.detect_head(max_head_length=10.0)

    def test_detect_head_max_bad_1(self):
        sd = self.load()
        begin = sd.detect_head(max_head_length=-10.0)

    def test_detect_head_max_bad_2(self):
        sd = self.load()
        begin = sd.detect_head(max_head_length=1000.0)

    def test_detect_head_max_bad_3(self):
        sd = self.load()
        begin = sd.detect_head(max_head_length="foo")

    def test_detect_head_metric_value(self):
        sd = self.load()
        begin = sd.detect_head(metric=SDMetric.VALUE)

    def test_detect_head_metric_distortion(self):
        sd = self.load()
        begin = sd.detect_head(metric=SDMetric.DISTORTION)

    def test_detect_head_metric_bad(self):
        sd = self.load()
        begin = sd.detect_head(metric="foo")

    def test_detect_tail(self):
        sd = self.load()
        tail = sd.detect_tail()

    def test_detect_tail_min(self):
        sd = self.load()
        begin = sd.detect_tail(min_tail_length=0.0)

    def test_detect_tail_min_bad_1(self):
        sd = self.load()
        begin = sd.detect_tail(min_tail_length=-10.0)

    def test_detect_tail_min_bad_2(self):
        sd = self.load()
        begin = sd.detect_tail(min_tail_length=1000.0)

    def test_detect_tail_min_bad_3(self):
        sd = self.load()
        begin = sd.detect_tail(min_tail_length="foo")

    def test_detect_tail_max(self):
        sd = self.load()
        begin = sd.detect_tail(max_tail_length=10.0)

    def test_detect_tail_max_bad_1(self):
        sd = self.load()
        begin = sd.detect_tail(max_tail_length=-10.0)

    def test_detect_tail_max_bad_2(self):
        sd = self.load()
        begin = sd.detect_tail(max_tail_length=1000.0)

    def test_detect_tail_max_bad_3(self):
        sd = self.load()
        begin = sd.detect_tail(max_tail_length="foo")

    def test_detect_tail_metric_value(self):
        sd = self.load()
        begin = sd.detect_tail(metric=SDMetric.VALUE)

    def test_detect_tail_metric_distortion(self):
        sd = self.load()
        begin = sd.detect_tail(metric=SDMetric.DISTORTION)

    def test_detect_tail_metric_bad(self):
        sd = self.load()
        begin = sd.detect_tail(metric="foo")

    # TODO add more meaningful tests about the actual detection of head/tail



if __name__ == '__main__':
    unittest.main()



