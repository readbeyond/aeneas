#!/usr/bin/env python
# coding=utf-8

import os
import sys
import tempfile
import unittest

from . import get_abs_path

from aeneas.language import Language
from aeneas.logger import Logger
from aeneas.synthesizer import Synthesizer
from aeneas.textfile import TextFile, TextFileFormat

class TestSynthesizer(unittest.TestCase):

    def test_synthesize(self):
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_plain.txt"), TextFileFormat.PLAIN)
        tfl.set_language(Language.EN)
        synth = Synthesizer()
        anchors = synth.synthesize(tfl, output_file_path)
        self.assertGreater(len(anchors), 0)
        os.remove(output_file_path)

    def test_synthesize_with_logger(self):
        logger = Logger()
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_plain.txt"), TextFileFormat.PLAIN, logger=logger)
        tfl.set_language(Language.EN)
        synth = Synthesizer(logger=logger)
        anchors = synth.synthesize(tfl, output_file_path)
        self.assertGreater(len(anchors), 0)
        os.remove(output_file_path)

    def test_synthesize_with_unicode(self):
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        tfl = TextFile(get_abs_path("res/inputtext/de_utf8.txt"), TextFileFormat.PARSED)
        tfl.set_language(Language.DE)
        synth = Synthesizer()
        anchors = synth.synthesize(tfl, output_file_path)
        self.assertGreater(len(anchors), 0)
        os.remove(output_file_path)

if __name__ == '__main__':
    unittest.main()



