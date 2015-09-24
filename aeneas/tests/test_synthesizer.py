#!/usr/bin/env python
# coding=utf-8

import tempfile
import unittest

from . import get_abs_path, delete_file

from aeneas.language import Language
from aeneas.logger import Logger
from aeneas.synthesizer import Synthesizer
from aeneas.textfile import TextFile, TextFileFormat

class TestSynthesizer(unittest.TestCase):
    
    def perform(self, path, logger=None, quit_after=None, backwards=False):
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        tfl = TextFile(get_abs_path(path), TextFileFormat.PLAIN)
        tfl.set_language(Language.EN)
        synth = Synthesizer(logger=logger)
        result = synth.synthesize(tfl, output_file_path, quit_after=quit_after, backwards=backwards)
        delete_file(handler, output_file_path)
        return result

    def test_synthesize(self):
        result = self.perform("res/inputtext/sonnet_plain.txt")
        self.assertEqual(len(result[0]), 15)

    def test_synthesize_logger(self):
        logger = Logger()
        result = self.perform("res/inputtext/sonnet_plain.txt", logger=logger)
        self.assertEqual(len(result[0]), 15)

    def test_synthesize_unicode(self):
        result = self.perform("res/inputtext/sonnet_plain_utf8.txt")
        self.assertEqual(len(result[0]), 15)

    def test_synthesize_quit_after(self):
        result = self.perform("res/inputtext/sonnet_plain.txt", quit_after=10.0)
        self.assertEqual(len(result[0]), 6)
        self.assertAlmostEqual(result[1], 11.9, places=1) # 11.914

    def test_synthesize_backwards(self):
        result = self.perform("res/inputtext/sonnet_plain.txt", backwards=True)
        self.assertEqual(len(result[0]), 15)

    def test_synthesize_quit_after_backwards(self):
        result = self.perform("res/inputtext/sonnet_plain.txt", quit_after=10.0, backwards=True)
        self.assertEqual(len(result[0]), 4)
        self.assertAlmostEqual(result[1], 10.0, places=1) # 10.049

if __name__ == '__main__':
    unittest.main()



