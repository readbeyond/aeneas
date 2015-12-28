#!/usr/bin/env python
# coding=utf-8

import unittest

from aeneas.language import Language
from aeneas.logger import Logger
from aeneas.synthesizer import Synthesizer
from aeneas.textfile import TextFile
from aeneas.textfile import TextFileFormat
import aeneas.globalfunctions as gf

class TestSynthesizer(unittest.TestCase):

    PATH_NOT_WRITEABLE = gf.get_abs_path("x/y/z/not_writeable.wav", __file__)

    def perform(self, path, logger=None, quit_after=None, backwards=False):
        handler, output_file_path = gf.tmp_file(suffix=".wav")
        tfl = TextFile(gf.get_abs_path(path, __file__), TextFileFormat.PLAIN)
        tfl.set_language(Language.EN)
        synth = Synthesizer(logger=logger)
        result = synth.synthesize(tfl, output_file_path, quit_after=quit_after, backwards=backwards)
        gf.delete_file(handler, output_file_path)
        return result

    def test_synthesize_none(self):
        synth = Synthesizer()
        with self.assertRaises(TypeError):
            synth.synthesize(None, self.PATH_NOT_WRITEABLE)

    def test_synthesize_invalid_text_file(self):
        synth = Synthesizer()
        with self.assertRaises(TypeError):
            synth.synthesize("foo", self.PATH_NOT_WRITEABLE)

    def test_synthesize_path_not_writeable(self):
        tfl = TextFile()
        synth = Synthesizer()
        with self.assertRaises(IOError):
            synth.synthesize(tfl, self.PATH_NOT_WRITEABLE)

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
        self.assertAlmostEqual(result[1], 12, places=0) # 11.914 (py) or 12.057 (c)

    def test_synthesize_backwards(self):
        result = self.perform("res/inputtext/sonnet_plain.txt", backwards=True)
        self.assertEqual(len(result[0]), 15)

    def test_synthesize_quit_after_backwards(self):
        result = self.perform("res/inputtext/sonnet_plain.txt", quit_after=10.0, backwards=True)
        self.assertEqual(len(result[0]), 4)
        self.assertAlmostEqual(result[1], 10.0, places=0) # 10.049 (py) or 10.170 (c)

if __name__ == '__main__':
    unittest.main()



