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

    PATH_NOT_WRITEABLE = gf.absolute_path("x/y/z/not_writeable.wav", __file__)

    def perform(self, path, expected, expected2=None, logger=None, quit_after=None, backwards=False):
        def inner(c_ext, cew_subprocess):
            handler, output_file_path = gf.tmp_file(suffix=".wav")
            tfl = TextFile(gf.absolute_path(path, __file__), TextFileFormat.PLAIN)
            tfl.set_language(Language.EN)
            synth = Synthesizer(logger=logger)
            synth.rconf["c_extensions"] = c_ext
            synth.rconf["cew_subprocess_enabled"] = cew_subprocess
            result = synth.synthesize(tfl, output_file_path, quit_after=quit_after, backwards=backwards)
            gf.delete_file(handler, output_file_path)
            self.assertEqual(len(result[0]), expected)
            if expected2 is not None:
                self.assertAlmostEqual(result[1], expected2, places=0)
        for p1 in [True, False]:
            for p2 in [True, False]:
                inner(p1, p2)

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
        with self.assertRaises(OSError):
            synth.synthesize(tfl, self.PATH_NOT_WRITEABLE)

    def test_synthesize(self):
        self.perform("res/inputtext/sonnet_plain.txt", 15)

    def test_synthesize_logger(self):
        logger = Logger()
        self.perform("res/inputtext/sonnet_plain.txt", 15, logger=logger)

    def test_synthesize_unicode(self):
        self.perform("res/inputtext/sonnet_plain_utf8.txt", 15)

    def test_synthesize_quit_after(self):
        self.perform("res/inputtext/sonnet_plain.txt", 6, 12.0, quit_after=10.0) # 11.914 (py) or 12.057 (c)

    def test_synthesize_backwards(self):
        self.perform("res/inputtext/sonnet_plain.txt", 15, backwards=True)

    def test_synthesize_quit_after_backwards(self):
        self.perform("res/inputtext/sonnet_plain.txt", 4, 10.0, quit_after=10.0, backwards=True) # 10.049 (py) or 10.170 (c)



if __name__ == '__main__':
    unittest.main()



