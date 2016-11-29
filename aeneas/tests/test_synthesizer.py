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

import unittest

from aeneas.exacttiming import TimeValue
from aeneas.language import Language
from aeneas.logger import Logger
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.synthesizer import Synthesizer
from aeneas.textfile import TextFile
from aeneas.textfile import TextFileFormat
import aeneas.globalfunctions as gf


class TestSynthesizer(unittest.TestCase):

    PATH_NOT_WRITEABLE = gf.absolute_path("x/y/z/not_writeable.wav", __file__)

    def perform(self, path, expected, expected2=None, logger=None, quit_after=None, backwards=False):
        def inner(c_ext, cew_subprocess, tts_cache):
            handler, output_file_path = gf.tmp_file(suffix=".wav")
            tfl = TextFile(gf.absolute_path(path, __file__), TextFileFormat.PLAIN)
            tfl.set_language(Language.ENG)
            synth = Synthesizer(logger=logger)
            synth.rconf[RuntimeConfiguration.C_EXTENSIONS] = c_ext
            synth.rconf[RuntimeConfiguration.CEW_SUBPROCESS_ENABLED] = cew_subprocess
            synth.rconf[RuntimeConfiguration.TTS_CACHE] = tts_cache
            result = synth.synthesize(tfl, output_file_path, quit_after=quit_after, backwards=backwards)
            gf.delete_file(handler, output_file_path)
            self.assertEqual(len(result[0]), expected)
            if expected2 is not None:
                self.assertAlmostEqual(result[1], expected2, places=0)
        for c_ext in [True, False]:
            for cew_subprocess in [True, False]:
                for tts_cache in [True, False]:
                    inner(c_ext, cew_subprocess, tts_cache)

    def test_clear_cache(self):
        synth = Synthesizer()
        synth.clear_cache()

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
        self.perform("res/inputtext/sonnet_plain.txt", 6, TimeValue("12.000"), quit_after=TimeValue("10.000"))

    def test_synthesize_backwards(self):
        self.perform("res/inputtext/sonnet_plain.txt", 15, backwards=True)

    def test_synthesize_quit_after_backwards(self):
        self.perform("res/inputtext/sonnet_plain.txt", 4, TimeValue("10.000"), quit_after=TimeValue("10.000"), backwards=True)

    def test_synthesize_plain_with_empty_lines(self):
        self.perform("res/inputtext/plain_with_empty_lines.txt", 19)


if __name__ == "__main__":
    unittest.main()
