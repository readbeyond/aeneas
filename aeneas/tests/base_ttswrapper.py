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
import os

from aeneas.textfile import TextFile
from aeneas.textfile import TextFragment
from aeneas.ttswrappers.basettswrapper import BaseTTSWrapper
from aeneas.runtimeconfiguration import RuntimeConfiguration
import aeneas.globalfunctions as gf


class TestBaseTTSWrapper(unittest.TestCase):

    TTS = u""
    TTS_PATH = u""

    TTS_CLASS = BaseTTSWrapper
    TTS_LANGUAGE = u"eng"
    TTS_LANGUAGE_VARIATION = None

    def synthesize(self, text_file, ofp=None, quit_after=None, backwards=False, zero_length=False, expected_exc=None):
        if (self.TTS == u"") or (self.TTS_PATH == u"") or (not os.path.exists(self.TTS_PATH)):
            return

        def inner(c_ext, cew_subprocess, cache):
            if ofp is None:
                handler, output_file_path = gf.tmp_file(suffix=".wav")
            else:
                handler = None
                output_file_path = ofp
            try:
                rconf = RuntimeConfiguration()
                rconf[RuntimeConfiguration.TTS] = self.TTS
                rconf[RuntimeConfiguration.TTS_PATH] = self.TTS_PATH
                rconf[RuntimeConfiguration.C_EXTENSIONS] = c_ext
                rconf[RuntimeConfiguration.CEW_SUBPROCESS_ENABLED] = cew_subprocess
                rconf[RuntimeConfiguration.TTS_CACHE] = cache
                tts_engine = self.TTS_CLASS(rconf=rconf)
                anchors, total_time, num_chars = tts_engine.synthesize_multiple(
                    text_file,
                    output_file_path,
                    quit_after,
                    backwards
                )
                gf.delete_file(handler, output_file_path)
                if cache:
                    tts_engine.clear_cache()
                if zero_length:
                    self.assertEqual(total_time, 0.0)
                else:
                    self.assertGreater(total_time, 0.0)
            except (OSError, TypeError, UnicodeDecodeError, ValueError) as exc:
                gf.delete_file(handler, output_file_path)
                if (cache) and (tts_engine is not None):
                    tts_engine.clear_cache()
                with self.assertRaises(expected_exc):
                    raise exc
        if self.TTS == "espeak":
            for c_ext in [True, False]:
                for cew_subprocess in [True, False]:
                    for cache in [True, False]:
                        inner(c_ext=c_ext, cew_subprocess=cew_subprocess, cache=cache)
        elif self.TTS == "festival":
            for c_ext in [True, False]:
                for cache in [True, False]:
                    inner(c_ext=c_ext, cew_subprocess=False, cache=cache)
        else:
            for cache in [True, False]:
                inner(c_ext=True, cew_subprocess=False, cache=cache)

    def tfl(self, frags):
        tfl = TextFile()
        for language, lines in frags:
            tfl.add_fragment(TextFragment(language=language, lines=lines, filtered_lines=lines))
        return tfl

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            tts_engine = BaseTTSWrapper()

    def test_use_cache(self):
        if self.TTS == u"":
            return
        rconf = RuntimeConfiguration()
        rconf[RuntimeConfiguration.TTS_CACHE] = True
        tts_engine = self.TTS_CLASS(rconf=rconf)
        self.assertTrue(tts_engine.use_cache)
        self.assertIsNotNone(tts_engine.cache)

    def test_clear_cache(self):
        if self.TTS == u"":
            return
        tts_engine = self.TTS_CLASS()
        tts_engine.clear_cache()

    def test_tfl_none(self):
        self.synthesize(None, zero_length=True, expected_exc=TypeError)

    def test_invalid_output_path(self):
        tfl = self.tfl([(self.TTS_LANGUAGE, [u"word"])])
        self.synthesize(tfl, ofp="x/y/z/not_existing.wav", expected_exc=OSError)

    def test_no_fragments(self):
        tfl = TextFile()
        tfl.set_language(self.TTS_LANGUAGE)
        self.synthesize(tfl, expected_exc=ValueError)

    def test_unicode_ascii(self):
        tfl = self.tfl([(self.TTS_LANGUAGE, [u"word"])])
        self.synthesize(tfl)

    def test_unicode_unicode(self):
        tfl = self.tfl([(self.TTS_LANGUAGE, [u"Ausführliche"])])
        self.synthesize(tfl)

    def test_empty(self):
        tfl = self.tfl([(self.TTS_LANGUAGE, [u""])])
        self.synthesize(tfl, expected_exc=ValueError)

    def test_empty_multiline(self):
        tfl = self.tfl([(self.TTS_LANGUAGE, [u"", u"", u""])])
        self.synthesize(tfl, expected_exc=ValueError)

    def test_empty_fragments(self):
        tfl = self.tfl([
            (self.TTS_LANGUAGE, [u""]),
            (self.TTS_LANGUAGE, [u""]),
            (self.TTS_LANGUAGE, [u""]),
        ])
        self.synthesize(tfl, expected_exc=ValueError)

    def test_empty_mixed(self):
        tfl = self.tfl([(self.TTS_LANGUAGE, [u"Word", u"", u"Word"])])
        self.synthesize(tfl)

    def test_empty_mixed_fragments(self):
        tfl = self.tfl([
            (self.TTS_LANGUAGE, [u"Word"]),
            (self.TTS_LANGUAGE, [u""]),
            (self.TTS_LANGUAGE, [u"Word"]),
        ])
        self.synthesize(tfl)

    def test_invalid_language(self):
        tfl = self.tfl([("zzzz", [u"Word"])])
        self.synthesize(tfl, expected_exc=ValueError)

    def test_variation_language(self):
        if self.TTS_LANGUAGE_VARIATION is not None:
            tfl = self.tfl([(self.TTS_LANGUAGE_VARIATION, [u"Word"])])
            self.synthesize(tfl)


if __name__ == "__main__":
    unittest.main()
