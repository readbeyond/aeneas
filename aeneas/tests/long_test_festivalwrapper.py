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

from aeneas.festivalwrapper import FESTIVALWrapper
from aeneas.textfile import TextFile
from aeneas.textfile import TextFragment
from aeneas.runtimeconfiguration import RuntimeConfiguration
import aeneas.globalfunctions as gf


class TestFESTIVALWrapper(unittest.TestCase):

    def synthesize_single(self, text, language, ofp=None, zero_length=False):
        if ofp is None:
            handler, output_file_path = gf.tmp_file(suffix=".wav")
        else:
            handler = None
            output_file_path = ofp
        try:
            rconf = RuntimeConfiguration()
            rconf[RuntimeConfiguration.TTS] = u"festival"
            rconf[RuntimeConfiguration.TTS_PATH] = u"text2wave"
            tts_engine = FESTIVALWrapper(rconf=rconf)
            result = tts_engine.synthesize_single(text, language, output_file_path)
            gf.delete_file(handler, output_file_path)
            if zero_length:
                self.assertEqual(result, 0)
            else:
                self.assertGreater(result, 0)
        except (OSError, TypeError, UnicodeDecodeError, ValueError) as exc:
            gf.delete_file(handler, output_file_path)
            raise exc

    def synthesize_multiple(self, text_file, ofp=None, quit_after=None, backwards=False, zero_length=False):
        if ofp is None:
            handler, output_file_path = gf.tmp_file(suffix=".wav")
        else:
            handler = None
            output_file_path = ofp
        try:
            rconf = RuntimeConfiguration()
            rconf[RuntimeConfiguration.TTS] = u"festival"
            rconf[RuntimeConfiguration.TTS_PATH] = u"text2wave"
            tts_engine = FESTIVALWrapper(rconf=rconf)
            anchors, total_time, num_chars = tts_engine.synthesize_multiple(
                text_file,
                output_file_path,
                quit_after,
                backwards
            )
            gf.delete_file(handler, output_file_path)
            if zero_length:
                self.assertEqual(total_time, 0.0)
            else:
                self.assertGreater(total_time, 0.0)
        except (OSError, TypeError, UnicodeDecodeError, ValueError) as exc:
            gf.delete_file(handler, output_file_path)
            raise exc

    def tfl(self, frags):
        tfl = TextFile()
        for language, lines in frags:
            tfl.add_fragment(TextFragment(language=language, lines=lines, filtered_lines=lines), as_last=True)
        return tfl

    def test_multiple_tfl_none(self):
        with self.assertRaises(TypeError):
            self.synthesize_multiple(None, zero_length=True)

    def test_multiple_invalid_output_path(self):
        tfl = self.tfl([(FESTIVALWrapper.ENG, [u"word"])])
        with self.assertRaises(OSError):
            self.synthesize_multiple(tfl, ofp="x/y/z/not_existing.wav")

    def test_multiple_no_fragments(self):
        tfl = TextFile()
        tfl.set_language(FESTIVALWrapper.ENG)
        with self.assertRaises(ValueError):
            self.synthesize_multiple(tfl)

    def test_multiple_unicode_ascii(self):
        tfl = self.tfl([(FESTIVALWrapper.ENG, [u"word"])])
        self.synthesize_multiple(tfl)

    def test_multiple_unicode_unicode(self):
        tfl = self.tfl([(FESTIVALWrapper.ENG, [u"Ausführliche"])])
        self.synthesize_multiple(tfl)

    # TODO disabling this test, festival does not handle empty text
    # COMMENTED def test_multiple_empty(self):
    # COMMENTED   tfl = self.tfl([(FESTIVALWrapper.ENG, [u""])])
    # COMMENTED   self.synthesize_multiple(tfl)

    # TODO disabling this test, festival does not handle empty text
    # COMMENTED def test_multiple_empty_multiline(self):
    # COMMENTED     tfl = self.tfl([(FESTIVALWrapper.ENG, [u"", u"", u""])])
    # COMMENTED     self.synthesize_multiple(tfl)

    # TODO disabling this test, festival does not handle empty text
    # COMMENTED def test_multiple_empty_fragments(self):
    # COMMENTED     tfl = self.tfl([
    # COMMENTED         (FESTIVALWrapper.ENG, [u""]),
    # COMMENTED         (FESTIVALWrapper.ENG, [u""]),
    # COMMENTED         (FESTIVALWrapper.ENG, [u""]),
    # COMMENTED     ])
    # COMMENTED     self.synthesize_multiple(tfl)

    def test_multiple_empty_mixed(self):
        tfl = self.tfl([(FESTIVALWrapper.ENG, [u"Word", u"", u"Word"])])
        self.synthesize_multiple(tfl)

    # TODO disabling this test, festival does not handle empty text
    # COMMENTED def test_multiple_empty_mixed_fragments(self):
    # COMMENTED     tfl = self.tfl([
    # COMMENTED         (FESTIVALWrapper.ENG, [u"Word"]),
    # COMMENTED         (FESTIVALWrapper.ENG, [u""]),
    # COMMENTED         (FESTIVALWrapper.ENG, [u"Word"]),
    # COMMENTED     ])
    # COMMENTED     self.synthesize_multiple(tfl)

    def test_multiple_invalid_language(self):
        tfl = self.tfl([("zzzz", [u"Word"])])
        with self.assertRaises(ValueError):
            self.synthesize_multiple(tfl)

    def test_multiple_variation_language(self):
        tfl = self.tfl([(FESTIVALWrapper.ENG_GBR, [u"Word"])])
        self.synthesize_multiple(tfl)

    def test_single_none(self):
        with self.assertRaises(TypeError):
            self.synthesize_single(None, FESTIVALWrapper.ENG)

    def test_single_invalid_output_path(self):
        with self.assertRaises(OSError):
            self.synthesize_single(u"word", FESTIVALWrapper.ENG, ofp="x/y/z/not_existing.wav")

    def test_single_empty_string(self):
        self.synthesize_single(u"", FESTIVALWrapper.ENG, zero_length=True)

    def test_single_text_str_ascii(self):
        with self.assertRaises(TypeError):
            self.synthesize_single(b"Word", FESTIVALWrapper.ENG)

    def test_single_text_str_unicode(self):
        with self.assertRaises(TypeError):
            self.synthesize_single(b"Ausf\xc3\xbchrliche", FESTIVALWrapper.ENG)

    def test_single_text_unicode_ascii(self):
        self.synthesize_single(u"Word", FESTIVALWrapper.ENG)

    def test_single_text_unicode_unicode(self):
        self.synthesize_single(u"Ausführliche", FESTIVALWrapper.ENG)

    def test_single_variation_language(self):
        self.synthesize_single(u"Word", FESTIVALWrapper.ENG_GBR)

    def test_single_invalid_language(self):
        with self.assertRaises(ValueError):
            self.synthesize_single(u"Word", "zzzz")


if __name__ == '__main__':
    unittest.main()
