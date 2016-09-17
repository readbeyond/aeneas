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

from aeneas.textfile import TextFile
from aeneas.textfile import TextFragment
from aeneas.ttswrappers.festivalttswrapper import FESTIVALTTSWrapper
from aeneas.runtimeconfiguration import RuntimeConfiguration
import aeneas.globalfunctions as gf


class TestFESTIVALTTSWrapper(unittest.TestCase):

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
            tts_engine = FESTIVALTTSWrapper(rconf=rconf)
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
            tfl.add_fragment(TextFragment(language=language, lines=lines, filtered_lines=lines))
        return tfl

    def test_multiple_tfl_none(self):
        with self.assertRaises(TypeError):
            self.synthesize_multiple(None, zero_length=True)

    def test_multiple_invalid_output_path(self):
        tfl = self.tfl([(FESTIVALTTSWrapper.ENG, [u"word"])])
        with self.assertRaises(OSError):
            self.synthesize_multiple(tfl, ofp="x/y/z/not_existing.wav")

    def test_multiple_no_fragments(self):
        tfl = TextFile()
        tfl.set_language(FESTIVALTTSWrapper.ENG)
        with self.assertRaises(ValueError):
            self.synthesize_multiple(tfl)

    def test_multiple_unicode_ascii(self):
        tfl = self.tfl([(FESTIVALTTSWrapper.ENG, [u"word"])])
        self.synthesize_multiple(tfl)

    def test_multiple_unicode_unicode(self):
        tfl = self.tfl([(FESTIVALTTSWrapper.ENG, [u"Ausführliche"])])
        self.synthesize_multiple(tfl)

    def test_multiple_empty(self):
        tfl = self.tfl([(FESTIVALTTSWrapper.ENG, [u""])])
        with self.assertRaises(ValueError):
            self.synthesize_multiple(tfl)

    def test_multiple_empty_multiline(self):
        tfl = self.tfl([(FESTIVALTTSWrapper.ENG, [u"", u"", u""])])
        with self.assertRaises(ValueError):
            self.synthesize_multiple(tfl)

    def test_multiple_empty_fragments(self):
        tfl = self.tfl([
            (FESTIVALTTSWrapper.ENG, [u""]),
            (FESTIVALTTSWrapper.ENG, [u""]),
            (FESTIVALTTSWrapper.ENG, [u""]),
        ])
        with self.assertRaises(ValueError):
            self.synthesize_multiple(tfl)

    def test_multiple_empty_mixed(self):
        tfl = self.tfl([(FESTIVALTTSWrapper.ENG, [u"Word", u"", u"Word"])])
        self.synthesize_multiple(tfl)

    def test_multiple_empty_mixed_fragments(self):
        tfl = self.tfl([
            (FESTIVALTTSWrapper.ENG, [u"Word"]),
            (FESTIVALTTSWrapper.ENG, [u""]),
            (FESTIVALTTSWrapper.ENG, [u"Word"]),
        ])
        self.synthesize_multiple(tfl)

    def test_multiple_invalid_language(self):
        tfl = self.tfl([("zzzz", [u"Word"])])
        with self.assertRaises(ValueError):
            self.synthesize_multiple(tfl)

    def test_multiple_variation_language(self):
        tfl = self.tfl([(FESTIVALTTSWrapper.ENG_GBR, [u"Word"])])
        self.synthesize_multiple(tfl)


if __name__ == '__main__':
    unittest.main()
