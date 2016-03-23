#!/usr/bin/env python
# coding=utf-8

import unittest

from aeneas.espeakwrapper import ESPEAKWrapper
from aeneas.textfile import TextFile
from aeneas.textfile import TextFragment
from aeneas.runtimeconfiguration import RuntimeConfiguration
import aeneas.globalfunctions as gf

class TestESPEAKWrapper(unittest.TestCase):

    def synthesize_single(self, text, language, ofp=None, zero_length=False):
        def inner(c_ext, cew_subprocess):
            if ofp is None:
                handler, output_file_path = gf.tmp_file(suffix=".wav")
            else:
                handler = None
                output_file_path = ofp
            try:
                rconf = RuntimeConfiguration()
                rconf[RuntimeConfiguration.C_EXTENSIONS] = c_ext
                rconf[RuntimeConfiguration.CEW_SUBPROCESS_ENABLED] = cew_subprocess
                tts_engine = ESPEAKWrapper(rconf=rconf)
                result = tts_engine.synthesize_single(text, language, output_file_path)
                gf.delete_file(handler, output_file_path)
                if zero_length:
                    self.assertEqual(result, 0)
                else:
                    self.assertGreater(result, 0)
            except (OSError, TypeError, UnicodeDecodeError, ValueError) as exc:
                gf.delete_file(handler, output_file_path)
                raise exc
        for p1 in [True, False]:
            for p2 in [True, False]:
                inner(p1, p2)

    def synthesize_multiple(self, text_file, ofp=None, quit_after=None, backwards=False, zero_length=False):
        def inner(c_ext, cew_subprocess):
            if ofp is None:
                handler, output_file_path = gf.tmp_file(suffix=".wav")
            else:
                handler = None
                output_file_path = ofp
            try:
                rconf = RuntimeConfiguration()
                rconf[RuntimeConfiguration.C_EXTENSIONS] = c_ext
                rconf[RuntimeConfiguration.CEW_SUBPROCESS_ENABLED] = cew_subprocess
                tts_engine = ESPEAKWrapper(rconf=rconf)
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
        for a in [True, False]:
            for b in [True, False]:
                inner(a, b)

    def tfl(self, frags):
        tfl = TextFile()
        for language, lines in frags:
            tfl.add_fragment(TextFragment(language=language, lines=lines, filtered_lines=lines))
        return tfl

    def test_multiple_tfl_none(self):
        with self.assertRaises(TypeError):
            self.synthesize_multiple(None, zero_length=True)

    def test_multiple_invalid_output_path(self):
        tfl = self.tfl([(ESPEAKWrapper.ENG, [u"word"])])
        with self.assertRaises(OSError):
            self.synthesize_multiple(tfl, ofp="x/y/z/not_existing.wav")

    def test_multiple_no_fragments(self):
        tfl = TextFile()
        tfl.set_language(ESPEAKWrapper.ENG)
        with self.assertRaises(ValueError):
            self.synthesize_multiple(tfl)

    def test_multiple_unicode_ascii(self):
        tfl = self.tfl([(ESPEAKWrapper.ENG, [u"word"])])
        self.synthesize_multiple(tfl)

    def test_multiple_unicode_unicode(self):
        tfl = self.tfl([(ESPEAKWrapper.DEU, [u"Ausführliche"])])
        self.synthesize_multiple(tfl)

    def test_multiple_empty(self):
        tfl = self.tfl([(ESPEAKWrapper.ENG, [u""])])
        self.synthesize_multiple(tfl)

    def test_multiple_empty_multiline(self):
        tfl = self.tfl([(ESPEAKWrapper.ENG, [u"", u"", u""])])
        self.synthesize_multiple(tfl)

    def test_multiple_empty_fragments(self):
        tfl = self.tfl([
            (ESPEAKWrapper.ENG, [u""]),
            (ESPEAKWrapper.ENG, [u""]),
            (ESPEAKWrapper.ENG, [u""]),
        ])
        self.synthesize_multiple(tfl)

    def test_multiple_empty_mixed(self):
        tfl = self.tfl([(ESPEAKWrapper.ENG, [u"Word", u"", u"Word"])])
        self.synthesize_multiple(tfl)

    def test_multiple_empty_mixed_fragments(self):
        tfl = self.tfl([
            (ESPEAKWrapper.ENG, [u"Word"]),
            (ESPEAKWrapper.ENG, [u""]),
            (ESPEAKWrapper.ENG, [u"Word"]),
        ])
        self.synthesize_multiple(tfl)

    def test_multiple_replace_language(self):
        tfl = self.tfl([(ESPEAKWrapper.UKR, [u"Временами Сашке хотелось перестать делать то"])])
        self.synthesize_multiple(tfl)

    def test_multiple_replace_language_mixed(self):
        tfl = self.tfl([
            (ESPEAKWrapper.UKR, [u"Word"]),
            (ESPEAKWrapper.UKR, [u"Временами Сашке хотелось перестать делать то"]),
            (ESPEAKWrapper.UKR, [u"Word"])
        ])
        self.synthesize_multiple(tfl)

    def test_multiple_replace_language_mixed_fragments(self):
        tfl = self.tfl([
            (ESPEAKWrapper.ENG, [u"Word"]),
            (ESPEAKWrapper.UKR, [u"Временами Сашке хотелось перестать делать то"]),
            (ESPEAKWrapper.ENG, [u"Word"])
        ])
        self.synthesize_multiple(tfl)

    def test_multiple_invalid_language(self):
        tfl = self.tfl([("zzzz", [u"Word"])])
        with self.assertRaises(ValueError):
            self.synthesize_multiple(tfl)

    def test_multiple_variation_language(self):
        tfl = self.tfl([(ESPEAKWrapper.ENG_GBR, [u"Word"])])
        self.synthesize_multiple(tfl)

    def test_single_none(self):
        with self.assertRaises(TypeError):
            self.synthesize_single(None, ESPEAKWrapper.ENG)

    def test_single_invalid_output_path(self):
        with self.assertRaises(OSError):
            self.synthesize_single(u"word", ESPEAKWrapper.ENG, ofp="x/y/z/not_existing.wav")

    def test_single_empty_string(self):
        self.synthesize_single(u"", ESPEAKWrapper.ENG, zero_length=True)

    def test_single_text_str_ascii(self):
        with self.assertRaises(TypeError):
            self.synthesize_single(b"Word", ESPEAKWrapper.ENG)

    def test_single_text_str_unicode(self):
        with self.assertRaises(TypeError):
            self.synthesize_single(b"Ausf\xc3\xbchrliche", ESPEAKWrapper.DEU)

    def test_single_text_unicode_ascii(self):
        self.synthesize_single(u"Word", ESPEAKWrapper.ENG)

    def test_single_text_unicode_unicode(self):
        self.synthesize_single(u"Ausführliche", ESPEAKWrapper.DEU)

    def test_single_variation_language(self):
        self.synthesize_single(u"Word", ESPEAKWrapper.ENG_GBR)

    def test_single_replace_language(self):
        self.synthesize_single(u"Временами Сашке хотелось перестать делать то", ESPEAKWrapper.UKR)

    def test_single_invalid_language(self):
        with self.assertRaises(ValueError):
            self.synthesize_single(u"Word", "zzzz")



if __name__ == '__main__':
    unittest.main()



