#!/usr/bin/env python
# coding=utf-8

import unittest

from aeneas.espeakwrapper import ESPEAKWrapper
from aeneas.language import Language
from aeneas.textfile import TextFile
from aeneas.textfile import TextFragment
import aeneas.globalfunctions as gf

class TestESPEAKWrapper(unittest.TestCase):

    def synthesize_single(self, text, language, ofp=None, zero_length=False):
        if ofp is None:
            handler, output_file_path = gf.tmp_file(suffix=".wav")
        else:
            handler = None
            output_file_path = ofp
        try:
            espeak = ESPEAKWrapper()
            result = espeak.synthesize_single(text, language, output_file_path)
            gf.delete_file(handler, output_file_path)
            if zero_length:
                self.assertEqual(result, 0)
            else:
                self.assertGreater(result, 0)
        except (IOError, TypeError, UnicodeDecodeError) as exc:
            gf.delete_file(handler, output_file_path)
            raise exc

    def synthesize_multiple(self, text_file, ofp=None, quit_after=None, backwards=False, zero_length=False):
        if ofp is None:
            handler, output_file_path = gf.tmp_file(suffix=".wav")
        else:
            handler = None
            output_file_path = ofp
        try:
            espeak = ESPEAKWrapper()
            anchors, total_time, num_chars = espeak.synthesize_multiple(
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
        except (IOError, TypeError, UnicodeDecodeError) as exc:
            gf.delete_file(handler, output_file_path)
            raise exc

    def tfl(self, frags):
        tfl = TextFile()
        for language, lines in frags:
            tfl.append_fragment(TextFragment(language=language, lines=lines, filtered_lines=lines))
        return tfl

    def test_multiple_tfl_none(self):
        with self.assertRaises(TypeError):
            self.synthesize_multiple(None, zero_length=True)

    def test_multiple_invalid_output_path(self):
        tfl = self.tfl([(Language.EN, [u"word"])])
        with self.assertRaises(IOError):
            self.synthesize_multiple(tfl, ofp="x/y/z/not_existing.wav")

    def test_multiple_no_fragments(self):
        tfl = TextFile()
        tfl.set_language(Language.EN)
        self.synthesize_multiple(tfl, zero_length=True)

    def test_multiple_unicode_ascii(self):
        tfl = self.tfl([(Language.EN, [u"word"])])
        self.synthesize_multiple(tfl)

    def test_multiple_unicode_unicode(self):
        tfl = self.tfl([(Language.DE, [u"Ausführliche"])])
        self.synthesize_multiple(tfl)

    def test_multiple_empty(self):
        tfl = self.tfl([(Language.EN, [u""])])
        self.synthesize_multiple(tfl)

    def test_multiple_empty_multiline(self):
        tfl = self.tfl([(Language.EN, [u"", u"", u""])])
        self.synthesize_multiple(tfl)

    def test_multiple_empty_fragments(self):
        tfl = self.tfl([
            (Language.EN, [u""]),
            (Language.EN, [u""]),
            (Language.EN, [u""]),
        ])
        self.synthesize_multiple(tfl)

    def test_multiple_empty_mixed(self):
        tfl = self.tfl([(Language.EN, [u"Word", u"", u"Word"])])
        self.synthesize_multiple(tfl)

    def test_multiple_empty_mixed_fragments(self):
        tfl = self.tfl([
            (Language.EN, [u"Word"]),
            (Language.EN, [u""]),
            (Language.EN, [u"Word"]),
        ])
        self.synthesize_multiple(tfl)

    def test_multiple_replace_language(self):
        tfl = self.tfl([(Language.UK, [u"Временами Сашке хотелось перестать делать то"])])
        self.synthesize_multiple(tfl)

    def test_multiple_replace_language_mixed(self):
        tfl = self.tfl([
            (Language.UK, [u"Word"]),
            (Language.UK, [u"Временами Сашке хотелось перестать делать то"]),
            (Language.UK, [u"Word"])
        ])
        self.synthesize_multiple(tfl)

    def test_multiple_replace_language_mixed_fragments(self):
        tfl = self.tfl([
            (Language.EN, [u"Word"]),
            (Language.UK, [u"Временами Сашке хотелось перестать делать то"]),
            (Language.EN, [u"Word"])
        ])
        self.synthesize_multiple(tfl)

    def test_multiple_invalid_language(self):
        tfl = self.tfl([("zzzz", [u"Word"])])
        with self.assertRaises(ValueError):
            self.synthesize_multiple(tfl)

    def test_multiple_variation_language(self):
        tfl = self.tfl([(Language.EN_GB, [u"Word"])])
        self.synthesize_multiple(tfl)

    def test_single_none(self):
        with self.assertRaises(TypeError):
            self.synthesize_single(None, Language.EN)

    def test_single_invalid_output_path(self):
        with self.assertRaises(IOError):
            self.synthesize_single(u"word", Language.EN, ofp="x/y/z/not_existing.wav")

    def test_single_empty_string(self):
        self.synthesize_single(u"", Language.EN, zero_length=True)

    def test_single_text_str_ascii(self):
        with self.assertRaises(TypeError):
            self.synthesize_single(b"Word", Language.EN)

    def test_single_text_str_unicode(self):
        with self.assertRaises(TypeError):
            self.synthesize_single(b"Ausf\xc3\xbchrliche", Language.DE)

    def test_single_text_unicode_ascii(self):
        self.synthesize_single(u"Word", Language.EN)

    def test_single_text_unicode_unicode(self):
        self.synthesize_single(u"Ausführliche", Language.DE)

    def test_single_variation_language(self):
        self.synthesize_single(u"Word", Language.EN_GB)

    def test_single_replace_language(self):
        self.synthesize_single(u"Временами Сашке хотелось перестать делать то", Language.UK)

    def test_single_invalid_language(self):
        with self.assertRaises(ValueError):
            self.synthesize_single(u"Word", "zzzz")



if __name__ == '__main__':
    unittest.main()



