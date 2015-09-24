#!/usr/bin/env python
# coding=utf-8

import tempfile
import unittest

from . import delete_file

from aeneas.espeakwrapper import ESPEAKWrapper
from aeneas.language import Language

class TestESPEAKWrapper(unittest.TestCase):

    def synthesize(self, text, language, zero_length=False):
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        espeak = ESPEAKWrapper()
        result = espeak.synthesize(text, language, output_file_path)
        if zero_length:
            self.assertEqual(result, 0)
        else:
            self.assertGreater(result, 0)
        delete_file(handler, output_file_path)

    def test_str_ascii(self):
        self.synthesize("Word", Language.EN)

    def test_str_unicode(self):
        with self.assertRaises(UnicodeDecodeError):
            self.synthesize("Ausführliche", Language.DE)

    def test_unicode_ascii(self):
        self.synthesize(u"Word", Language.EN)

    def test_unicode_unicode(self):
        self.synthesize(u"Ausführliche", Language.DE)

    def test_none(self):
        self.synthesize(None, Language.IT, zero_length=True)

    def test_empty(self):
        self.synthesize("", Language.IT, zero_length=True)

    def test_replace_language(self):
        self.synthesize(u"Временами Сашке хотелось перестать делать то", Language.UK)

    def test_invalid_language(self):
        # "zzzz" is not a valid espeak voice
        self.synthesize(u"Word", "zzzz", zero_length=True)

    def test_unsupported_language(self):
        # "en-gb" is valid for espeak, but not listed in language.py
        # NOTE disabling this check to allow testing new languages
        #self.synthesize(u"Word", "en-gb", zero_length=True)
        self.synthesize(u"Word", "en-gb")

if __name__ == '__main__':
    unittest.main()



