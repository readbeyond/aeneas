#!/usr/bin/env python
# coding=utf-8

import os
import sys
import tempfile
import unittest

from aeneas.espeakwrapper import ESPEAKWrapper
from aeneas.language import Language

class TestESPEAKWrapper(unittest.TestCase):

    def test_synthesize(self):
        text = u"Nel mezzo del cammin di nostra vita"
        language = Language.IT
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        espeak = ESPEAKWrapper()
        result = espeak.synthesize(text, language, output_file_path)
        self.assertGreater(result, 0)
        os.close(handler)
        os.remove(output_file_path)

    def test_synthesize_unicode(self):
        text = u"Ausführliche"
        language = Language.DE
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        espeak = ESPEAKWrapper()
        result = espeak.synthesize(text, language, output_file_path)
        self.assertGreater(result, 0)
        os.close(handler)
        os.remove(output_file_path)

    def test_none_text(self):
        text = None
        language = Language.IT
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        espeak = ESPEAKWrapper()
        result = espeak.synthesize(text, language, output_file_path)
        self.assertEqual(result, 0)
        os.close(handler)
        os.remove(output_file_path)

    def test_empty_text(self):
        text = ""
        language = Language.IT
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        espeak = ESPEAKWrapper()
        result = espeak.synthesize(text, language, output_file_path)
        self.assertEqual(result, 0)
        os.close(handler)
        os.remove(output_file_path)

    def test_replace_language(self):
        text = u"Временами Сашке хотелось перестать делать то"
        language = Language.UK
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        espeak = ESPEAKWrapper()
        result = espeak.synthesize(text, language, output_file_path)
        self.assertGreater(result, 0)
        os.close(handler)
        os.remove(output_file_path)

if __name__ == '__main__':
    unittest.main()



