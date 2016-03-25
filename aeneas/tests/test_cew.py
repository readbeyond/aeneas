#!/usr/bin/env python
# coding=utf-8

import unittest

import aeneas.globalfunctions as gf

class TestCEW(unittest.TestCase):

    def test_cew_synthesize_single(self):
        handler, output_file_path = gf.tmp_file(suffix=".wav")
        try:
            import aeneas.cew.cew
            sr, begin, end = aeneas.cew.cew.synthesize_single(
                output_file_path,
                u"en",                      # NOTE cew requires the actual eSpeak voice code
                u"Dummy"
            )
            self.assertEqual(sr, 22050)
            self.assertEqual(begin, 0)
            self.assertGreater(end, 0)
        except ImportError:
            pass
        gf.delete_file(handler, output_file_path)

    def test_cew_synthesize_multiple(self):
        handler, output_file_path = gf.tmp_file(suffix=".wav")
        try:
            c_quit_after = 0.0
            c_backwards = 0
            c_text = [
                (u"en", u"Dummy 1"),        # NOTE cew requires the actual eSpeak voice code
                (u"en", u"Dummy 2"),        # NOTE cew requires the actual eSpeak voice code
                (u"en", u"Dummy 3"),        # NOTE cew requires the actual eSpeak voice code
            ]
            import aeneas.cew.cew
            sr, sf, intervals = aeneas.cew.cew.synthesize_multiple(
                output_file_path,
                c_quit_after,
                c_backwards,
                c_text
            )
            self.assertEqual(sr, 22050)
            self.assertEqual(sf, 3)
            self.assertEqual(len(intervals), 3)
        except ImportError:
            pass
        gf.delete_file(handler, output_file_path)

    def test_cew_synthesize_multiple_lang(self):
        handler, output_file_path = gf.tmp_file(suffix=".wav")
        try:
            c_quit_after = 0.0
            c_backwards = 0
            c_text = [
                (u"en", u"Dummy 1"),        # NOTE cew requires the actual eSpeak voice code
                (u"it", u"Segnaposto 2"),   # NOTE cew requires the actual eSpeak voice code
                (u"en", u"Dummy 3"),        # NOTE cew requires the actual eSpeak voice code
            ]
            import aeneas.cew.cew
            sr, sf, intervals = aeneas.cew.cew.synthesize_multiple(
                output_file_path,
                c_quit_after,
                c_backwards,
                c_text
            )
            self.assertEqual(sr, 22050)
            self.assertEqual(sf, 3)
            self.assertEqual(len(intervals), 3)
        except ImportError:
            pass
        gf.delete_file(handler, output_file_path)

if __name__ == '__main__':
    unittest.main()



