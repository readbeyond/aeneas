#!/usr/bin/env python
# coding=utf-8

import os
import sys
import tempfile
import unittest

from . import get_abs_path
from aeneas.globalconstants import PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX
from aeneas.globalconstants import PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX
from aeneas.globalconstants import PPN_JOB_IS_TEXT_UNPARSED_ID_SORT
from aeneas.idsortingalgorithm import IDSortingAlgorithm
from aeneas.language import Language
from aeneas.textfile import TextFile, TextFileFormat

class TestTextFile(unittest.TestCase):

    def test_constructor(self):
        tfl = TextFile()
        self.assertEqual(len(tfl), 0)

    def test_read_subtitles_1(self):
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_subtitles_no_end_newline.txt"), TextFileFormat.SUBTITLES)
        self.assertEqual(len(tfl), 15)

    def test_read_subtitles_2(self):
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_subtitles_with_end_newline.txt"), TextFileFormat.SUBTITLES)
        self.assertEqual(len(tfl), 15)

    def test_read_subtitles_3(self):
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_subtitles_multiple_blank.txt"), TextFileFormat.SUBTITLES)
        self.assertEqual(len(tfl), 15)

    def test_read_subtitles_4(self):
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_subtitles_multiple_rows.txt"), TextFileFormat.SUBTITLES)
        self.assertEqual(len(tfl), 15)

    def test_read_subtitles_5(self):
        tfl = TextFile(get_abs_path("res/inputtext/empty.txt"), TextFileFormat.SUBTITLES)
        self.assertEqual(len(tfl), 0)

    def test_read_subtitles_6(self):
        tfl = TextFile(get_abs_path("res/inputtext/blank.txt"), TextFileFormat.SUBTITLES)
        self.assertEqual(len(tfl), 0)

    def test_read_plain_1(self):
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_plain.txt"), TextFileFormat.PLAIN)
        self.assertEqual(len(tfl), 15)

    def test_read_plain_2(self):
        tfl = TextFile(get_abs_path("res/inputtext/empty.txt"), TextFileFormat.PLAIN)
        self.assertEqual(len(tfl), 0)

    def test_read_plain_3(self):
        tfl = TextFile(get_abs_path("res/inputtext/blank.txt"), TextFileFormat.PLAIN)
        self.assertEqual(len(tfl), 5)

    def test_read_parsed_1(self):
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_parsed.txt"), TextFileFormat.PARSED)
        self.assertEqual(len(tfl), 15)

    def test_read_parsed_2(self):
        tfl = TextFile(get_abs_path("res/inputtext/empty.txt"), TextFileFormat.PARSED)
        self.assertEqual(len(tfl), 0)

    def test_read_parsed_3(self):
        tfl = TextFile(get_abs_path("res/inputtext/blank.txt"), TextFileFormat.PARSED)
        self.assertEqual(len(tfl), 0)

    def test_read_parsed_4(self):
        tfl = TextFile(get_abs_path("res/inputtext/badly_parsed_1.txt"), TextFileFormat.PARSED)
        self.assertEqual(len(tfl), 0)

    def test_read_parsed_5(self):
        tfl = TextFile(get_abs_path("res/inputtext/badly_parsed_2.txt"), TextFileFormat.PARSED)
        self.assertEqual(len(tfl), 0)

    def test_read_parsed_6(self):
        tfl = TextFile(get_abs_path("res/inputtext/badly_parsed_3.txt"), TextFileFormat.PARSED)
        self.assertEqual(len(tfl), 0)

    def test_read_parsed_unicode(self):
        tfl = TextFile(get_abs_path("res/inputtext/de_utf8.txt"), TextFileFormat.PARSED)
        self.assertEqual(len(tfl), 24)

    def test_read_unparsed_empty(self):
        parameters = {}
        parameters[PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = "f[0-9]*"
        tfl = TextFile(get_abs_path("res/inputtext/empty.txt"), TextFileFormat.UNPARSED, parameters)
        self.assertEqual(len(tfl), 0)

    def test_read_unparsed_blank(self):
        parameters = {}
        parameters[PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = "f[0-9]*"
        tfl = TextFile(get_abs_path("res/inputtext/blank.txt"), TextFileFormat.UNPARSED, parameters)
        self.assertEqual(len(tfl), 0)

    def test_read_unparsed_soup_1(self):
        parameters = {}
        parameters[PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = "f[0-9]*"
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_unparsed_soup_1.txt"), TextFileFormat.UNPARSED, parameters)
        self.assertEqual(len(tfl), 15)

    def test_read_unparsed_soup_2(self):
        parameters = {}
        parameters[PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = "f[0-9]*"
        parameters[PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX] = "ra"
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_unparsed_soup_2.txt"), TextFileFormat.UNPARSED, parameters)
        self.assertEqual(len(tfl), 15)

    def test_read_unparsed_soup_3(self):
        parameters = {}
        parameters[PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX] = "ra"
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_unparsed_soup_3.txt"), TextFileFormat.UNPARSED, parameters)
        self.assertEqual(len(tfl), 15)

    def test_read_unparsed_xhtml(self):
        parameters = {}
        parameters[PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = "f[0-9]*"
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_unparsed.xhtml"), TextFileFormat.UNPARSED, parameters)
        self.assertEqual(len(tfl), 15)

    def test_read_unparsed_order_1(self):
        parameters = {}
        parameters[PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = "f[0-9]*"
        parameters[PPN_JOB_IS_TEXT_UNPARSED_ID_SORT] = IDSortingAlgorithm.UNSORTED
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_unparsed_order_1.txt"), TextFileFormat.UNPARSED, parameters)
        self.assertEqual(len(tfl), 5)
        self.assertEqual(tfl.fragments[0].identifier, "f001")
        self.assertEqual(tfl.fragments[1].identifier, "f003")
        self.assertEqual(tfl.fragments[2].identifier, "f005")
        self.assertEqual(tfl.fragments[3].identifier, "f004")
        self.assertEqual(tfl.fragments[4].identifier, "f002")

    def test_read_unparsed_order_2(self):
        parameters = {}
        parameters[PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = "f[0-9]*"
        parameters[PPN_JOB_IS_TEXT_UNPARSED_ID_SORT] = IDSortingAlgorithm.NUMERIC
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_unparsed_order_2.txt"), TextFileFormat.UNPARSED, parameters)
        self.assertEqual(len(tfl), 5)
        self.assertEqual(tfl.fragments[0].identifier, "f001")
        self.assertEqual(tfl.fragments[1].identifier, "f2")
        self.assertEqual(tfl.fragments[2].identifier, "f003")
        self.assertEqual(tfl.fragments[3].identifier, "f4")
        self.assertEqual(tfl.fragments[4].identifier, "f050")

    def test_read_unparsed_order_3(self):
        parameters = {}
        parameters[PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = "f[0-9]*"
        parameters[PPN_JOB_IS_TEXT_UNPARSED_ID_SORT] = IDSortingAlgorithm.NUMERIC
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_unparsed_order_3.txt"), TextFileFormat.UNPARSED, parameters)
        self.assertEqual(len(tfl), 5)
        self.assertEqual(tfl.fragments[0].identifier, "f001")
        self.assertEqual(tfl.fragments[1].identifier, "f2")
        self.assertEqual(tfl.fragments[2].identifier, "f003")
        self.assertEqual(tfl.fragments[3].identifier, "f4")
        self.assertEqual(tfl.fragments[4].identifier, "f050")

    def test_read_unparsed_order_4(self):
        parameters = {}
        parameters[PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = "[a-z][0-9]*"
        parameters[PPN_JOB_IS_TEXT_UNPARSED_ID_SORT] = IDSortingAlgorithm.LEXICOGRAPHIC
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_unparsed_order_4.txt"), TextFileFormat.UNPARSED, parameters)
        self.assertEqual(len(tfl), 5)
        self.assertEqual(tfl.fragments[0].identifier, "a005")
        self.assertEqual(tfl.fragments[1].identifier, "b002")
        self.assertEqual(tfl.fragments[2].identifier, "c004")
        self.assertEqual(tfl.fragments[3].identifier, "d001")
        self.assertEqual(tfl.fragments[4].identifier, "e003")

    def test_read_unparsed_order_5(self):
        parameters = {}
        parameters[PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = "[a-z][0-9]*"
        parameters[PPN_JOB_IS_TEXT_UNPARSED_ID_SORT] = IDSortingAlgorithm.NUMERIC
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_unparsed_order_5.txt"), TextFileFormat.UNPARSED, parameters)
        self.assertEqual(len(tfl), 5)
        self.assertEqual(tfl.fragments[0].identifier, "d001")
        self.assertEqual(tfl.fragments[1].identifier, "b002")
        self.assertEqual(tfl.fragments[2].identifier, "e003")
        self.assertEqual(tfl.fragments[3].identifier, "c004")
        self.assertEqual(tfl.fragments[4].identifier, "a005")

    def test_set_language(self):
        tfl = TextFile(get_abs_path("res/inputtext/sonnet_plain.txt"), TextFileFormat.PLAIN)
        self.assertEqual(len(tfl), 15)
        tfl.set_language(Language.EN)
        for fragment in tfl.fragments:
            self.assertEqual(fragment.language, Language.EN)
        tfl.set_language(Language.IT)
        for fragment in tfl.fragments:
            self.assertEqual(fragment.language, Language.IT)

    def test_set_language_on_empty(self):
        tfl = TextFile()
        self.assertEqual(len(tfl), 0)
        tfl.set_language(Language.EN)
        self.assertEqual(len(tfl), 0)

    def test_read_from_list(self):
        tfl = TextFile()
        text_list = [
            "fragment 1",
            "fragment 2",
            "fragment 3",
            "fragment 4",
            "fragment 5"
        ]
        tfl.read_from_list(text_list)
        self.assertEqual(len(tfl), 5)

    def test_read_from_list_with_ids(self):
        tfl = TextFile()
        text_list = [
            ["a1", "fragment 1"],
            ["b2", "fragment 2"],
            ["c3", "fragment 3"],
            ["d4", "fragment 4"],
            ["e5", "fragment 5"]
        ]
        tfl.read_from_list_with_ids(text_list)
        self.assertEqual(len(tfl), 5)

if __name__ == '__main__':
    unittest.main()



