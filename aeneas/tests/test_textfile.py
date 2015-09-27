#!/usr/bin/env python
# coding=utf-8

import unittest

from . import get_abs_path

import aeneas.globalconstants as gc
from aeneas.idsortingalgorithm import IDSortingAlgorithm
from aeneas.language import Language
from aeneas.textfile import TextFile, TextFileFormat, TextFragment

class TestTextFile(unittest.TestCase):

    EMPTY_FILE_PATH = "res/inputtext/empty.txt"
    BLANK_FILE_PATH = "res/inputtext/blank.txt"
    PLAIN_FILE_PATH = "res/inputtext/sonnet_plain.txt"
    UNPARSED_PARAMETERS = {
        gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX : "f[0-9]*",
        gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX : "ra",
        gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT : IDSortingAlgorithm.UNSORTED,
    }

    def load(self, input_file_path=PLAIN_FILE_PATH, fmt=TextFileFormat.PLAIN, expected_length=15, parameters=None):
        tfl = TextFile(get_abs_path(input_file_path), fmt, parameters)
        self.assertEqual(len(tfl), expected_length)
        return tfl

    def load_and_sort_id(self, input_file_path, id_regex, id_sort, expected):
        parameters = {}
        parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = id_regex
        parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT] = id_sort
        tfl = self.load(input_file_path, TextFileFormat.UNPARSED, 5, parameters)
        i = 0
        for e in expected:
            self.assertEqual(tfl.fragments[i].identifier, e)
            i += 1

    def load_and_slice(self, expected, start=None, end=None):
        tfl = self.load()
        sli = tfl.get_slice(start, end)
        self.assertEqual(len(sli), expected)

    def test_constructor(self):
        tfl = TextFile()
        self.assertEqual(len(tfl), 0)

    def test_read_empty(self):
        for fmt in TextFileFormat.ALLOWED_VALUES:
            self.load(self.EMPTY_FILE_PATH, fmt, 0, self.UNPARSED_PARAMETERS)

    def test_read_blank(self):
        for fmt in TextFileFormat.ALLOWED_VALUES:
            expected = 0
            if fmt == TextFileFormat.PLAIN:
                expected = 5
            self.load(self.BLANK_FILE_PATH, fmt, expected, self.UNPARSED_PARAMETERS)

    def test_read_subtitles(self):
        for path in [
                "res/inputtext/sonnet_subtitles_with_end_newline.txt",
                "res/inputtext/sonnet_subtitles_no_end_newline.txt",
                "res/inputtext/sonnet_subtitles_multiple_blank.txt",
                "res/inputtext/sonnet_subtitles_multiple_rows.txt"
        ]:
            self.load(path, TextFileFormat.SUBTITLES, 15)

    def test_read_plain(self):
        self.load("res/inputtext/sonnet_plain.txt", TextFileFormat.PLAIN, 15)

    def test_read_plain_utf8(self):
        self.load("res/inputtext/sonnet_plain_utf8.txt", TextFileFormat.PLAIN, 15)

    def test_read_parsed(self):
        self.load("res/inputtext/sonnet_parsed.txt", TextFileFormat.PARSED, 15)

    def test_read_parsed_bad(self):
        for path in [
                "res/inputtext/badly_parsed_1.txt",
                "res/inputtext/badly_parsed_2.txt",
                "res/inputtext/badly_parsed_3.txt"
        ]:
            self.load(path, TextFileFormat.PARSED, 0)

    def test_read_unparsed(self):
        for case in [
                {
                    "path": "res/inputtext/sonnet_unparsed_soup_1.txt",
                    "parameters": {
                        gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX : "f[0-9]*"
                        }
                },
                {
                    "path": "res/inputtext/sonnet_unparsed_soup_2.txt",
                    "parameters": {
                        gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX : "f[0-9]*",
                        gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX : "ra"
                        }
                },
                {
                    "path": "res/inputtext/sonnet_unparsed_soup_3.txt",
                    "parameters": {
                        gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX : "ra"
                        }
                },
                {
                    "path": "res/inputtext/sonnet_unparsed.xhtml",
                    "parameters": {
                        gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX : "f[0-9]*"
                        }
                },
        ]:
            self.load(case["path"], TextFileFormat.UNPARSED, 15, case["parameters"])

    def test_read_unparsed_unsorted(self):
        self.load_and_sort_id(
            "res/inputtext/sonnet_unparsed_order_1.txt",
            "f[0-9]*",
            IDSortingAlgorithm.UNSORTED,
            ["f001", "f003", "f005", "f004", "f002"]
        )

    def test_read_unparsed_numeric(self):
        self.load_and_sort_id(
            "res/inputtext/sonnet_unparsed_order_2.txt",
            "f[0-9]*",
            IDSortingAlgorithm.NUMERIC,
            ["f001", "f2", "f003", "f4", "f050"]
        )

    def test_read_unparsed_numeric_2(self):
        self.load_and_sort_id(
            "res/inputtext/sonnet_unparsed_order_3.txt",
            "f[0-9]*",
            IDSortingAlgorithm.NUMERIC,
            ["f001", "f2", "f003", "f4", "f050"]
        )

    def test_read_unparsed_lexicographic(self):
        self.load_and_sort_id(
            "res/inputtext/sonnet_unparsed_order_4.txt",
            "[a-z][0-9]*",
            IDSortingAlgorithm.LEXICOGRAPHIC,
            ["a005", "b002", "c004", "d001", "e003"]
        )

    def test_read_unparsed_numeric_3(self):
        self.load_and_sort_id(
            "res/inputtext/sonnet_unparsed_order_5.txt",
            "[a-z][0-9]*",
            IDSortingAlgorithm.NUMERIC,
            ["d001", "b002", "e003", "c004", "a005"]
        )

    def test_set_language(self):
        tfl = self.load()
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

    def test_append_fragment(self):
        tfl = TextFile()
        self.assertEqual(len(tfl), 0)
        tfl.append_fragment(TextFragment("a1", Language.EN, "fragment 1"))
        self.assertEqual(len(tfl), 1)

    def test_append_fragment_multiple(self):
        tfl = TextFile()
        self.assertEqual(len(tfl), 0)
        tfl.append_fragment(TextFragment("a1", Language.EN, "fragment 1"))
        self.assertEqual(len(tfl), 1)
        tfl.append_fragment(TextFragment("a2", Language.EN, "fragment 2"))
        self.assertEqual(len(tfl), 2)
        tfl.append_fragment(TextFragment("a3", Language.EN, "fragment 3"))
        self.assertEqual(len(tfl), 3)

    def test_get_slice_no_args(self):
        tfl = self.load()
        sli = tfl.get_slice()
        self.assertEqual(len(sli), 15)

    def test_get_slice_only_start(self):
        self.load_and_slice(10, 5)

    def test_get_slice_start_and_end(self):
        self.load_and_slice(5, 5, 10)

    def test_get_slice_start_greater_than_length(self):
        self.load_and_slice(1, 100)

    def test_get_slice_start_less_than_zero(self):
        self.load_and_slice(15, -1)

    def test_get_slice_end_greater_then_length(self):
        self.load_and_slice(15, 0, 100)

    def test_get_slice_end_less_than_zero(self):
        self.load_and_slice(1, 0, -1)

    def test_get_slice_end_less_than_start(self):
        self.load_and_slice(1, 10, 5)

if __name__ == '__main__':
    unittest.main()



