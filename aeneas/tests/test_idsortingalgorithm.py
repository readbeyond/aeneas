#!/usr/bin/env python
# coding=utf-8

import unittest

from aeneas.idsortingalgorithm import IDSortingAlgorithm

class TestIDSortingAlgorithm(unittest.TestCase):

    IDS = [u"b001", u"c03", u"d4", u"a2"]

    def test_invalid_algorithm(self):
        with self.assertRaises(ValueError):
            idsa = IDSortingAlgorithm(u"foo")

    def test_unsorted(self):
        expected = [u"b001", u"c03", u"d4", u"a2"]
        idsa = IDSortingAlgorithm(IDSortingAlgorithm.UNSORTED)
        sids = idsa.sort(self.IDS)
        self.assertTrue(sids == expected)

    def test_lexicographic(self):
        expected = [u"a2", u"b001", u"c03", u"d4"]
        idsa = IDSortingAlgorithm(IDSortingAlgorithm.LEXICOGRAPHIC)
        sids = idsa.sort(self.IDS)
        self.assertTrue(sids == expected)

    def test_numeric(self):
        expected = [u"b001", u"a2", u"c03", u"d4"]
        idsa = IDSortingAlgorithm(IDSortingAlgorithm.NUMERIC)
        sids = idsa.sort(self.IDS)
        self.assertTrue(sids == expected)

    def test_numeric_exception(self):
        bad_ids = [u"b002", u"d", u"c", u"a1"]
        idsa = IDSortingAlgorithm(IDSortingAlgorithm.NUMERIC)
        sids = idsa.sort(bad_ids)
        self.assertTrue(sids == bad_ids)

if __name__ == '__main__':
    unittest.main()



