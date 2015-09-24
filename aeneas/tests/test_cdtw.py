#!/usr/bin/env python
# coding=utf-8

import numpy
import unittest

from . import get_abs_path

class TestCDTW(unittest.TestCase):

    MFCC1 = get_abs_path("res/cdtw/mfcc1_53")
    MFCC2 = get_abs_path("res/cdtw/mfcc2_53")

    def test_compute_path(self):
        try:
            import aeneas.cdtw
            mfcc1 = numpy.loadtxt(self.MFCC1)
            mfcc2 = numpy.loadtxt(self.MFCC2)
            l, n = mfcc1.shape
            l, m = mfcc2.shape
            norm2_1 = numpy.sqrt(numpy.sum(mfcc1 ** 2, 0))
            norm2_2 = numpy.sqrt(numpy.sum(mfcc2 ** 2, 0))
            delta = 3000
            if delta > m:
                delta = m
            best_path = aeneas.cdtw.cdtw_compute_best_path(
                mfcc1,
                mfcc2,
                norm2_1,
                norm2_2,
                delta
            )
            self.assertEqual(len(best_path), 1418)
            self.assertEqual(best_path[0], (0, 0))
            self.assertEqual(best_path[-1], (n-1, m-1))
        except ImportError as e:
            pass

if __name__ == '__main__':
    unittest.main()



