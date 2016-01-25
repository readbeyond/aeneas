#!/usr/bin/env python
# coding=utf-8

import numpy
import unittest

import aeneas.globalfunctions as gf

class TestCDTW(unittest.TestCase):

    MFCC1 = gf.absolute_path("res/cdtw/mfcc1_12_1332", __file__)
    MFCC2 = gf.absolute_path("res/cdtw/mfcc2_12_868", __file__)

    def test_compute_path(self):
        try:
            import aeneas.cdtw.cdtw
            mfcc1 = numpy.loadtxt(self.MFCC1)
            mfcc2 = numpy.loadtxt(self.MFCC2)
            l, n = mfcc1.shape
            l, m = mfcc2.shape
            delta = 3000
            if delta > m:
                delta = m
            best_path = aeneas.cdtw.cdtw.compute_best_path(mfcc1, mfcc2, delta)
            self.assertEqual(len(best_path), 1418)
            self.assertEqual(best_path[0], (0, 0))
            self.assertEqual(best_path[-1], (n-1, m-1))
        except ImportError:
            pass

if __name__ == '__main__':
    unittest.main()



