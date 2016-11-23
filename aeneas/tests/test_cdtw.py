#!/usr/bin/env python
# coding=utf-8

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2016, Alberto Pettarin (www.albertopettarin.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
            self.assertEqual(best_path[-1], (n - 1, m - 1))
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()
