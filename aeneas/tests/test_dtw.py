#!/usr/bin/env python
# coding=utf-8

import numpy
import os
import unittest

from aeneas.dtw import DTWAlgorithm
from aeneas.dtw import DTWAligner
import aeneas.globalfunctions as gf

class TestDTWAligner(unittest.TestCase):

    AUDIO_FILE = gf.absolute_path("res/cmfcc/audio.wav", __file__)
    NUMPY_ARRAY_1 = numpy.loadtxt(gf.absolute_path("res/cdtw/mfcc1_12_1332", __file__))
    NUMPY_ARRAY_2 = numpy.loadtxt(gf.absolute_path("res/cdtw/mfcc2_12_868", __file__))

    def test_create_aligner(self):
        aligner = DTWAligner()

    def test_get_length(self):
        aligner = DTWAligner()
        self.assertIsNone(aligner.real_wave_length)
        self.assertIsNone(aligner.synt_wave_length)

    def test_set_real_wave_path(self):
        aligner = DTWAligner()
        aligner.real_wave_path = self.AUDIO_FILE
        self.assertIsNotNone(aligner.real_wave_path)

    def test_set_synt_wave_path(self):
        aligner = DTWAligner()
        aligner.synt_wave_path = self.AUDIO_FILE
        self.assertIsNotNone(aligner.synt_wave_path)

    def test_set_real_wave_full_mfcc(self):
        aligner = DTWAligner()
        aligner.real_wave_full_mfcc = self.NUMPY_ARRAY_1
        self.assertIsNotNone(aligner.real_wave_full_mfcc)

    def test_set_real_wave_length(self):
        aligner = DTWAligner()
        aligner.real_wave_length = 123.45
        self.assertIsNotNone(aligner.real_wave_length)

    def test_set_synt_wave_full_mfcc(self):
        aligner = DTWAligner()
        aligner.synt_wave_full_mfcc = self.NUMPY_ARRAY_2
        self.assertIsNotNone(aligner.synt_wave_full_mfcc)

    def test_set_synt_wave_length(self):
        aligner = DTWAligner()
        aligner.synt_wave_length = 123.45
        self.assertIsNotNone(aligner.synt_wave_length)

    def test_compute_mfcc_not_set(self):
        aligner = DTWAligner()
        with self.assertRaises(OSError):
            aligner.compute_mfcc()

    def test_compute_mfcc(self):
        # NOTE this takes too long, run as part of the long_ tests
        pass

    def test_compute_accumulated_cost_matrix_not_set(self):
        aligner = DTWAligner()
        with self.assertRaises(Exception):
            aligner.compute_accumulated_cost_matrix()

    def test_compute_accumulated_cost_matrix(self):
        # NOTE this takes too long, run as part of the long_ tests
        pass

    def test_compute_path_not_set(self):
        aligner = DTWAligner()
        with self.assertRaises(Exception):
            aligner.compute_path()

    def test_compute_path(self):
        # NOTE this takes too long, run as part of the long_ tests
        pass



if __name__ == '__main__':
    unittest.main()



