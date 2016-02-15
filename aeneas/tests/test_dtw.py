#!/usr/bin/env python
# coding=utf-8

import numpy
import os
import unittest

from aeneas.audiofilemfcc import AudioFileMFCC
from aeneas.dtw import DTWAlgorithm
from aeneas.dtw import DTWAligner
from aeneas.dtw import DTWAlignerNotInitialized
import aeneas.globalfunctions as gf

class TestDTWAligner(unittest.TestCase):

    AUDIO_FILE = gf.absolute_path("res/audioformats/mono.16000.wav", __file__)
    NUMPY_ARRAY_1 = numpy.loadtxt(gf.absolute_path("res/cdtw/mfcc1_12_1332", __file__))
    NUMPY_ARRAY_2 = numpy.loadtxt(gf.absolute_path("res/cdtw/mfcc2_12_868", __file__))

    def test_create_aligner(self):
        aligner = DTWAligner()
        self.assertIsNone(aligner.real_wave_mfcc)
        self.assertIsNone(aligner.synt_wave_mfcc)
        self.assertIsNone(aligner.real_wave_path)
        self.assertIsNone(aligner.synt_wave_path)

    def test_set_real_wave_path(self):
        aligner = DTWAligner(real_wave_path=self.AUDIO_FILE)
        self.assertIsNotNone(aligner.real_wave_mfcc)
        self.assertIsNone(aligner.synt_wave_mfcc)
        self.assertIsNotNone(aligner.real_wave_path)
        self.assertIsNone(aligner.synt_wave_path)

    def test_set_synt_wave_path(self):
        aligner = DTWAligner(synt_wave_path=self.AUDIO_FILE)
        self.assertIsNone(aligner.real_wave_mfcc)
        self.assertIsNotNone(aligner.synt_wave_path)
        self.assertIsNone(aligner.real_wave_path)
        self.assertIsNotNone(aligner.synt_wave_mfcc)

    def test_set_real_wave_mfcc(self):
        af = AudioFileMFCC(self.AUDIO_FILE)
        aligner = DTWAligner(real_wave_mfcc=af)
        self.assertIsNotNone(aligner.real_wave_mfcc)
        self.assertIsNone(aligner.synt_wave_mfcc)
        self.assertIsNone(aligner.real_wave_path)
        self.assertIsNone(aligner.synt_wave_path)

    def test_set_synt_wave_mfcc(self):
        af = AudioFileMFCC(self.AUDIO_FILE)
        aligner = DTWAligner(synt_wave_mfcc=af)
        self.assertIsNone(aligner.real_wave_mfcc)
        self.assertIsNotNone(aligner.synt_wave_mfcc)
        self.assertIsNone(aligner.real_wave_path)
        self.assertIsNone(aligner.synt_wave_path)

    def test_compute_acm_none(self):
        aligner = DTWAligner()
        with self.assertRaises(DTWAlignerNotInitialized):
            aligner.compute_accumulated_cost_matrix()

    def test_compute_acm_real_path(self):
        aligner = DTWAligner(real_wave_path=self.AUDIO_FILE)
        with self.assertRaises(DTWAlignerNotInitialized):
            aligner.compute_accumulated_cost_matrix()

    def test_compute_acm_synt_path(self):
        aligner = DTWAligner(synt_wave_path=self.AUDIO_FILE)
        with self.assertRaises(DTWAlignerNotInitialized):
            aligner.compute_accumulated_cost_matrix()

    def test_compute_acm_real_mfcc(self):
        af = AudioFileMFCC(self.AUDIO_FILE)
        aligner = DTWAligner(real_wave_mfcc=af)
        with self.assertRaises(DTWAlignerNotInitialized):
            aligner.compute_accumulated_cost_matrix()

    def test_compute_acm_synt_mfcc(self):
        af = AudioFileMFCC(self.AUDIO_FILE)
        aligner = DTWAligner(synt_wave_mfcc=af)
        with self.assertRaises(DTWAlignerNotInitialized):
            aligner.compute_accumulated_cost_matrix()

    def test_compute_path_none(self):
        aligner = DTWAligner()
        with self.assertRaises(DTWAlignerNotInitialized):
            aligner.compute_accumulated_cost_matrix()

    def test_compute_path_real_path(self):
        aligner = DTWAligner(real_wave_path=self.AUDIO_FILE)
        with self.assertRaises(DTWAlignerNotInitialized):
            aligner.compute_path()

    def test_compute_path_synt_path(self):
        aligner = DTWAligner(synt_wave_path=self.AUDIO_FILE)
        with self.assertRaises(DTWAlignerNotInitialized):
            aligner.compute_path()

    def test_compute_path_real_mfcc(self):
        af = AudioFileMFCC(self.AUDIO_FILE)
        aligner = DTWAligner(real_wave_mfcc=af)
        with self.assertRaises(DTWAlignerNotInitialized):
            aligner.compute_path()

    def test_compute_path_synt_mfcc(self):
        af = AudioFileMFCC(self.AUDIO_FILE)
        aligner = DTWAligner(synt_wave_mfcc=af)
        with self.assertRaises(DTWAlignerNotInitialized):
            aligner.compute_path()

    def test_compute_acm(self):
        # NOTE this takes too long, run as part of the long_ tests
        pass

    def test_compute_path(self):
        # NOTE this takes too long, run as part of the long_ tests
        pass



if __name__ == '__main__':
    unittest.main()



