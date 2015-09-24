#!/usr/bin/env python
# coding=utf-8

import numpy
import unittest
from scikits.audiolab import wavread

from . import get_abs_path

class TestCMFCC(unittest.TestCase):

    AUDIO = get_abs_path("res/cmfcc/audio.wav")
    MFCC_PRE_PY = get_abs_path("res/cmfcc/mfcc_py")
    MFCC_PRE_C = get_abs_path("res/cmfcc/mfcc_c")

    def compare_with_tolerance(self, a, b, tolerance=1E-6):
        return not ((a - b) > tolerance).any()

    def test_compute_mfcc(self):
        try:
            import aeneas.cmfcc
            mfcc_pre_py = numpy.loadtxt(self.MFCC_PRE_PY)
            mfcc_pre_c = numpy.loadtxt(self.MFCC_PRE_C)
            data, sample_rate, encoding = wavread(self.AUDIO)
            mfcc_c = aeneas.cmfcc.cmfcc_compute_mfcc(
                    data,
                    sample_rate,
                    25,
                    40,
                    13,
                    512,
                    133.3333,
                    6855.4976,
                    0.97,
                    0.0256
                ).transpose()
            self.assertTrue(self.compare_with_tolerance(mfcc_c, mfcc_pre_py))
            self.assertTrue(self.compare_with_tolerance(mfcc_c, mfcc_pre_c))
        except ImportError as e:
            pass

if __name__ == '__main__':
    unittest.main()



