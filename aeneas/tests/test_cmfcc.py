#!/usr/bin/env python
# coding=utf-8

import numpy
import unittest

from aeneas.audiofile import AudioFileMonoWAVE
import aeneas.globalfunctions as gf

class TestCMFCC(unittest.TestCase):

    AUDIO = gf.absolute_path("res/cmfcc/audio.wav", __file__)
    MFCC_PRE_PY = gf.absolute_path("res/cmfcc/mfcc_py", __file__)
    MFCC_PRE_C = gf.absolute_path("res/cmfcc/mfcc_c", __file__)

    def compare_with_tolerance(self, a, b, tolerance=1E-6):
        return not ((a - b) > tolerance).any()

    def test_compute_mfcc(self):
        try:
            import aeneas.cmfcc
            mfcc_pre_py = numpy.loadtxt(self.MFCC_PRE_PY)
            mfcc_pre_c = numpy.loadtxt(self.MFCC_PRE_C)
            audio_file = AudioFileMonoWAVE(self.AUDIO)
            audio_file.load_data()
            mfcc_c = aeneas.cmfcc.cmfcc_compute_mfcc(
                audio_file.audio_data,
                audio_file.audio_sample_rate,
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
        except ImportError:
            pass

if __name__ == '__main__':
    unittest.main()



