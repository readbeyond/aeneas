#!/usr/bin/env python
# coding=utf-8

import numpy
import unittest

from aeneas.audiofile import AudioFileMonoWAV
import aeneas.tests as at

class TestCMFCC(unittest.TestCase):

    AUDIO = at.get_abs_path("res/cmfcc/audio.wav")
    MFCC_PRE_PY = at.get_abs_path("res/cmfcc/mfcc_py")
    MFCC_PRE_C = at.get_abs_path("res/cmfcc/mfcc_c")

    def compare_with_tolerance(self, a, b, tolerance=1E-6):
        return not ((a - b) > tolerance).any()

    def test_compute_mfcc(self):
        try:
            import aeneas.cmfcc
            mfcc_pre_py = numpy.loadtxt(self.MFCC_PRE_PY)
            mfcc_pre_c = numpy.loadtxt(self.MFCC_PRE_C)
            audio_file = AudioFileMonoWAV(self.AUDIO)
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



