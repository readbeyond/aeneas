#!/usr/bin/env python
# coding=utf-8

import numpy
import unittest

from aeneas.audiofile import AudioFileMonoWAVE
import aeneas.globalfunctions as gf

class TestCMFCC(unittest.TestCase):

    AUDIO = gf.absolute_path("res/audioformats/mono.16000.wav", __file__)

    def test_compute_mfcc(self):
        try:
            import aeneas.cmfcc.cmfcc
            audio_file = AudioFileMonoWAVE(self.AUDIO)
            mfcc_c = (aeneas.cmfcc.cmfcc.compute_from_data(
                audio_file.audio_samples,
                audio_file.audio_sample_rate,
                40,
                13,
                512,
                133.3333,
                6855.4976,
                0.97,
                0.025,
                0.010
            )[0]).transpose()
            self.assertEqual(mfcc_c.shape[0], 13)
            self.assertGreater(mfcc_c.shape[1], 0)
        except ImportError:
            pass

if __name__ == '__main__':
    unittest.main()



