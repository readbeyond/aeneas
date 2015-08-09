#!/usr/bin/env python
# coding=utf-8

import unittest

from . import get_abs_path

from aeneas.vad import VAD

class TestVAD(unittest.TestCase):

    def test_vad_01(self):
        vad = VAD(get_abs_path("res/vad/nsn.wav"))
        vad.compute_mfcc()
        vad.compute_vad()
        self.assertEqual(len(vad.speech), 1)
        self.assertEqual(len(vad.nonspeech), 2)

    def test_vad_02(self):
        vad = VAD(get_abs_path("res/vad/ns.wav"))
        vad.compute_mfcc()
        vad.compute_vad()
        self.assertEqual(len(vad.speech), 1)
        self.assertEqual(len(vad.nonspeech), 1)

    def test_vad_03(self):
        vad = VAD(get_abs_path("res/vad/n.wav"))
        vad.compute_mfcc()
        vad.compute_vad()
        self.assertEqual(len(vad.speech), 0)
        self.assertEqual(len(vad.nonspeech), 1)

    def test_vad_04(self):
        vad = VAD(get_abs_path("res/vad/sns.wav"))
        vad.compute_mfcc()
        vad.compute_vad()
        self.assertEqual(len(vad.speech), 2)
        self.assertEqual(len(vad.nonspeech), 1)

    def test_vad_05(self):
        vad = VAD(get_abs_path("res/vad/sn.wav"))
        vad.compute_mfcc()
        vad.compute_vad()
        self.assertEqual(len(vad.speech), 1)
        self.assertEqual(len(vad.nonspeech), 1)

    def test_vad_06(self):
        vad = VAD(get_abs_path("res/vad/s.wav"))
        vad.compute_mfcc()
        vad.compute_vad()
        self.assertEqual(len(vad.speech), 1)
        self.assertEqual(len(vad.nonspeech), 0)

    def test_vad_07(self):
        vad = VAD(get_abs_path("res/vad/zero.wav"))
        vad.compute_mfcc()
        vad.compute_vad()
        self.assertEqual(len(vad.speech), 0)
        self.assertEqual(len(vad.nonspeech), 1)


if __name__ == '__main__':
    unittest.main()



