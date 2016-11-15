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

from aeneas.audiofile import AudioFile
import aeneas.globalfunctions as gf


class TestCMFCC(unittest.TestCase):

    AUDIO = gf.absolute_path("res/audioformats/mono.16000.wav", __file__)

    def test_compute_mfcc(self):
        try:
            import aeneas.cmfcc.cmfcc
            audio_file = AudioFile(self.AUDIO)
            audio_file.read_samples_from_file()
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


if __name__ == "__main__":
    unittest.main()
