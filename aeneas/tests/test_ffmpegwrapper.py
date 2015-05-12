#!/usr/bin/env python
# coding=utf-8

import os
import sys
import tempfile
import unittest

from . import get_abs_path

from aeneas.ffmpegwrapper import FFMPEGWrapper

class TestFFMPEGWrapper(unittest.TestCase):

    def test_convert(self):
        input_file_path = get_abs_path("res/container/job/assets/p001.mp3")
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        converter = FFMPEGWrapper()
        result = converter.convert(input_file_path, output_file_path)
        self.assertEqual(result, output_file_path)
        os.close(handler)
        os.remove(output_file_path)

    def test_cannotload(self):
        input_file_path = get_abs_path("res/this_file_does_not_exist.mp3")
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        converter = FFMPEGWrapper()
        with self.assertRaises(OSError):
            converter.convert(input_file_path, output_file_path)
        os.close(handler)
        os.remove(output_file_path)
 
if __name__ == '__main__':
    unittest.main()



