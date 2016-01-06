#!/usr/bin/env python
# coding=utf-8

import os
import unittest

from aeneas.tools.download import DownloadCLI 
import aeneas.globalfunctions as gf

class TestDownloadCLI(unittest.TestCase):

    def execute(self, parameters, expected_exit_code):
        output_path = gf.tmp_directory()
        params = ["placeholder"]
        for p_type, p_value in parameters:
            if p_type == "in":
                params.append(gf.absolute_path(p_value, __file__))
            elif p_type == "out":
                params.append(os.path.join(output_path, p_value))
            else:
                params.append(p_value)
        exit_code = DownloadCLI(use_sys=False).run(arguments=params)
        gf.delete_directory(output_path)
        self.assertEqual(exit_code, expected_exit_code)

    def test_help(self):
        self.execute([], 2)
        self.execute([("", "-h")], 2)
        self.execute([("", "--help")], 2)
        self.execute([("", "--version")], 2)

    def test_list(self):
        self.execute([
            ("", "https://www.youtube.com/watch?v=rU4a7AA8wM0"),
            ("", "--list")
        ], 0)

    def test_download(self):
        self.execute([
            ("", "https://www.youtube.com/watch?v=rU4a7AA8wM0"),
            ("out", "sonnet.m4a")
        ], 0)

    def test_download_bad_url(self):
        self.execute([
            ("", "https://www.youtube.com/watch?v=aaaaaaaaaaa"),
            ("out", "sonnet.m4a")
        ], 1)

    def test_download_cannot_write(self):
        self.execute([
            ("", "https://www.youtube.com/watch?v=rU4a7AA8wM0"),
            ("", "/foo/bar/baz.m4a")
        ], 1)

    def test_download_missing_1(self):
        self.execute([
            ("", "https://www.youtube.com/watch?v=rU4a7AA8wM0"),
        ], 2)
  
    def test_download_missing_2(self):
        self.execute([
            ("out", "sonnet.m4a")
        ], 2)

    def test_download_index(self):
        self.execute([
            ("", "https://www.youtube.com/watch?v=rU4a7AA8wM0"),
            ("out", "sonnet.m4a"),
            ("", "--index=0")
        ], 0)

    def test_download_smallest(self):
        self.execute([
            ("", "https://www.youtube.com/watch?v=rU4a7AA8wM0"),
            ("out", "sonnet.ogg"),
            ("", "--smallest-audio")
        ], 0)

    def test_download_format(self):
        self.execute([
            ("", "https://www.youtube.com/watch?v=rU4a7AA8wM0"),
            ("out", "sonnet.ogg"),
            ("", "--largest-audio"),
            ("", "--format=ogg")
        ], 0)



if __name__ == '__main__':
    unittest.main()



