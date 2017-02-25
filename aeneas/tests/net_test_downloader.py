#!/usr/bin/env python
# coding=utf-8

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2017, Alberto Pettarin (www.albertopettarin.it)
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

import unittest

from aeneas.downloader import Downloader
from aeneas.downloader import DownloadError
import aeneas.globalfunctions as gf


class TestDownloader(unittest.TestCase):

    URL_MALFORMED = u"foo"
    URL_INVALID = u"aaaaaaaaaaa"
    URL_VALID = u"https://www.youtube.com/watch?v=rU4a7AA8wM0"
    OUTPUT_PATH_INVALID = u"/foo/bar/baz"

    def audio_from_youtube(
            self,
            source_url,
            download=True,
            output_file_path=None,
            download_format=None,
            largest_audio=True
    ):
        return Downloader().audio_from_youtube(
            source_url,
            download=download,
            output_file_path=output_file_path,
            download_format=download_format,
            largest_audio=largest_audio
        )

    def download(
            self,
            expected_size,
            download_format=None,
            largest_audio=True,
    ):
        path = self.audio_from_youtube(
            self.URL_VALID,
            download=True,
            output_file_path=None,
            download_format=download_format,
            largest_audio=largest_audio
        )
        self.assertTrue(gf.file_can_be_read(path))
        self.assertEqual(gf.file_size(path), expected_size)
        gf.delete_file(None, path)

    def test_malformed_url(self):
        with self.assertRaises(DownloadError):
            self.audio_from_youtube(self.URL_MALFORMED, download=False)

    def test_invalid_url(self):
        with self.assertRaises(DownloadError):
            self.audio_from_youtube(self.URL_INVALID, download=False)

    def test_invalid_output_file(self):
        with self.assertRaises(OSError):
            self.audio_from_youtube(
                self.URL_VALID,
                download=True,
                output_file_path=self.OUTPUT_PATH_INVALID
            )

    def test_download_list(self):
        audiostreams = self.audio_from_youtube(self.URL_VALID, download=False)
        self.assertEqual(len(audiostreams), 5)

    def test_download_simple(self):
        self.download(1146884)

    def test_download_smallest(self):
        self.download(353237, largest_audio=False)

    def test_download_format(self):
        self.download(1146884, download_format=u"140")

    def test_download_format_smallest(self):
        self.download(1146884, download_format=u"140", largest_audio=False)

    def test_download_bad_format(self):
        self.download(1146884, download_format=u"-1")


if __name__ == "__main__":
    unittest.main()
