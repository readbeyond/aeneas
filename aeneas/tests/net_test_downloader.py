#!/usr/bin/env python
# coding=utf-8

import unittest

from aeneas.downloader import Downloader
import aeneas.globalfunctions as gf

class TestDownloader(unittest.TestCase):

    URL_MALFORMED = "foo"
    URL_INVALID = "aaaaaaaaaaa"
    URL_VALID = "https://www.youtube.com/watch?v=rU4a7AA8wM0"

    def audio_from_youtube(
            self,
            source_url,
            download=True,
            output_file_path=None,
            preferred_index=None,
            largest_audio=True,
            preferred_format=None
    ):
        return Downloader().audio_from_youtube(
            source_url,
            download=download,
            output_file_path=output_file_path,
            preferred_index=preferred_index,
            largest_audio=largest_audio,
            preferred_format=preferred_format
        )

    def download(
            self,
            expected_size,
            preferred_index=None,
            largest_audio=True,
            preferred_format=None
    ):
        path = self.audio_from_youtube(
            self.URL_VALID,
            download=True,
            output_file_path=None,
            preferred_index=preferred_index,
            largest_audio=largest_audio,
            preferred_format=preferred_format
        )
        self.assertTrue(gf.file_can_be_read(path))
        self.assertEqual(gf.file_size(path), expected_size)
        gf.delete_file(None, path)

    def test_malformed_url(self):
        with self.assertRaises(ValueError):
            self.audio_from_youtube(self.URL_MALFORMED, download=False)

    def test_invalid_url(self):
        with self.assertRaises(ValueError):
            self.audio_from_youtube(self.URL_INVALID, download=False)

    def test_invalid_output_file(self):
        with self.assertRaises(OSError):
            self.audio_from_youtube(self.URL_VALID, download=True, output_file_path="/foo/bar/baz")

    def test_download_list(self):
        audiostreams = self.audio_from_youtube(self.URL_VALID, download=False)
        self.assertEqual(len(audiostreams), 6)

    def test_download_simple(self):
        self.download(2303039)

    def test_download_smallest(self):
        self.download(353237, largest_audio=False)

    def test_download_format(self):
        # NOTE on Python 2 pafy uses "ogg", while on Python 3 pafy uses "opus"
        if gf.PY2:
            fmt = "ogg"
        else:
            fmt = "opus"
        self.download(880809, preferred_format=fmt)

    def test_download_format_smallest(self):
        self.download(1147614, preferred_format="m4a", largest_audio=False)

    def test_download_index(self):
        self.download(405826, preferred_index=4)

    def test_download_index_out_of_range(self):
        self.download(2303039, preferred_index=1000)

    def test_download_index_and_bad_format(self):
        self.download(405826, preferred_index=4, preferred_format="m4a", largest_audio=True)



if __name__ == '__main__':
    unittest.main()



