#!/usr/bin/env python
# coding=utf-8

import os
import shutil
import sys
import tempfile
import unittest

from . import get_abs_path

from aeneas.container import Container, ContainerFormat

class TestContainer(unittest.TestCase):

    EXPECTED_ENTRIES = [
        "assets/p001.mp3",
        "assets/p001.xhtml",
        "assets/p002.mp3",
        "assets/p002.xhtml",
        "assets/p003.mp3",
        "assets/p003.xhtml",
        "config.txt"
    ]

    JOB_EPUB = get_abs_path("res/container/job.epub")
    JOB_TAR = get_abs_path("res/container/job.tar")
    JOB_TAR_BZ2 = get_abs_path("res/container/job.tar.bz2")
    JOB_TAR_GZ = get_abs_path("res/container/job.tar.gz")
    JOB_UNPACKED = get_abs_path("res/container/job")
    JOB_ZIP = get_abs_path("res/container/job.zip")
    JOB_ZIP_UTF8 = get_abs_path("res/container/job_utf8.zip")

    def test_constructor_01(self):
        file_path = self.JOB_ZIP 
        container_format = ContainerFormat.ZIP
        cont = Container(file_path, container_format)
        self.assertEqual(cont.file_path, file_path)
        self.assertEqual(cont.container_format, container_format)

    def test_guess_container_format_01(self):
        file_path = self.JOB_ZIP
        cont = Container(file_path)
        self.assertEqual(cont.container_format, ContainerFormat.ZIP)

    def test_guess_container_format_02(self):
        file_path = self.JOB_TAR
        cont = Container(file_path)
        self.assertEqual(cont.container_format, ContainerFormat.TAR)

    def test_guess_container_format_03(self):
        file_path = self.JOB_TAR_GZ
        cont = Container(file_path)
        self.assertEqual(cont.container_format, ContainerFormat.TAR_GZ)

    def test_guess_container_format_04(self):
        file_path = self.JOB_TAR_BZ2
        cont = Container(file_path)
        self.assertEqual(cont.container_format, ContainerFormat.TAR_BZ2)

    def test_guess_container_format_05(self):
        file_path = self.JOB_UNPACKED
        cont = Container(file_path)
        self.assertEqual(cont.container_format, ContainerFormat.UNPACKED)

    def test_guess_container_format_06(self):
        file_path = self.JOB_EPUB
        cont = Container(file_path)
        self.assertEqual(cont.container_format, ContainerFormat.EPUB)

    def test_entries_zip(self):
        file_path = self.JOB_ZIP
        cont = Container(file_path)
        self.assertEqual(cont.entries(), self.EXPECTED_ENTRIES)

    def test_entries_epub(self):
        file_path = self.JOB_EPUB
        cont = Container(file_path)
        self.assertEqual(cont.entries(), self.EXPECTED_ENTRIES)

    def test_entries_unpacked_relative(self):
        file_path = self.JOB_UNPACKED
        cont = Container(file_path)
        self.assertEqual(cont.entries(), self.EXPECTED_ENTRIES)

    def test_entries_unpacked_absolute(self):
        file_path = os.path.abspath(self.JOB_UNPACKED)
        cont = Container(file_path)
        self.assertEqual(cont.entries(), self.EXPECTED_ENTRIES)

    def test_entries_tar(self):
        file_path = self.JOB_TAR
        cont = Container(file_path)
        self.assertEqual(cont.entries(), self.EXPECTED_ENTRIES)

    def test_entries_tar_gz(self):
        file_path = self.JOB_TAR_GZ
        cont = Container(file_path)
        self.assertEqual(cont.entries(), self.EXPECTED_ENTRIES)

    def test_entries_tar_bz2(self):
        file_path = self.JOB_TAR_BZ2
        cont = Container(file_path)
        self.assertEqual(cont.entries(), self.EXPECTED_ENTRIES)

    def test_is_safe(self):
        file_path = self.JOB_ZIP
        cont = Container(file_path)
        self.assertTrue(cont.is_safe)

    def test_read_entry_zip(self):
        file_path = self.JOB_ZIP
        entry = "config.txt"
        cont = Container(file_path)
        result = cont.read_entry(entry)
        self.assertNotEqual(result, None)
        self.assertEqual(len(result), 599)

    def test_read_entry_zip_utf8(self):
        file_path = self.JOB_ZIP_UTF8
        entry = "config.txt"
        cont = Container(file_path)
        result = cont.read_entry(entry)
        self.assertNotEqual(result, None)
        self.assertEqual(len(result), 633)

    def test_read_entry_epub(self):
        file_path = self.JOB_EPUB
        entry = "config.txt"
        cont = Container(file_path)
        result = cont.read_entry(entry)
        self.assertNotEqual(result, None)
        self.assertEqual(len(result), 599)

    def test_read_entry_tar(self):
        file_path = self.JOB_TAR
        entry = "config.txt"
        cont = Container(file_path)
        result = cont.read_entry(entry)
        self.assertNotEqual(result, None)
        self.assertEqual(len(result), 599)

    def test_read_entry_tar_gz(self):
        file_path = self.JOB_TAR_GZ
        entry = "config.txt"
        cont = Container(file_path)
        result = cont.read_entry(entry)
        self.assertNotEqual(result, None)
        self.assertEqual(len(result), 599)

    def test_read_entry_tar_bz2(self):
        file_path = self.JOB_TAR_BZ2
        entry = "config.txt"
        cont = Container(file_path)
        result = cont.read_entry(entry)
        self.assertNotEqual(result, None)
        self.assertEqual(len(result), 599)

    def test_read_entry_unpacked(self):
        file_path = self.JOB_UNPACKED
        entry = "config.txt"
        cont = Container(file_path)
        result = cont.read_entry(entry)
        self.assertNotEqual(result, None)
        self.assertEqual(len(result), 599)

    def test_decompress_zip(self):
        file_path = self.JOB_ZIP
        output_path = tempfile.mkdtemp()
        cont = Container(file_path)
        cont.decompress(output_path)
        copy = Container(output_path, ContainerFormat.UNPACKED)
        self.assertEqual(copy.entries(), self.EXPECTED_ENTRIES)
        shutil.rmtree(output_path)

    def test_decompress_epub(self):
        file_path = self.JOB_EPUB
        output_path = tempfile.mkdtemp()
        cont = Container(file_path)
        cont.decompress(output_path)
        copy = Container(output_path, ContainerFormat.UNPACKED)
        self.assertEqual(copy.entries(), self.EXPECTED_ENTRIES)
        shutil.rmtree(output_path)

    def test_decompress_tar(self):
        file_path = self.JOB_TAR
        output_path = tempfile.mkdtemp()
        cont = Container(file_path)
        cont.decompress(output_path)
        copy = Container(output_path, ContainerFormat.UNPACKED)
        self.assertEqual(copy.entries(), self.EXPECTED_ENTRIES)
        shutil.rmtree(output_path)

    def test_decompress_tar_gz(self):
        file_path = self.JOB_TAR_GZ
        output_path = tempfile.mkdtemp()
        cont = Container(file_path)
        cont.decompress(output_path)
        copy = Container(output_path, ContainerFormat.UNPACKED)
        self.assertEqual(copy.entries(), self.EXPECTED_ENTRIES)
        shutil.rmtree(output_path)

    def test_decompress_tar_bz2(self):
        file_path = self.JOB_TAR_BZ2
        output_path = tempfile.mkdtemp()
        cont = Container(file_path)
        cont.decompress(output_path)
        copy = Container(output_path, ContainerFormat.UNPACKED)
        self.assertEqual(copy.entries(), self.EXPECTED_ENTRIES)
        shutil.rmtree(output_path)

    def test_decompress_unpacked(self):
        file_path = self.JOB_UNPACKED
        output_path = tempfile.mkdtemp()
        cont = Container(file_path)
        cont.decompress(output_path)
        copy = Container(output_path, ContainerFormat.UNPACKED)
        self.assertEqual(copy.entries(), self.EXPECTED_ENTRIES)
        shutil.rmtree(output_path)

    def test_compress_zip(self):
        input_path = self.JOB_UNPACKED
        handler, output_path = tempfile.mkstemp(suffix=".zip")
        cont = Container(output_path, ContainerFormat.ZIP)
        cont.compress(input_path)
        self.assertTrue(os.path.isfile(output_path))
        copy = Container(output_path, ContainerFormat.ZIP)
        self.assertEqual(copy.entries(), self.EXPECTED_ENTRIES)
        os.remove(output_path)

    def test_compress_tar(self):
        input_path = self.JOB_UNPACKED
        handler, output_path = tempfile.mkstemp(suffix=".tar")
        cont = Container(output_path, ContainerFormat.TAR)
        cont.compress(input_path)
        self.assertTrue(os.path.isfile(output_path))
        copy = Container(output_path, ContainerFormat.TAR)
        self.assertEqual(copy.entries(), self.EXPECTED_ENTRIES)
        os.remove(output_path)

    def test_compress_tar_gz(self):
        input_path = self.JOB_UNPACKED
        handler, output_path = tempfile.mkstemp(suffix=".tar.gz")
        cont = Container(output_path, ContainerFormat.TAR_GZ)
        cont.compress(input_path)
        self.assertTrue(os.path.isfile(output_path))
        copy = Container(output_path, ContainerFormat.TAR_GZ)
        self.assertEqual(copy.entries(), self.EXPECTED_ENTRIES)
        os.remove(output_path)

    def test_compress_tar_bz2(self):
        input_path = self.JOB_UNPACKED
        handler, output_path = tempfile.mkstemp(suffix=".tar.bz2")
        cont = Container(output_path, ContainerFormat.TAR_BZ2)
        cont.compress(input_path)
        self.assertTrue(os.path.isfile(output_path))
        copy = Container(output_path, ContainerFormat.TAR_BZ2)
        self.assertEqual(copy.entries(), self.EXPECTED_ENTRIES)
        os.remove(output_path)

    def test_compress_unpacked(self):
        input_path = self.JOB_UNPACKED
        output_path = tempfile.mkdtemp()
        cont = Container(output_path, ContainerFormat.UNPACKED)
        cont.compress(input_path)
        self.assertFalse(os.path.isfile(output_path))
        copy = Container(output_path, ContainerFormat.UNPACKED)
        self.assertEqual(copy.entries(), self.EXPECTED_ENTRIES)
        shutil.rmtree(output_path)

    def test_find_entry(self):
        file_path = self.JOB_ZIP
        entry = "p001.xhtml"
        cont = Container(file_path)
        self.assertFalse(cont.find_entry(entry, exact=True))
        self.assertTrue(cont.find_entry(entry, exact=False))
        entry = "config.txt"
        self.assertTrue(cont.find_entry(entry, exact=True))
        self.assertTrue(cont.find_entry(entry, exact=False))
        entry = "not.existing.txt"
        self.assertFalse(cont.find_entry(entry, exact=True))
        self.assertFalse(cont.find_entry(entry, exact=False))

if __name__ == '__main__':
    unittest.main()



