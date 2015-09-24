#!/usr/bin/env python
# coding=utf-8

import os
import tempfile
import unittest

from . import get_abs_path, delete_file, delete_directory

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

    FILES = {
        "epub": {
            "path": get_abs_path("res/container/job.epub"),
            "format": ContainerFormat.EPUB,
            "config_size": 599
         },
        "tar": {
            "path": get_abs_path("res/container/job.tar"),
            "format": ContainerFormat.TAR,
            "config_size": 599
         },
        "tar_bz2": {
            "path": get_abs_path("res/container/job.tar.bz2"),
            "format": ContainerFormat.TAR_BZ2,
            "config_size": 599
         },
        "tar": {
            "path": get_abs_path("res/container/job.tar.gz"),
            "format": ContainerFormat.TAR_GZ,
            "config_size": 599
         },
        "unpacked": {
            "path": get_abs_path("res/container/job"),
            "format": ContainerFormat.UNPACKED,
            "config_size": 599
         },
        "zip": {
            "path": get_abs_path("res/container/job.zip"),
            "format": ContainerFormat.ZIP,
            "config_size": 599
         },
        "zip_utf8": {
            "path": get_abs_path("res/container/job_utf8.zip"),
            "format": ContainerFormat.ZIP,
            "config_size": 633
         },
    }

    def test_constructor(self):
        for key in self.FILES:
            f = self.FILES[key]
            file_path = f["path"]
            container_format = f["format"]
            cont = Container(file_path, container_format)
            self.assertEqual(cont.file_path, file_path)
            self.assertEqual(cont.container_format, container_format)

    def test_guess_container(self):
        for key in self.FILES:
            f = self.FILES[key]
            cont = Container(f["path"])
            self.assertEqual(cont.container_format, f["format"])

    def test_entries(self):
        for key in self.FILES:
            f = self.FILES[key]
            cont = Container(f["path"])
            self.assertEqual(cont.entries(), self.EXPECTED_ENTRIES)

    def test_entries_unpacked_relative(self):
        f = self.FILES["unpacked"]
        cont = Container(f["path"])
        self.assertEqual(cont.entries(), self.EXPECTED_ENTRIES)

    def test_entries_unpacked_absolute(self):
        f = self.FILES["unpacked"]
        cont = Container(os.path.abspath(f["path"]))
        self.assertEqual(cont.entries(), self.EXPECTED_ENTRIES)

    def test_is_safe(self):
        for key in self.FILES:
            f = self.FILES[key]
            cont = Container(f["path"])
            self.assertTrue(cont.is_safe)

    def test_is_entry_safe_false(self):
        cont = Container(self.FILES["unpacked"]["path"])
        for entry in [
                "../foo",
                "/foo",
                "foo/../../../../../../../../../../../../bar",
                "foo/../../../../../bar/../../../../../../baz"
            ]:
            self.assertFalse(cont.is_entry_safe(entry))

    def test_is_entry_safe_true(self):
        cont = Container(self.FILES["unpacked"]["path"])
        for entry in [
                "foo",
                "foo/bar",
                "foo/../bar",
                "foo/../bar/baz",
                "foo/../bar/../baz",
                "./foo",
                "./foo/bar",
                "foo/./bar"
            ]:
            self.assertTrue(cont.is_entry_safe(entry))

    def test_read_entry_existing(self):
        entry = "config.txt"
        for key in self.FILES:
            f = self.FILES[key]
            cont = Container(f["path"])
            result = cont.read_entry(entry)
            self.assertNotEqual(result, None)
            self.assertEqual(len(result), f["config_size"])

    def test_find_entry_existing(self):
        entry = "config.txt"
        for key in self.FILES:
            f = self.FILES[key]
            cont = Container(f["path"])
            self.assertTrue(cont.find_entry(entry, exact=True))
            self.assertTrue(cont.find_entry(entry, exact=False))

    def test_find_entry_existing_not_exact(self):
        entry = "p001.xhtml"
        for key in self.FILES:
            f = self.FILES[key]
            cont = Container(f["path"])
            self.assertFalse(cont.find_entry(entry, exact=True))
            self.assertTrue(cont.find_entry(entry, exact=False))

    def test_read_entry_missing(self):
        entry = "config_not_existing.txt"
        for key in self.FILES:
            f = self.FILES[key]
            cont = Container(f["path"])
            result = cont.read_entry(entry)
            self.assertEqual(result, None)

    def test_find_entry_missing(self):
        entry = "config_not_existing.txt"
        for key in self.FILES:
            f = self.FILES[key]
            cont = Container(f["path"])
            self.assertFalse(cont.find_entry(entry, exact=True))
            self.assertFalse(cont.find_entry(entry, exact=False))

    def test_decompress(self):
        output_path = tempfile.mkdtemp()
        for key in self.FILES:
            f = self.FILES[key]
            cont = Container(f["path"])
            cont.decompress(output_path)
            copy = Container(output_path, ContainerFormat.UNPACKED)
            self.assertEqual(copy.entries(), self.EXPECTED_ENTRIES)
            delete_directory(output_path)

    def test_compress_unpacked(self):
        input_path = self.FILES["unpacked"]["path"]
        output_path = tempfile.mkdtemp()
        cont = Container(output_path, ContainerFormat.UNPACKED)
        cont.compress(input_path)
        self.assertFalse(os.path.isfile(output_path))
        copy = Container(output_path, ContainerFormat.UNPACKED)
        self.assertEqual(copy.entries(), self.EXPECTED_ENTRIES)
        delete_directory(output_path)

    def test_compress_file(self):
        input_path = self.FILES["unpacked"]["path"]
        for key in self.FILES:
            fmt = self.FILES[key]["format"]
            if fmt != ContainerFormat.UNPACKED:
                handler, output_path = tempfile.mkstemp(suffix="." + fmt)
                cont = Container(output_path, fmt)
                cont.compress(input_path)
                self.assertTrue(os.path.isfile(output_path))
                copy = Container(output_path, fmt)
                self.assertEqual(copy.entries(), self.EXPECTED_ENTRIES)
                delete_file(handler, output_path)

if __name__ == '__main__':
    unittest.main()



