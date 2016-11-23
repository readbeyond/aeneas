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

import os
import unittest

from aeneas.container import Container
from aeneas.container import ContainerFormat
import aeneas.globalfunctions as gf


class TestContainer(unittest.TestCase):

    NOT_EXISTING = gf.absolute_path("not_existing.zip", __file__)
    EMPTY_FILES = [
        gf.absolute_path("res/container/empty_file.epub", __file__),
        gf.absolute_path("res/container/empty_file.tar", __file__),
        gf.absolute_path("res/container/empty_file.tar.bz2", __file__),
        gf.absolute_path("res/container/empty_file.tar.gz", __file__),
        gf.absolute_path("res/container/empty_file.zip", __file__)
    ]

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
            "path": gf.absolute_path("res/container/job.epub", __file__),
            "format": ContainerFormat.EPUB,
            "config_size": 599
        },
        "tar": {
            "path": gf.absolute_path("res/container/job.tar", __file__),
            "format": ContainerFormat.TAR,
            "config_size": 599
        },
        "tar_bz2": {
            "path": gf.absolute_path("res/container/job.tar.bz2", __file__),
            "format": ContainerFormat.TAR_BZ2,
            "config_size": 599
        },
        "tar": {
            "path": gf.absolute_path("res/container/job.tar.gz", __file__),
            "format": ContainerFormat.TAR_GZ,
            "config_size": 599
        },
        "unpacked": {
            "path": gf.absolute_path("res/container/job", __file__),
            "format": ContainerFormat.UNPACKED,
            "config_size": 599
        },
        "zip": {
            "path": gf.absolute_path("res/container/job.zip", __file__),
            "format": ContainerFormat.ZIP,
            "config_size": 599
        },
        "zip_utf8": {
            "path": gf.absolute_path("res/container/job_utf8.zip", __file__),
            "format": ContainerFormat.ZIP,
            "config_size": 633
        },
    }

    def test_path_none(self):
        with self.assertRaises(TypeError):
            cont = Container(file_path=None)

    def test_invalid_container_format(self):
        with self.assertRaises(ValueError):
            con = Container(file_path=self.FILES["zip"]["path"], container_format="foo")

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

    def test_exists_file_not_existing(self):
        cont = Container(self.NOT_EXISTING)
        self.assertFalse(cont.exists())

    def test_exists_empty_file(self):
        for f in self.EMPTY_FILES:
            cont = Container(f)
            self.assertTrue(cont.exists())

    def test_exists_empty_directory(self):
        output_path = gf.tmp_directory()
        cont = Container(output_path)
        self.assertTrue(cont.exists())
        gf.delete_directory(output_path)

    def test_entries_file_not_existing(self):
        cont = Container(self.NOT_EXISTING)
        with self.assertRaises(TypeError):
            entries = cont.entries

    def test_entries_empty_file(self):
        for f in self.EMPTY_FILES:
            cont = Container(f)
            with self.assertRaises(OSError):
                self.assertEqual(len(cont.entries), 0)

    def test_entries_empty_directory(self):
        output_path = gf.tmp_directory()
        cont = Container(output_path)
        self.assertEqual(len(cont.entries), 0)
        gf.delete_directory(output_path)

    def test_entries(self):
        for key in self.FILES:
            f = self.FILES[key]
            cont = Container(f["path"])
            self.assertEqual(cont.entries, self.EXPECTED_ENTRIES)

    def test_entries_unpacked_relative(self):
        f = self.FILES["unpacked"]
        cont = Container(f["path"])
        self.assertEqual(cont.entries, self.EXPECTED_ENTRIES)

    def test_entries_unpacked_absolute(self):
        f = self.FILES["unpacked"]
        cont = Container(os.path.abspath(f["path"]))
        self.assertEqual(cont.entries, self.EXPECTED_ENTRIES)

    def test_is_safe_not_existing(self):
        cont = Container(self.NOT_EXISTING)
        with self.assertRaises(TypeError):
            self.assertTrue(cont.is_safe)

    def test_is_safe_empty_file(self):
        for f in self.EMPTY_FILES:
            cont = Container(f)
            with self.assertRaises(OSError):
                self.assertTrue(cont.is_safe)

    def test_is_safe_empty_directory(self):
        output_path = gf.tmp_directory()
        cont = Container(output_path)
        self.assertTrue(cont.is_safe)
        gf.delete_directory(output_path)

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

    def test_read_entry_not_existing(self):
        cont = Container(self.NOT_EXISTING)
        with self.assertRaises(TypeError):
            self.assertIsNone(cont.read_entry(self.EXPECTED_ENTRIES[0]))

    def test_read_entry_empty_file(self):
        for f in self.EMPTY_FILES:
            cont = Container(f)
            with self.assertRaises(OSError):
                self.assertIsNone(cont.read_entry(self.EXPECTED_ENTRIES[0]))

    def test_read_entry_empty_directory(self):
        output_path = gf.tmp_directory()
        cont = Container(output_path)
        self.assertIsNone(cont.read_entry(self.EXPECTED_ENTRIES[0]))
        gf.delete_directory(output_path)

    def test_read_entry_existing(self):
        entry = "config.txt"
        for key in self.FILES:
            f = self.FILES[key]
            cont = Container(f["path"])
            result = cont.read_entry(entry)
            self.assertIsNotNone(result)
            self.assertEqual(len(result), f["config_size"])

    def test_find_entry_not_existing(self):
        cont = Container(self.NOT_EXISTING)
        with self.assertRaises(TypeError):
            self.assertIsNone(cont.find_entry(self.EXPECTED_ENTRIES[0]))

    def test_find_entry_empty_file(self):
        for f in self.EMPTY_FILES:
            cont = Container(f)
            with self.assertRaises(OSError):
                self.assertIsNone(cont.find_entry(self.EXPECTED_ENTRIES[0]))

    def test_find_entry_empty_directory(self):
        output_path = gf.tmp_directory()
        cont = Container(output_path)
        self.assertIsNone(cont.find_entry(self.EXPECTED_ENTRIES[0]))
        gf.delete_directory(output_path)

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
            self.assertIsNone(result)

    def test_find_entry_missing(self):
        entry = "config_not_existing.txt"
        for key in self.FILES:
            f = self.FILES[key]
            cont = Container(f["path"])
            self.assertFalse(cont.find_entry(entry, exact=True))
            self.assertFalse(cont.find_entry(entry, exact=False))

    def test_decompress(self):
        for key in self.FILES:
            output_path = gf.tmp_directory()
            f = self.FILES[key]
            cont = Container(f["path"])
            cont.decompress(output_path)
            copy = Container(output_path, ContainerFormat.UNPACKED)
            self.assertEqual(copy.entries, self.EXPECTED_ENTRIES)
            gf.delete_directory(output_path)

    def test_compress_unpacked(self):
        input_path = self.FILES["unpacked"]["path"]
        output_path = gf.tmp_directory()
        cont = Container(output_path, ContainerFormat.UNPACKED)
        cont.compress(input_path)
        self.assertFalse(os.path.isfile(output_path))
        copy = Container(output_path, ContainerFormat.UNPACKED)
        self.assertEqual(copy.entries, self.EXPECTED_ENTRIES)
        gf.delete_directory(output_path)

    def test_compress_file(self):
        input_path = self.FILES["unpacked"]["path"]
        for key in self.FILES:
            fmt = self.FILES[key]["format"]
            if fmt != ContainerFormat.UNPACKED:
                handler, output_path = gf.tmp_file(suffix="." + fmt)
                cont = Container(output_path, fmt)
                cont.compress(input_path)
                self.assertTrue(os.path.isfile(output_path))
                copy = Container(output_path, fmt)
                self.assertEqual(copy.entries, self.EXPECTED_ENTRIES)
                gf.delete_file(handler, output_path)


if __name__ == "__main__":
    unittest.main()
