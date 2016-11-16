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

import unittest

from aeneas.exacttiming import Decimal
from aeneas.exacttiming import TimeValue
from aeneas.language import Language
from aeneas.syncmap import SyncMap
from aeneas.syncmap import SyncMapFormat
from aeneas.syncmap import SyncMapFragment
from aeneas.syncmap import SyncMapMissingParameterError
from aeneas.textfile import TextFragment
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf


class TestSyncMap(unittest.TestCase):

    NOT_EXISTING_SRT = gf.absolute_path("not_existing.srt", __file__)
    EXISTING_SRT = gf.absolute_path("res/syncmaps/sonnet001.srt", __file__)
    NOT_WRITEABLE_SRT = gf.absolute_path("x/y/z/not_writeable.srt", __file__)

    PARAMETERS = {
        gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF: "sonnet001.xhtml",
        gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF: "sonnet001.mp3",
        gc.PPN_SYNCMAP_LANGUAGE: Language.ENG,
    }

    def read(self, fmt, multiline=False, utf8=False, parameters=PARAMETERS):
        syn = SyncMap()
        if multiline and utf8:
            path = "res/syncmaps/sonnet001_mu."
        elif multiline:
            path = "res/syncmaps/sonnet001_m."
        elif utf8:
            path = "res/syncmaps/sonnet001_u."
        else:
            path = "res/syncmaps/sonnet001."
        syn.read(fmt, gf.absolute_path(path + fmt, __file__), parameters=parameters)
        return syn

    def write(self, fmt, multiline=False, utf8=False, parameters=PARAMETERS):
        suffix = "." + fmt
        syn = self.read(SyncMapFormat.XML, multiline, utf8, self.PARAMETERS)
        handler, output_file_path = gf.tmp_file(suffix=suffix)
        syn.write(fmt, output_file_path, parameters)
        gf.delete_file(handler, output_file_path)

    def test_constructor(self):
        syn = SyncMap()
        self.assertEqual(len(syn), 0)

    def test_fragment_constructor(self):
        frag = SyncMapFragment()
        self.assertEqual(frag.audio_duration, 0)
        self.assertEqual(frag.chars, 0)
        self.assertIsNone(frag.rate)

    def test_fragment_rate_none(self):
        text = TextFragment(lines=[u"Hello", u"World"])
        frag = SyncMapFragment(text_fragment=text)
        self.assertEqual(frag.audio_duration, 0)
        self.assertEqual(frag.chars, 10)
        self.assertIsNone(frag.rate)

    def test_fragment_rate_valid(self):
        text = TextFragment(lines=[u"Hello", u"World"])
        frag = SyncMapFragment(text_fragment=text, begin=TimeValue("1.234"), end=TimeValue("6.234"))
        self.assertEqual(frag.audio_duration, 5)
        self.assertEqual(frag.chars, 10)
        self.assertEqual(frag.rate, 2.000)
        self.assertEqual(frag.rate, Decimal("2.000"))

    def test_fragment_rate_zero(self):
        text = TextFragment(lines=[u"Hello", u"World"])
        frag = SyncMapFragment(text_fragment=text, begin=TimeValue("1.234"), end=TimeValue("1.234"))
        self.assertEqual(frag.audio_duration, 0)
        self.assertEqual(frag.chars, 10)
        self.assertIsNone(frag.rate, None)

    def test_append_none(self):
        syn = SyncMap()
        with self.assertRaises(TypeError):
            syn.add_fragment(None)

    def test_append_invalid_fragment(self):
        syn = SyncMap()
        with self.assertRaises(TypeError):
            syn.add_fragment("foo")

    def test_read_none(self):
        syn = SyncMap()
        with self.assertRaises(ValueError):
            syn.read(None, self.EXISTING_SRT)

    def test_read_invalid_format(self):
        syn = SyncMap()
        with self.assertRaises(ValueError):
            syn.read("foo", self.EXISTING_SRT)

    def test_read_not_existing_path(self):
        syn = SyncMap()
        with self.assertRaises(OSError):
            syn.read(SyncMapFormat.SRT, self.NOT_EXISTING_SRT)

    def test_read(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            syn = self.read(fmt)
            self.assertEqual(len(syn), 15)
            ignored = str(syn)

    def test_read_m(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            syn = self.read(fmt, multiline=True)
            self.assertEqual(len(syn), 15)
            ignored = str(syn)

    def test_read_u(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            syn = self.read(fmt, utf8=True)
            self.assertEqual(len(syn), 15)
            ignored = str(syn)

    def test_read_mu(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            syn = self.read(fmt, multiline=True, utf8=True)
            self.assertEqual(len(syn), 15)
            ignored = str(syn)

    def test_write_none(self):
        syn = SyncMap()
        with self.assertRaises(ValueError):
            syn.write(None, self.NOT_EXISTING_SRT)

    def test_write_invalid_format(self):
        syn = SyncMap()
        with self.assertRaises(ValueError):
            syn.write("foo", self.NOT_EXISTING_SRT)

    def test_write_not_existing_path(self):
        syn = SyncMap()
        with self.assertRaises(OSError):
            syn.write(SyncMapFormat.SRT, self.NOT_WRITEABLE_SRT)

    def test_write(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            self.write(fmt)

    def test_write_m(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            self.write(fmt, multiline=True)

    def test_write_u(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            self.write(fmt, utf8=True)

    def test_write_mu(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            self.write(fmt, multiline=True, utf8=True)

    def test_write_smil_no_both(self):
        fmt = SyncMapFormat.SMIL
        with self.assertRaises(SyncMapMissingParameterError):
            self.write(fmt, parameters=None)

    def test_write_smil_no_page(self):
        fmt = SyncMapFormat.SMIL
        parameters = {gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF: "sonnet001.mp3"}
        with self.assertRaises(SyncMapMissingParameterError):
            self.write(fmt, parameters=parameters)

    def test_write_smil_no_audio(self):
        fmt = SyncMapFormat.SMIL
        parameters = {gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF: "sonnet001.xhtml"}
        with self.assertRaises(SyncMapMissingParameterError):
            self.write(fmt, parameters=parameters)

    def test_write_ttml_no_language(self):
        fmt = SyncMapFormat.TTML
        self.write(fmt, parameters=None)

    def test_write_ttml_language(self):
        fmt = SyncMapFormat.TTML
        parameters = {gc.PPN_SYNCMAP_LANGUAGE: Language.ENG}
        self.write(fmt, parameters=parameters)

    def test_output_html_for_tuning(self):
        syn = self.read(SyncMapFormat.XML, multiline=True, utf8=True)
        handler, output_file_path = gf.tmp_file(suffix=".html")
        audio_file_path = "foo.mp3"
        syn.output_html_for_tuning(audio_file_path, output_file_path, None)
        gf.delete_file(handler, output_file_path)


if __name__ == "__main__":
    unittest.main()
