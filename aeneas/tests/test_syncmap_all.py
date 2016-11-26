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
from aeneas.exacttiming import TimeInterval
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


if __name__ == "__main__":
    unittest.main()
