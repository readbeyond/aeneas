#!/usr/bin/env python
# coding=utf-8

import tempfile
import unittest

from aeneas.language import Language
from aeneas.syncmap import SyncMap
from aeneas.syncmap import SyncMapFormat
from aeneas.syncmap import SyncMapMissingParameterError
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf
import aeneas.tests as at

class TestSyncMap(unittest.TestCase):

    NOT_EXISTING_SRT = at.get_abs_path("not_existing.srt")
    EXISTING_SRT = at.get_abs_path("res/syncmaps/sonnet001.srt")
    NOT_WRITEABLE_SRT = at.get_abs_path("x/y/z/not_writeable.srt")

    PARAMETERS = {
        gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF: "sonnet001.xhtml",
        gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF: "sonnet001.mp3",
        "language": Language.EN,
    }

    def read(self, fmt, multiline=False, utf8=False, parameters=PARAMETERS):
        syn = SyncMap()
        if multiline and unicode:
            path = "res/syncmaps/sonnet001_mu."
        elif multiline:
            path = "res/syncmaps/sonnet001_m."
        elif utf8:
            path = "res/syncmaps/sonnet001_u."
        else:
            path = "res/syncmaps/sonnet001."
        syn.read(fmt, at.get_abs_path(path + fmt), parameters=parameters)
        return syn

    def write(self, fmt, multiline=False, utf8=False, parameters=PARAMETERS):
        suffix = "." + fmt
        syn = self.read(SyncMapFormat.XML, multiline, utf8, self.PARAMETERS)
        handler, output_file_path = tempfile.mkstemp(suffix=suffix)
        syn.write(fmt, output_file_path, parameters)
        gf.delete_file(handler, output_file_path)

    def test_constructor(self):
        syn = SyncMap()
        self.assertEqual(len(syn), 0)

    def test_append_none(self):
        syn = SyncMap()
        with self.assertRaises(TypeError):
            syn.append_fragment(None)

    def test_append_invalid_fragment(self):
        syn = SyncMap()
        with self.assertRaises(TypeError):
            syn.append_fragment("foo")

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
        with self.assertRaises(IOError):
            syn.read(SyncMapFormat.SRT, self.NOT_EXISTING_SRT)

    def test_read(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            syn = self.read(fmt)
            self.assertEqual(len(syn), 15)

    def test_read_m(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            syn = self.read(fmt, multiline=True)
            self.assertEqual(len(syn), 15)

    def test_read_u(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            syn = self.read(fmt, utf8=True)
            self.assertEqual(len(syn), 15)

    def test_read_mu(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            syn = self.read(fmt, multiline=True, utf8=True)
            self.assertEqual(len(syn), 15)

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
        with self.assertRaises(IOError):
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
        parameters = {"language": Language.EN}
        self.write(fmt, parameters=parameters)

if __name__ == '__main__':
    unittest.main()



