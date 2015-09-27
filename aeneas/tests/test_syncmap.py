#!/usr/bin/env python
# coding=utf-8

import tempfile
import unittest

from . import get_abs_path, delete_file

import aeneas.globalconstants as gc
from aeneas.language import Language
from aeneas.syncmap import SyncMap, SyncMapFormat

class TestSyncMap(unittest.TestCase):

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
        result = syn.read(fmt, get_abs_path(path + fmt), parameters=parameters)
        return (syn, result)

    def write(self, fmt, multiline=False, utf8=False, parameters=PARAMETERS):
        suffix = "." + fmt
        syn, result = self.read(SyncMapFormat.XML, multiline, utf8, self.PARAMETERS)
        handler, output_file_path = tempfile.mkstemp(suffix=suffix)
        result = syn.write(fmt, output_file_path, parameters)
        delete_file(handler, output_file_path)
        return result

    def test_constructor(self):
        syn = SyncMap()
        self.assertEqual(len(syn), 0)

    def test_read(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            syn, result = self.read(fmt)
            self.assertTrue(result)
            self.assertEqual(len(syn), 15)

    def test_read_m(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            syn, result = self.read(fmt, multiline=True)
            self.assertTrue(result)
            self.assertEqual(len(syn), 15)

    def test_read_u(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            syn, result = self.read(fmt, utf8=True)
            self.assertTrue(result)
            self.assertEqual(len(syn), 15)

    def test_read_mu(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            syn, result = self.read(fmt, multiline=True, utf8=True)
            self.assertTrue(result)
            self.assertEqual(len(syn), 15)

    def test_write(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            result = self.write(fmt)
            self.assertTrue(result)

    def test_write_m(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            result = self.write(fmt, multiline=True)
            self.assertTrue(result)

    def test_write_u(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            result = self.write(fmt, utf8=True)
            self.assertTrue(result)

    def test_write_mu(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            result = self.write(fmt, multiline=True, utf8=True)
            self.assertTrue(result)

    def test_write_smil_no_both(self):
        fmt = SyncMapFormat.SMIL
        result = self.write(fmt, parameters=None)
        self.assertFalse(result)

    def test_write_smil_no_page(self):
        fmt = SyncMapFormat.SMIL
        parameters = {gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF: "sonnet001.mp3"}
        result = self.write(fmt, parameters=parameters)
        self.assertFalse(result)

    def test_write_smil_no_audio(self):
        fmt = SyncMapFormat.SMIL
        parameters = {gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF: "sonnet001.xhtml"}
        result = self.write(fmt, parameters=parameters)
        self.assertFalse(result)

    def test_write_ttml_no_language(self):
        fmt = SyncMapFormat.TTML
        result = self.write(fmt, parameters=None)
        self.assertTrue(result)

    def test_write_ttml_language(self):
        fmt = SyncMapFormat.TTML
        parameters = {"language": Language.EN}
        result = self.write(fmt, parameters=parameters)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()



