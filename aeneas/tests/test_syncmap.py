#!/usr/bin/env python
# coding=utf-8

import os
import sys
import tempfile
import unittest

from . import get_abs_path

from aeneas.globalconstants import PPN_TASK_OS_FILE_SMIL_PAGE_REF
from aeneas.globalconstants import PPN_TASK_OS_FILE_SMIL_AUDIO_REF
from aeneas.syncmap import SyncMap, SyncMapFormat, SyncMapFragment
from aeneas.textfile import TextFile, TextFileFormat

class TestSyncMap(unittest.TestCase):

    def load(self, path=None, lines=None):
        syn = SyncMap()
        if path == None:
            path = "res/inputtext/sonnet_parsed.txt"
            lines = 15
        tfl = TextFile(get_abs_path(path), TextFileFormat.PARSED)
        self.assertEqual(len(tfl), lines)
        i = 0
        for fragment in tfl.fragments:
            # dummy time values!
            syn_frag = SyncMapFragment(fragment, i, i + 1)
            syn.append(syn_frag)
            i += 1
        return syn

    def test_constructor(self):
        syn = SyncMap()
        self.assertEqual(len(syn), 0)

    def test_add_fragments(self):
        syn = self.load() 
        self.assertEqual(len(syn), 15)

    def test_output_csv(self):
        syn = self.load()
        handler, output_file_path = tempfile.mkstemp(suffix=".csv")
        result = syn.output(SyncMapFormat.CSV, output_file_path)
        self.assertTrue(result)
        self.assertTrue(os.path.isfile(output_file_path))
        #print output_file_path
        os.close(handler)
        os.remove(output_file_path)

    def test_output_json(self):
        syn = self.load()
        handler, output_file_path = tempfile.mkstemp(suffix=".js")
        result = syn.output(SyncMapFormat.JSON, output_file_path)
        self.assertTrue(result)
        self.assertTrue(os.path.isfile(output_file_path))
        #print output_file_path
        os.close(handler)
        os.remove(output_file_path)

    def test_output_smil(self):
        syn = self.load()
        handler, output_file_path = tempfile.mkstemp(suffix=".smil")
        parameters = dict()
        parameters[PPN_TASK_OS_FILE_SMIL_PAGE_REF] = "p001.xhtml"
        parameters[PPN_TASK_OS_FILE_SMIL_AUDIO_REF] = "../Audio/p001.mp3"
        result = syn.output(SyncMapFormat.SMIL, output_file_path, parameters)
        self.assertTrue(result)
        self.assertTrue(os.path.isfile(output_file_path))
        #print output_file_path
        os.close(handler)
        os.remove(output_file_path)

    def test_output_smil_fail_1(self):
        syn = self.load()
        handler, output_file_path = tempfile.mkstemp(suffix=".smil")
        parameters = dict()
        parameters[PPN_TASK_OS_FILE_SMIL_PAGE_REF] = "p001.xhtml"
        result = syn.output(SyncMapFormat.SMIL, output_file_path, parameters)
        self.assertFalse(result)
        #print output_file_path
        os.close(handler)
        os.remove(output_file_path)

    def test_output_smil_fail_2(self):
        syn = self.load()
        handler, output_file_path = tempfile.mkstemp(suffix=".smil")
        parameters = dict()
        parameters[PPN_TASK_OS_FILE_SMIL_AUDIO_REF] = "../Audio/p001.mp3"
        result = syn.output(SyncMapFormat.SMIL, output_file_path, parameters)
        self.assertFalse(result)
        #print output_file_path
        os.close(handler)
        os.remove(output_file_path)

    def test_output_srt(self):
        syn = self.load()
        handler, output_file_path = tempfile.mkstemp(suffix=".srt")
        result = syn.output(SyncMapFormat.SRT, output_file_path)
        self.assertTrue(result)
        self.assertTrue(os.path.isfile(output_file_path))
        #print output_file_path
        os.close(handler)
        os.remove(output_file_path)

    def test_output_ttml(self):
        syn = self.load()
        handler, output_file_path = tempfile.mkstemp(suffix=".ttml")
        result = syn.output(SyncMapFormat.TTML, output_file_path)
        self.assertTrue(result)
        self.assertTrue(os.path.isfile(output_file_path))
        #print output_file_path
        os.close(handler)
        os.remove(output_file_path)

    def test_output_txt(self):
        syn = self.load()
        handler, output_file_path = tempfile.mkstemp(suffix=".txt")
        result = syn.output(SyncMapFormat.TXT, output_file_path)
        self.assertTrue(result)
        self.assertTrue(os.path.isfile(output_file_path))
        #print output_file_path
        os.close(handler)
        os.remove(output_file_path)

    def test_output_vtt(self):
        syn = self.load()
        handler, output_file_path = tempfile.mkstemp(suffix=".vtt")
        result = syn.output(SyncMapFormat.VTT, output_file_path)
        self.assertTrue(result)
        self.assertTrue(os.path.isfile(output_file_path))
        #print output_file_path
        os.close(handler)
        os.remove(output_file_path)

    def test_output_xml(self):
        syn = self.load()
        handler, output_file_path = tempfile.mkstemp(suffix=".xml")
        result = syn.output(SyncMapFormat.XML, output_file_path)
        self.assertTrue(result)
        self.assertTrue(os.path.isfile(output_file_path))
        #print output_file_path
        os.close(handler)
        os.remove(output_file_path)

    def test_output_csv_unicode(self):
        syn = self.load("res/example_jobs/example7/OEBPS/Resources/de.txt", 24)
        handler, output_file_path = tempfile.mkstemp(suffix=".csv")
        result = syn.output(SyncMapFormat.CSV, output_file_path)
        self.assertTrue(result)
        self.assertTrue(os.path.isfile(output_file_path))
        #print output_file_path
        os.close(handler)
        os.remove(output_file_path)

    def test_output_ttml_unicode(self):
        syn = self.load("res/example_jobs/example7/OEBPS/Resources/de.txt", 24)
        handler, output_file_path = tempfile.mkstemp(suffix=".ttml")
        result = syn.output(SyncMapFormat.TTML, output_file_path)
        self.assertTrue(result)
        self.assertTrue(os.path.isfile(output_file_path))
        #print output_file_path
        os.close(handler)
        os.remove(output_file_path)

    def test_output_txt_unicode(self):
        syn = self.load("res/example_jobs/example7/OEBPS/Resources/de.txt", 24)
        handler, output_file_path = tempfile.mkstemp(suffix=".txt")
        result = syn.output(SyncMapFormat.TXT, output_file_path)
        self.assertTrue(result)
        self.assertTrue(os.path.isfile(output_file_path))
        #print output_file_path
        os.close(handler)
        os.remove(output_file_path)

if __name__ == '__main__':
    unittest.main()



