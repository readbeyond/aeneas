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

    def load(self, path=None, lines=None, fmt=TextFileFormat.PARSED):
        syn = SyncMap()
        if path is None:
            path = "res/inputtext/sonnet_parsed.txt"
            lines = 15
        tfl = TextFile(get_abs_path(path), fmt)
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

    def write_output(self, suffix, fmt, uni=False):
        if uni:
            syn = self.load("res/example_jobs/example7/OEBPS/Resources/de.txt", 24)
        else:
            syn = self.load()
        handler, output_file_path = tempfile.mkstemp(suffix=suffix)
        result = syn.output(fmt, output_file_path)
        self.assertTrue(result)
        self.assertTrue(os.path.isfile(output_file_path))
        #print output_file_path
        os.close(handler)
        os.remove(output_file_path)
    
    def write_multiline(self, suffix, fmt):
        syn = self.load("res/inputtext/sonnet_subtitles_multiple_rows.txt", 15, TextFileFormat.SUBTITLES)
        handler, output_file_path = tempfile.mkstemp(suffix=suffix)
        result = syn.output(fmt, output_file_path)
        self.assertTrue(result)
        self.assertTrue(os.path.isfile(output_file_path))
        #print output_file_path
        os.close(handler)
        os.remove(output_file_path)

    def test_output_csv(self):
        self.write_output(".csv", SyncMapFormat.CSV)

    def test_output_csv_unicode(self):
        self.write_output(".csv", SyncMapFormat.CSV, True)

    def test_output_csvh(self):
        self.write_output(".csvh", SyncMapFormat.CSVH)

    def test_output_csvh_unicode(self):
        self.write_output(".csvh", SyncMapFormat.CSVH, True)

    def test_output_json(self):
        self.write_output(".js", SyncMapFormat.JSON)

    #def test_output_json_unicode(self):
    #    self.write_output(".js", SyncMapFormat.JSON, True)

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
        self.write_output(".srt", SyncMapFormat.SRT)

    def test_output_srt_unicode(self):
        self.write_output(".srt", SyncMapFormat.SRT, True)

    def test_output_srt_multiline(self):
        self.write_multiline(".srt", SyncMapFormat.SRT)

    def test_output_ssv(self):
        self.write_output(".ssv", SyncMapFormat.SSV)

    def test_output_ssv_unicode(self):
        self.write_output(".ssv", SyncMapFormat.SSV, True)

    def test_output_ssvh(self):
        self.write_output(".ssvh", SyncMapFormat.SSVH)

    def test_output_ssvh_unicode(self):
        self.write_output(".ssvh", SyncMapFormat.SSVH, True)

    def test_output_tsv(self):
        self.write_output(".tsv", SyncMapFormat.TSV)

    def test_output_tsv_unicode(self):
        self.write_output(".tsv", SyncMapFormat.TSV, True)

    def test_output_tsvh(self):
        self.write_output(".tsvh", SyncMapFormat.TSVH)

    def test_output_tsvh_unicode(self):
        self.write_output(".tsvh", SyncMapFormat.TSVH, True)
    
    def test_output_ttml(self):
        self.write_output(".ttml", SyncMapFormat.TTML)

    def test_output_ttml_unicode(self):
        self.write_output(".ttml", SyncMapFormat.TTML, True)

    def test_output_ttml_multiline(self):
        self.write_multiline(".ttml", SyncMapFormat.TTML)

    def test_output_txt(self):
        self.write_output(".txt", SyncMapFormat.TXT)

    def test_output_txt_unicode(self):
        self.write_output(".txt", SyncMapFormat.TXT, True)

    def test_output_txth(self):
        self.write_output(".txth", SyncMapFormat.TXTH)

    def test_output_txth_unicode(self):
        self.write_output(".txth", SyncMapFormat.TXTH, True)

    def test_output_vtt(self):
        self.write_output(".vtt", SyncMapFormat.VTT)

    def test_output_vtt_unicode(self):
        self.write_output(".vtt", SyncMapFormat.VTT, True)

    def test_output_vtt_multiline(self):
        self.write_multiline(".vtt", SyncMapFormat.VTT)

    def test_output_xml(self):
        self.write_output(".xml", SyncMapFormat.XML)

    #def test_output_xml_unicode(self):
    #    self.write_output(".xml", SyncMapFormat.XML, True)


if __name__ == '__main__':
    unittest.main()



