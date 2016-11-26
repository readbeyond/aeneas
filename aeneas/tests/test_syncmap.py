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
from aeneas.tree import Tree
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

    def test_constructor_none(self):
        syn = SyncMap(tree=None)
        self.assertEqual(len(syn), 0)

    def test_constructor_invalid(self):
        with self.assertRaises(TypeError):
            syn = SyncMap(tree=[])

    def test_fragments_tree_not_given(self):
        syn = SyncMap()
        self.assertEqual(len(syn.fragments_tree), 0)

    def test_fragments_tree_empty(self):
        tree = Tree()
        syn = SyncMap(tree=tree)
        self.assertEqual(len(syn.fragments_tree), 0)

    def test_fragments_tree_not_empty(self):
        smf = SyncMapFragment()
        child = Tree(value=smf)
        tree = Tree()
        tree.add_child(child)
        syn = SyncMap(tree=tree)
        self.assertEqual(len(syn.fragments_tree), 1)

    def test_is_single_level_true_empty(self):
        syn = SyncMap()
        self.assertTrue(syn.is_single_level)

    def test_is_single_level_true_not_empty(self):
        smf = SyncMapFragment()
        child = Tree(value=smf)
        tree = Tree()
        tree.add_child(child)
        syn = SyncMap(tree=tree)
        self.assertTrue(syn.is_single_level)

    def test_is_single_level_false(self):
        smf2 = SyncMapFragment()
        child2 = Tree(value=smf2)
        smf = SyncMapFragment()
        child = Tree(value=smf)
        child.add_child(child2)
        tree = Tree()
        tree.add_child(child)
        syn = SyncMap(tree=tree)
        self.assertFalse(syn.is_single_level)

    def test_fragments_empty(self):
        syn = SyncMap()
        self.assertEqual(len(syn.fragments), 0)

    def test_fragments(self):
        syn = self.read("txt")
        self.assertTrue(len(syn.fragments) > 0)

    def test_leaves_empty(self):
        syn = SyncMap()
        self.assertEqual(len(syn.leaves()), 0)

    def test_leaves(self):
        syn = self.read("txt")
        self.assertTrue(len(syn.leaves()) > 0)

    def test_json_string(self):
        syn = self.read("txt")
        self.assertTrue(len(syn.json_string) > 0)

    def test_clear(self):
        syn = self.read("txt")
        self.assertEqual(len(syn), 15)
        syn.clear()
        self.assertEqual(len(syn), 0)

    def test_clone(self):
        syn = self.read("txt")
        text_first_fragment = syn.fragments[0].text
        syn2 = syn.clone()
        syn2.fragments[0].text_fragment.lines = [u"foo"]
        text_first_fragment2 = syn2.fragments[0].text
        self.assertEqual(syn.fragments[0].text, text_first_fragment)
        self.assertNotEqual(syn2.fragments[0].text, text_first_fragment)
        self.assertEqual(syn2.fragments[0].text, text_first_fragment2)

    def test_has_adjacent_leaves_only_empty(self):
        syn = SyncMap()
        self.assertTrue(syn.has_adjacent_leaves_only)

    def test_has_adjacent_leaves_only_not_empty(self):
        syn = self.read("txt")
        self.assertTrue(syn.has_adjacent_leaves_only)

    def test_has_adjacent_leaves_only(self):
        params = [
            ([("0.000", "0.000"), ("0.000", "0.000")], True),
            ([("0.000", "0.000"), ("0.000", "1.000")], True),
            ([("0.000", "1.000"), ("1.000", "1.000")], True),
            ([("0.000", "1.000"), ("1.000", "2.000")], True),
            ([("0.000", "0.000"), ("1.000", "1.000")], False),
            ([("0.000", "0.000"), ("1.000", "2.000")], False),
            ([("0.000", "1.000"), ("2.000", "2.000")], False),
            ([("0.000", "1.000"), ("2.000", "3.000")], False),
        ]
        for l, exp in params:
            tree = Tree()
            for b, e in l:
                interval = TimeInterval(begin=TimeValue(b), end=TimeValue(e))
                smf = SyncMapFragment(interval=interval)
                child = Tree(value=smf)
                tree.add_child(child, as_last=True)
            syn = SyncMap(tree=tree)
            self.assertEqual(syn.has_adjacent_leaves_only, exp)

    def test_has_zero_length_leaves_empty(self):
        syn = SyncMap()
        self.assertFalse(syn.has_zero_length_leaves)

    def test_has_zero_length_leaves_not_empty(self):
        syn = self.read("txt")
        self.assertFalse(syn.has_zero_length_leaves)

    def test_has_zero_length_leaves(self):
        params = [
            ([("0.000", "0.000"), ("0.000", "0.000")], True),
            ([("0.000", "0.000"), ("0.000", "1.000")], True),
            ([("0.000", "1.000"), ("1.000", "1.000")], True),
            ([("0.000", "1.000"), ("1.000", "2.000")], False),
            ([("0.000", "0.000"), ("1.000", "1.000")], True),
            ([("0.000", "0.000"), ("1.000", "2.000")], True),
            ([("0.000", "1.000"), ("2.000", "2.000")], True),
            ([("0.000", "1.000"), ("2.000", "3.000")], False),
        ]
        for l, exp in params:
            tree = Tree()
            for b, e in l:
                interval = TimeInterval(begin=TimeValue(b), end=TimeValue(e))
                smf = SyncMapFragment(interval=interval)
                child = Tree(value=smf)
                tree.add_child(child, as_last=True)
            syn = SyncMap(tree=tree)
            self.assertEqual(syn.has_zero_length_leaves, exp)

    def test_leaves_are_consistent_empty(self):
        syn = SyncMap()
        self.assertTrue(syn.leaves_are_consistent)

    def test_leaves_are_consistent_not_empty(self):
        syn = self.read("txt")
        self.assertTrue(syn.leaves_are_consistent)

    def test_leaves_are_consistent(self):
        params = [
            ([("0.000", "0.000"), ("0.000", "0.000")], True),
            ([("0.000", "0.000"), ("0.000", "1.000")], True),
            ([("0.000", "1.000"), ("1.000", "1.000")], True),
            ([("0.000", "1.000"), ("1.000", "2.000")], True),
            ([("0.000", "0.000"), ("1.000", "1.000")], True),
            ([("0.000", "0.000"), ("1.000", "2.000")], True),
            ([("0.000", "1.000"), ("2.000", "2.000")], True),
            ([("0.000", "1.000"), ("2.000", "3.000")], True),
            ([("0.000", "1.000"), ("1.000", "1.000"), ("1.000", "2.000")], True),
            ([("0.000", "1.000"), ("1.000", "1.000"), ("2.000", "2.000")], True),
            ([("0.000", "1.000"), ("2.000", "3.000"), ("1.500", "1.500")], True),
            ([("0.000", "1.000"), ("2.000", "3.000"), ("1.500", "1.750")], True),
            ([("0.000", "1.000"), ("1.040", "2.000")], True),
            ([("0.000", "1.000"), ("0.000", "0.500")], False),
            ([("0.000", "1.000"), ("0.000", "1.000")], False),
            ([("0.000", "1.000"), ("0.000", "1.500")], False),
            ([("0.000", "1.000"), ("0.500", "0.500")], False),
            ([("0.000", "1.000"), ("0.500", "0.750")], False),
            ([("0.000", "1.000"), ("0.500", "1.000")], False),
            ([("0.000", "1.000"), ("0.500", "1.500")], False),
            ([("0.000", "1.000"), ("2.000", "2.000"), ("1.500", "2.500")], False),
            ([("0.000", "1.000"), ("2.000", "3.000"), ("1.500", "2.500")], False),
            ([("0.000", "1.000"), ("0.960", "2.000")], False),
        ]
        for l, exp in params:
            tree = Tree()
            for b, e in l:
                interval = TimeInterval(begin=TimeValue(b), end=TimeValue(e))
                smf = SyncMapFragment(interval=interval)
                child = Tree(value=smf)
                tree.add_child(child, as_last=True)
            syn = SyncMap(tree=tree)
            self.assertEqual(syn.leaves_are_consistent, exp)

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
