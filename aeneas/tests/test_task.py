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

from aeneas.adjustboundaryalgorithm import AdjustBoundaryAlgorithm
from aeneas.exacttiming import TimeValue
from aeneas.idsortingalgorithm import IDSortingAlgorithm
from aeneas.language import Language
from aeneas.logger import Logger
from aeneas.syncmap import SyncMap
from aeneas.syncmap import SyncMapFormat
from aeneas.syncmap import SyncMapFragment
from aeneas.syncmap import SyncMapHeadTailFormat
from aeneas.task import Task
from aeneas.task import TaskConfiguration
from aeneas.textfile import TextFileFormat
from aeneas.textfile import TextFragment
import aeneas.globalfunctions as gf


class TestTask(unittest.TestCase):

    def dummy_sync_map(self):
        sync_map = SyncMap()
        frag = TextFragment(u"f001", Language.ENG, [u"Fragment 1"])
        sync_map.add_fragment(SyncMapFragment(text_fragment=frag, begin=TimeValue("0.000"), end=TimeValue("12.345")))
        frag = TextFragment(u"f002", Language.ENG, [u"Fragment 2"])
        sync_map.add_fragment(SyncMapFragment(text_fragment=frag, begin=TimeValue("12.345"), end=TimeValue("23.456")))
        frag = TextFragment(u"f003", Language.ENG, [u"Fragment 3"])
        sync_map.add_fragment(SyncMapFragment(text_fragment=frag, begin=TimeValue("23.456"), end=TimeValue("34.567")))
        return sync_map

    def setter(self, attribute, value, expected):
        taskconf = TaskConfiguration()
        taskconf[attribute] = value
        read_value = taskconf[attribute]
        if expected is None:
            self.assertIsNone(read_value)
        else:
            self.assertEqual(read_value, expected)

    def set_text_file(self, path, fmt, expected, id_regex=None, class_regex=None, id_sort=None):
        task = Task()
        task.configuration = TaskConfiguration()
        task.configuration["language"] = Language.ENG
        task.configuration["i_t_format"] = fmt
        if class_regex is not None:
            task.configuration["i_t_unparsed_class_regex"] = class_regex
        if id_regex is not None:
            task.configuration["i_t_unparsed_id_regex"] = id_regex
        if id_sort is not None:
            task.configuration["i_t_unparsed_id_sort"] = id_sort
        task.text_file_path_absolute = gf.absolute_path(path, __file__)
        self.assertIsNotNone(task.text_file)
        self.assertEqual(len(task.text_file), expected)

    def tc_from_string(self, config_string, properties):
        taskconf = TaskConfiguration(config_string)
        for prop, value in properties:
            read_value = taskconf[prop]
            if value is None:
                self.assertIsNone(read_value)
            else:
                self.assertEqual(read_value, value)

    def tc_from_string_some_invalid(self, config_string, expected_description):
        properties = [
            ["language", Language.ITA],
            ["description", expected_description],
        ]
        self.tc_from_string(config_string, properties)

    def test_task_logger(self):
        logger = Logger()
        task = Task(logger=logger)

    def test_task_identifier(self):
        task = Task()
        self.assertEqual(len(task.identifier), 36)

    def test_task_empty_configuration(self):
        task = Task()
        self.assertIsNone(task.configuration)

    def test_task_string_configuration_invalid(self):
        with self.assertRaises(TypeError):
            task = Task(1)

    def test_task_string_configuration_str(self):
        with self.assertRaises(TypeError):
            task = Task(b"task_language=en")

    def test_task_string_configuration_unicode(self):
        task = Task(u"task_language=en")
        self.assertIsNotNone(task.configuration)

    def test_task_set_configuration(self):
        task = Task()
        taskconf = TaskConfiguration()
        task.configuration = taskconf
        self.assertIsNotNone(task.configuration)

    def test_task_empty_on_creation(self):
        task = Task()
        self.assertIsNone(task.configuration)
        self.assertIsNone(task.text_file)
        self.assertIsNone(task.audio_file)
        self.assertIsNone(task.sync_map)
        self.assertIsNone(task.audio_file_path)
        self.assertIsNone(task.audio_file_path_absolute)
        self.assertIsNone(task.text_file_path)
        self.assertIsNone(task.text_file_path_absolute)
        self.assertIsNone(task.sync_map_file_path)
        self.assertIsNone(task.sync_map_file_path_absolute)

    def test_task_sync_map_leaves_empty(self):
        task = Task()
        self.assertEqual(len(task.sync_map_leaves()), 0)

    def test_set_audio_file_path_absolute(self):
        task = Task()
        task.audio_file_path_absolute = gf.absolute_path("res/container/job/assets/p001.mp3", __file__)
        self.assertIsNotNone(task.audio_file)
        self.assertEqual(task.audio_file.file_size, 426735)
        self.assertAlmostEqual(task.audio_file.audio_length, TimeValue("53.3"), places=1)

    def test_set_audio_file_path_absolute_error(self):
        task = Task()
        with self.assertRaises(OSError):
            task.audio_file_path_absolute = gf.absolute_path("not_existing.mp3", __file__)

    def test_set_text_file_unparsed_id(self):
        self.set_text_file(
            "res/inputtext/sonnet_unparsed_id.xhtml",
            TextFileFormat.UNPARSED,
            15,
            id_regex=u"f[0-9]+",
            id_sort=IDSortingAlgorithm.NUMERIC
        )

    def test_set_text_file_unparsed_class(self):
        # NOTE this test fails because there are no id attributes in the html file
        self.set_text_file(
            "res/inputtext/sonnet_unparsed_class.xhtml",
            TextFileFormat.UNPARSED,
            0,
            class_regex=u"ra",
            id_sort=IDSortingAlgorithm.NUMERIC
        )

    def test_set_text_file_unparsed_id_class(self):
        self.set_text_file(
            "res/inputtext/sonnet_unparsed_class_id.xhtml",
            TextFileFormat.UNPARSED,
            15,
            id_regex=u"f[0-9]+",
            class_regex=u"ra",
            id_sort=IDSortingAlgorithm.NUMERIC
        )

    def test_set_text_file_unparsed_id_class_empty(self):
        # NOTE this test fails because there are no id attributes in the html file
        self.set_text_file(
            "res/inputtext/sonnet_unparsed_class.xhtml",
            TextFileFormat.UNPARSED,
            0,
            id_regex=u"f[0-9]+",
            class_regex=u"ra",
            id_sort=IDSortingAlgorithm.NUMERIC
        )

    def test_set_text_file_plain(self):
        self.set_text_file(
            "res/inputtext/sonnet_plain.txt",
            TextFileFormat.PLAIN,
            15
        )

    def test_set_text_file_parsed(self):
        return
        self.set_text_file(
            "res/inputtext/sonnet_parsed.txt",
            TextFileFormat.PARSED,
            15
        )

    def test_set_text_file_subtitles(self):
        return
        self.set_text_file(
            "res/inputtext/sonnet_subtitles.txt",
            TextFileFormat.SUBTITLES,
            15
        )

    def test_output_sync_map(self):
        task = Task()
        task.configuration = TaskConfiguration()
        task.configuration["language"] = Language.ENG
        task.configuration["o_format"] = SyncMapFormat.TXT
        task.sync_map = self.dummy_sync_map()
        handler, output_file_path = gf.tmp_file(suffix=".txt")
        task.sync_map_file_path_absolute = output_file_path
        path = task.output_sync_map_file()
        self.assertIsNotNone(path)
        self.assertEqual(path, output_file_path)
        gf.delete_file(handler, output_file_path)

    def test_task_sync_map_leaves(self):
        task = Task()
        task.configuration = TaskConfiguration()
        task.configuration["language"] = Language.ENG
        task.configuration["o_format"] = SyncMapFormat.TXT
        task.sync_map = self.dummy_sync_map()
        self.assertEqual(len(task.sync_map_leaves()), 3)

    def test_tc_custom_id(self):
        self.setter("custom_id", u"customid", u"customid")

    def test_tc_description_unicode_ascii(self):
        self.setter("description", u"Test description", u"Test description")

    def test_tc_description_unicode_unicode(self):
        self.setter("description", u"Test àèìòù", u"Test àèìòù")

    def test_tc_language(self):
        self.setter("language", Language.ITA, Language.ITA)

    def test_tc_adjust_boundary_algorithm(self):
        self.setter("aba_algorithm", AdjustBoundaryAlgorithm.AUTO, AdjustBoundaryAlgorithm.AUTO)

    def test_tc_adjust_boundary_aftercurrent_value(self):
        self.setter("aba_aftercurrent_value", u"0.100", TimeValue("0.100"))

    def test_tc_adjust_boundary_beforenext_value(self):
        self.setter("aba_beforenext_value", u"0.100", TimeValue("0.100"))

    def test_tc_adjust_boundary_no_zero(self):
        self.setter("aba_no_zero", True, True)

    def test_tc_adjust_boundary_offset_value(self):
        self.setter("aba_offset_value", u"0.100", TimeValue("0.100"))

    def test_tc_adjust_boundary_percent_value(self):
        self.setter("aba_percent_value", u"75", 75)

    def test_tc_adjust_boundary_rate_value(self):
        self.setter("aba_rate_value", u"22.5", 22.5)

    def test_tc_adjust_boundary_nonspeech_min(self):
        self.setter("aba_nonspeech_min", 1.000, 1.000)

    def test_tc_adjust_boundary_nonspeech_string(self):
        self.setter("aba_nonspeech_string", "REMOVE", "REMOVE")

    def test_tc_is_audio_file_detect_head_max(self):
        self.setter("i_a_head_max", u"10.000", 10.0)

    def test_tc_is_audio_file_detect_head_min(self):
        self.setter("i_a_head_min", u"1.000", 1.0)

    def test_tc_is_audio_file_detect_tail_max(self):
        self.setter("i_a_tail_max", u"5.000", 5.0)

    def test_tc_is_audio_file_detect_tail_min(self):
        self.setter("i_a_tail_min", u"1.000", 1.0)

    def test_tc_is_audio_file_head_length(self):
        self.setter("i_a_head", u"20", 20.0)

    def test_tc_is_audio_file_process_length(self):
        self.setter("i_a_process", u"100", 100.0)

    def test_tc_is_audio_file_tail_length(self):
        self.setter("i_a_tail", u"20", 20.)

    def test_tc_is_text_file_format(self):
        self.setter("i_t_format", TextFileFormat.PLAIN, TextFileFormat.PLAIN)

    def test_tc_is_text_ignore_regex(self):
        self.setter("i_t_ignore_regex", u"\[.*\]", u"\[.*\]")

    def test_tc_is_text_transliterate_map(self):
        self.setter("i_t_transliterate_map", u"/tmp/map.txt", u"/tmp/map.txt")

    def test_tc_is_text_unparsed_class_regex(self):
        self.setter("i_t_unparsed_class_regex", u"f[0-9]*", u"f[0-9]*")

    def test_tc_is_text_unparsed_id_regex(self):
        self.setter("i_t_unparsed_id_regex", u"ra", u"ra")

    def test_tc_is_text_unparsed_id_sort(self):
        self.setter("i_t_unparsed_id_sort", IDSortingAlgorithm.NUMERIC, IDSortingAlgorithm.NUMERIC)

    def test_tc_os_file_format(self):
        self.setter("o_format", SyncMapFormat.SMIL, SyncMapFormat.SMIL)

    def test_tc_os_file_head_tail_format(self):
        self.setter("o_h_t_format", SyncMapHeadTailFormat.ADD, SyncMapHeadTailFormat.ADD)

    def test_tc_os_file_name(self):
        self.setter("o_name", u"output.smil", u"output.smil")

    def test_tc_os_file_smil_audio_ref(self):
        self.setter("o_smil_audio_ref", u"../audio/audio001.mp3", u"../audio/audio001.mp3")

    def test_tc_os_file_smil_page_ref(self):
        self.setter("o_smil_page_ref", u"../text/page001.xhtml", u"../text/page001.xhtml")

    def test_tc_config_string(self):
        taskconf = TaskConfiguration()
        taskconf["language"] = Language.ITA
        taskconf["description"] = u"Test description"
        taskconf["custom_id"] = u"customid"
        taskconf["i_a_head"] = u"20"
        taskconf["i_a_process"] = u"100"
        taskconf["o_format"] = SyncMapFormat.SMIL
        taskconf["o_name"] = u"output.smil"
        taskconf["o_smil_audio_ref"] = u"../audio/audio001.mp3"
        taskconf["o_smil_page_ref"] = u"../text/page001.xhtml"
        expected = u"is_audio_file_head_length=20|is_audio_file_process_length=100|os_task_file_format=smil|os_task_file_name=output.smil|os_task_file_smil_audio_ref=../audio/audio001.mp3|os_task_file_smil_page_ref=../text/page001.xhtml|task_custom_id=customid|task_description=Test description|task_language=ita"
        self.assertEqual(taskconf.config_string, expected)

    def test_tc_from_string_with_optional(self):
        config_string = u"task_description=Test description|task_language=ita|task_custom_id=customid|is_audio_file_head_length=20|is_audio_file_process_length=100|os_task_file_format=smil|os_task_file_name=output.smil|os_task_file_smil_audio_ref=../audio/audio001.mp3|os_task_file_smil_page_ref=../text/page001.xhtml"
        properties = [
            ("language", Language.ITA),
            ("description", u"Test description"),
            ("custom_id", u"customid"),
            ("i_a_head", 20.0),
            ("i_a_process", 100.0),
            ("o_format", SyncMapFormat.SMIL),
            ("o_name", u"output.smil"),
            ("o_smil_audio_ref", u"../audio/audio001.mp3"),
            ("o_smil_page_ref", u"../text/page001.xhtml"),
        ]
        self.tc_from_string(config_string, properties)

    def test_tc_from_string_no_optional(self):
        config_string = u"task_description=Test description|task_language=ita|task_custom_id=customid|is_audio_file_head_length=20|is_audio_file_process_length=100|os_task_file_format=txt|os_task_file_name=output.txt"
        properties = [
            ("language", Language.ITA),
            ("description", u"Test description"),
            ("custom_id", u"customid"),
            ("i_a_head", 20.0),
            ("i_a_process", 100.0),
            ("o_format", SyncMapFormat.TXT),
            ("o_name", u"output.txt"),
            ("o_smil_audio_ref", None),
            ("o_smil_page_ref", None),
        ]
        self.tc_from_string(config_string, properties)

    def test_tc_from_string_simple(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description|task_language=ita",
            u"Test description"
        )

    def test_tc_from_string_unicode(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description àèìòù|task_language=ita",
            u"Test description àèìòù"
        )

    def test_tc_from_string_repeated_pipes(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description|||task_language=ita",
            u"Test description"
        )

    def test_tc_from_string_invalid_key_with_value(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description|not_a_valid_key=foo|task_language=ita",
            u"Test description"
        )

    def test_tc_from_string_invalid_key_no_value(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description|not_a_valid_key=|task_language=ita",
            u"Test description"
        )

    def test_tc_from_string_value_without_key(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description|=foo|task_language=ita",
            u"Test description"
        )

    def test_tc_from_string_trailing_pipe(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description|task_language=ita|",
            u"Test description"
        )

    def test_tc_from_string_leading_pipe(self):
        self.tc_from_string_some_invalid(
            u"|task_description=Test description|task_language=ita",
            u"Test description"
        )

    def test_tc_from_string_leading_and_trailing_pipe(self):
        self.tc_from_string_some_invalid(
            u"|task_description=Test description|task_language=ita|",
            u"Test description"
        )

    def test_tc_from_string_valid_key_no_value(self):
        self.tc_from_string_some_invalid(
            u"task_description=|task_language=ita",
            None
        )

    def test_tc_from_string_space_before_valid_key(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description=|task_language=ita",
            None
        )

    def test_tc_from_string_value_with_trailing_space(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test with space |task_language=ita",
            u"Test with space "
        )

    def test_tc_from_string_space_before_and_after_key_value_pair(self):
        self.tc_from_string_some_invalid(
            u" task_description=Test with space |task_language=ita",
            None
        )

    def test_tc_from_string_several_invalid_stuff(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description|foo=|=bar|foo=bar|||task_language=ita",
            u"Test description"
        )


if __name__ == "__main__":
    unittest.main()
