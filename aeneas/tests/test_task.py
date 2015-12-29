#!/usr/bin/env python
# coding=utf-8

import unittest

from aeneas.adjustboundaryalgorithm import AdjustBoundaryAlgorithm
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
        frag = TextFragment(u"f001", Language.EN, [u"Fragment 1"])
        sync_map.append_fragment(SyncMapFragment(frag, 0, 12.345))
        frag = TextFragment(u"f002", Language.EN, [u"Fragment 2"])
        sync_map.append_fragment(SyncMapFragment(frag, 12.345, 23.456))
        frag = TextFragment(u"f003", Language.EN, [u"Fragment 3"])
        sync_map.append_fragment(SyncMapFragment(frag, 23.456, 34.567))
        return sync_map

    def setter(self, attribute, value):
        taskconf = TaskConfiguration()
        setattr(taskconf, attribute, value)
        self.assertEqual(getattr(taskconf, attribute), value)

    def set_text_file(self, path, fmt, expected, id_regex=None, class_regex=None, id_sort=None):
        task = Task()
        task.configuration = TaskConfiguration()
        task.configuration.language = Language.EN
        task.configuration.is_text_file_format = fmt
        if id_regex is not None:
            task.configuration.is_text_unparsed_id_regex = id_regex
        if class_regex is not None:
            task.configuration.is_text_unparsed_class_regex = class_regex
        if id_sort is not None:
            task.configuration.is_text_unparsed_id_sort = id_sort
        task.text_file_path_absolute = gf.get_abs_path(path, __file__)
        self.assertNotEqual(task.text_file, None)
        self.assertEqual(len(task.text_file), expected)

    def tc_from_string(self, config_string, properties):
        taskconf = TaskConfiguration(config_string)
        for prop in properties:
            self.assertEqual(getattr(taskconf, prop[0]), prop[1])

    def tc_from_string_some_invalid(self, config_string, expected_description):
        properties = [
            ["language", Language.IT],
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
        self.assertEqual(task.configuration, None)

    def test_task_string_configuration_invalid(self):
        with self.assertRaises(TypeError):
            task = Task(1)

    def test_task_string_configuration_str(self):
        with self.assertRaises(TypeError):
            task = Task(b"task_language=en")

    def test_task_string_configuration_unicode(self):
        task = Task(u"task_language=en")
        self.assertNotEqual(task.configuration, None)

    def test_task_set_configuration(self):
        task = Task()
        taskconf = TaskConfiguration()
        task.configuration = taskconf
        self.assertNotEqual(task.configuration, None)

    def test_task_empty_on_creation(self):
        task = Task()
        self.assertEqual(task.configuration, None)
        self.assertEqual(task.text_file, None)
        self.assertEqual(task.audio_file, None)
        self.assertEqual(task.sync_map, None)
        self.assertEqual(task.audio_file_path, None)
        self.assertEqual(task.audio_file_path_absolute, None)
        self.assertEqual(task.text_file_path, None)
        self.assertEqual(task.text_file_path_absolute, None)
        self.assertEqual(task.sync_map_file_path, None)
        self.assertEqual(task.sync_map_file_path_absolute, None)

    def test_set_audio_file_path_absolute(self):
        task = Task()
        task.audio_file_path_absolute = gf.get_abs_path("res/container/job/assets/p001.mp3", __file__)
        self.assertNotEqual(task.audio_file, None)
        self.assertEqual(task.audio_file.file_size, 426735)
        self.assertAlmostEqual(task.audio_file.audio_length, 53.3, places=1)

    def test_set_audio_file_path_absolute_error(self):
        task = Task()
        with self.assertRaises(OSError):
            task.audio_file_path_absolute = gf.get_abs_path("not_existing.mp3", __file__)

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
        task.configuration.language = Language.EN
        task.configuration.os_file_format = SyncMapFormat.TXT
        task.sync_map = self.dummy_sync_map()
        handler, output_file_path = gf.tmp_file(suffix=".txt")
        task.sync_map_file_path_absolute = output_file_path
        path = task.output_sync_map_file()
        self.assertNotEqual(path, None)
        self.assertEqual(path, output_file_path)
        gf.delete_file(handler, output_file_path)

    def test_tc_language(self):
        # NOTE the string parameter is "task_language"
        self.setter("language", Language.IT)

    def test_tc_description_str(self):
        # NOTE the string parameter is "task_description"
        self.setter("description", u"Test description")

    def test_tc_description_unicode_ascii(self):
        self.setter("description", u"Test description")

    def test_tc_description_unicode_unicode(self):
        self.setter("description", u"Test àèìòù")

    def test_tc_custom_id(self):
        self.setter("custom_id", u"customid")

    def test_tc_adjust_boundary_algorithm(self):
        self.setter("adjust_boundary_algorithm", AdjustBoundaryAlgorithm.AUTO)

    def test_tc_adjust_boundary_aftercurrent_value(self):
        self.setter("adjust_boundary_aftercurrent_value", u"0.100")

    def test_tc_adjust_boundary_beforenext_value(self):
        self.setter("adjust_boundary_beforenext_value", u"0.100")

    def test_tc_adjust_boundary_offset_value(self):
        self.setter("adjust_boundary_offset_value", u"0.100")

    def test_tc_adjust_boundary_percent_value(self):
        self.setter("adjust_boundary_percent_value", u"75")

    def test_tc_adjust_boundary_rate_value(self):
        self.setter("adjust_boundary_rate_value", u"22.5")

    def test_tc_is_audio_file_detect_head_min(self):
        self.setter("is_audio_file_detect_head_min", u"1.000")

    def test_tc_is_audio_file_detect_head_max(self):
        self.setter("is_audio_file_detect_head_max", u"10.000")

    def test_tc_is_audio_file_detect_tail_min(self):
        self.setter("is_audio_file_detect_tail_min", u"1.000")

    def test_tc_is_audio_file_detect_tail_max(self):
        self.setter("is_audio_file_detect_tail_max", u"5.000")

    def test_tc_is_audio_file_head_length(self):
        self.setter("is_audio_file_head_length", u"20")

    def test_tc_is_audio_file_process_length(self):
        self.setter("is_audio_file_process_length", u"100")

    def test_tc_is_text_file_format(self):
        self.setter("is_text_file_format", TextFileFormat.PLAIN)

    def test_tc_is_text_unparsed_class_regex(self):
        self.setter("is_text_unparsed_class_regex", u"f[0-9]*")

    def test_tc_is_text_unparsed_id_regex(self):
        self.setter("is_text_unparsed_id_regex", u"ra")

    def test_tc_is_text_unparsed_id_sort(self):
        self.setter("is_text_unparsed_id_sort", IDSortingAlgorithm.NUMERIC)

    def test_tc_os_file_format(self):
        self.setter("os_file_format", SyncMapFormat.SMIL)

    def test_tc_os_file_name(self):
        self.setter("os_file_name", u"output.smil")

    def test_tc_os_file_smil_audio_ref(self):
        self.setter("os_file_smil_audio_ref", u"../audio/audio001.mp3")

    def test_tc_os_file_smil_page_ref(self):
        self.setter("os_file_smil_page_ref", u"../text/page001.xhtml")

    def test_tc_os_file_head_tail_format(self):
        self.setter("os_file_head_tail_format", SyncMapHeadTailFormat.ADD)

    def test_tc_config_string(self):
        taskconf = TaskConfiguration()
        taskconf.language = Language.IT
        taskconf.description = u"Test description"
        taskconf.custom_id = u"customid"
        taskconf.is_audio_file_head_length = u"20"
        taskconf.is_audio_file_process_length = u"100"
        taskconf.os_file_format = SyncMapFormat.SMIL
        taskconf.os_file_name = u"output.smil"
        taskconf.os_file_smil_audio_ref = u"../audio/audio001.mp3"
        taskconf.os_file_smil_page_ref = u"../text/page001.xhtml"
        expected = u"task_description=Test description|task_language=it|task_custom_id=customid|is_audio_file_head_length=20|is_audio_file_process_length=100|os_task_file_format=smil|os_task_file_name=output.smil|os_task_file_smil_audio_ref=../audio/audio001.mp3|os_task_file_smil_page_ref=../text/page001.xhtml"
        self.assertEqual(taskconf.config_string(), expected)

    def test_tc_from_string_with_optional(self):
        config_string = u"task_description=Test description|task_language=it|task_custom_id=customid|is_audio_file_head_length=20|is_audio_file_process_length=100|os_task_file_format=smil|os_task_file_name=output.smil|os_task_file_smil_audio_ref=../audio/audio001.mp3|os_task_file_smil_page_ref=../text/page001.xhtml"
        properties = [
            ["language", Language.IT],
            ["description", u"Test description"],
            ["custom_id", u"customid"],
            ["is_audio_file_head_length", u"20"],
            ["is_audio_file_process_length", u"100"],
            ["os_file_format", SyncMapFormat.SMIL],
            ["os_file_name", u"output.smil"],
            ["os_file_smil_audio_ref", u"../audio/audio001.mp3"],
            ["os_file_smil_page_ref", u"../text/page001.xhtml"],
        ]
        self.tc_from_string(config_string, properties)

    def test_tc_from_string_no_optional(self):
        config_string = u"task_description=Test description|task_language=it|task_custom_id=customid|is_audio_file_head_length=20|is_audio_file_process_length=100|os_task_file_format=txt|os_task_file_name=output.txt"
        properties = [
            ["language", Language.IT],
            ["description", u"Test description"],
            ["custom_id", u"customid"],
            ["is_audio_file_head_length", u"20"],
            ["is_audio_file_process_length", u"100"],
            ["os_file_format", SyncMapFormat.TXT],
            ["os_file_name", u"output.txt"],
            ["os_file_smil_audio_ref", None],
            ["os_file_smil_page_ref", None],
        ]
        self.tc_from_string(config_string, properties)

    def test_tc_from_string_simple(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description|task_language=it",
            u"Test description"
        )

    def test_tc_from_string_unicode(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description àèìòù|task_language=it",
            u"Test description àèìòù"
        )

    def test_tc_from_string_repeated_pipes(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description|||task_language=it",
            u"Test description"
        )

    def test_tc_from_string_invalid_key_with_value(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description|not_a_valid_key=foo|task_language=it",
            u"Test description"
        )

    def test_tc_from_string_invalid_key_no_value(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description|not_a_valid_key=|task_language=it",
            u"Test description"
        )

    def test_tc_from_string_value_without_key(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description|=foo|task_language=it",
            u"Test description"
        )

    def test_tc_from_string_trailing_pipe(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description|task_language=it|",
            u"Test description"
        )

    def test_tc_from_string_leading_pipe(self):
        self.tc_from_string_some_invalid(
            u"|task_description=Test description|task_language=it",
            u"Test description"
        )

    def test_tc_from_string_leading_and_trailing_pipe(self):
        self.tc_from_string_some_invalid(
            u"|task_description=Test description|task_language=it|",
            u"Test description"
        )

    def test_tc_from_string_valid_key_no_value(self):
        self.tc_from_string_some_invalid(
            u"task_description=|task_language=it",
            None
        )

    def test_tc_from_string_space_before_valid_key(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description=|task_language=it",
            None
        )

    def test_tc_from_string_value_with_trailing_space(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test with space |task_language=it",
            u"Test with space "
        )

    def test_tc_from_string_space_before_and_after_key_value_pair(self):
        self.tc_from_string_some_invalid(
            u" task_description=Test with space |task_language=it",
            None
        )

    def test_tc_from_string_several_invalid_stuff(self):
        self.tc_from_string_some_invalid(
            u"task_description=Test description|foo=|=bar|foo=bar|||task_language=it",
            u"Test description"
        )

if __name__ == '__main__':
    unittest.main()



