#!/usr/bin/env python
# coding=utf-8

import os
import shutil
import sys
import tempfile
import unittest

from . import get_abs_path

from aeneas.idsortingalgorithm import IDSortingAlgorithm
from aeneas.language import Language
from aeneas.syncmap import SyncMap, SyncMapFormat, SyncMapFragment
from aeneas.task import Task, TaskConfiguration
from aeneas.textfile import TextFile, TextFileFormat, TextFragment

class TestTask(unittest.TestCase):

    def dummy_sync_map(self):
        sync_map = SyncMap()
        frag = TextFragment("f001", Language.EN, "Fragment 1")
        sync_map.append(SyncMapFragment(frag, 0, 12.345))
        frag = TextFragment("f002", Language.EN, "Fragment 2")
        sync_map.append(SyncMapFragment(frag, 12.345, 23.456))
        frag = TextFragment("f003", Language.EN, "Fragment 3")
        sync_map.append(SyncMapFragment(frag, 23.456, 34.567))
        return sync_map

    def test_setter_01(self):
        value = Language.IT
        taskconf = TaskConfiguration()
        taskconf.language = value
        self.assertEqual(taskconf.language, value)

    def test_setter_02(self):
        value = "Test description"
        taskconf = TaskConfiguration()
        taskconf.description = value
        self.assertEqual(taskconf.description, value)

    def test_setter_03(self):
        value = "customid"
        taskconf = TaskConfiguration()
        taskconf.custom_id = value
        self.assertEqual(taskconf.custom_id, value)

    def test_setter_04(self):
        value = "20"
        taskconf = TaskConfiguration()
        taskconf.is_audio_file_head_length = value
        self.assertEqual(taskconf.is_audio_file_head_length, value)

    def test_setter_05(self):
        value = "100"
        taskconf = TaskConfiguration()
        taskconf.is_audio_file_process_length = value
        self.assertEqual(taskconf.is_audio_file_process_length, value)

    def test_setter_06(self):
        value = SyncMapFormat.SMIL
        taskconf = TaskConfiguration()
        taskconf.os_file_format = value
        self.assertEqual(taskconf.os_file_format, value)

    def test_setter_07(self):
        value = "output.smil"
        taskconf = TaskConfiguration()
        taskconf.os_file_name = value
        self.assertEqual(taskconf.os_file_name, value)

    def test_setter_08(self):
        value = "../audio/audio001.mp3"
        taskconf = TaskConfiguration()
        taskconf.os_file_smil_audio_ref = value
        self.assertEqual(taskconf.os_file_smil_audio_ref, value)

    def test_setter_09(self):
        value = "../text/page001.xhtml"
        taskconf = TaskConfiguration()
        taskconf.os_file_smil_page_ref = value
        self.assertEqual(taskconf.os_file_smil_page_ref, value)

    def test_config_string_full(self):
        taskconf = TaskConfiguration()
        taskconf.language = Language.IT
        taskconf.description = "Test description"
        taskconf.custom_id = "customid"
        taskconf.is_audio_file_head_length = "20"
        taskconf.is_audio_file_process_length = "100"
        taskconf.os_file_format = SyncMapFormat.SMIL
        taskconf.os_file_name = "output.smil"
        taskconf.os_file_smil_audio_ref = "../audio/audio001.mp3"
        taskconf.os_file_smil_page_ref = "../text/page001.xhtml"
        expected = "task_description=Test description|task_language=it|task_custom_id=customid|is_audio_file_head_length=20|is_audio_file_process_length=100|os_task_file_format=smil|os_task_file_name=output.smil|os_task_file_smil_audio_ref=../audio/audio001.mp3|os_task_file_smil_page_ref=../text/page001.xhtml"
        self.assertEqual(taskconf.config_string(), expected)

    def test_config_string_missing_keys(self):
        taskconf = TaskConfiguration()
        taskconf.language = Language.IT
        taskconf.description = "Test description"
        taskconf.custom_id = "customid"
        taskconf.is_audio_file_head_length = "20"
        taskconf.is_audio_file_process_length = "100"
        taskconf.os_file_format = SyncMapFormat.TXT
        taskconf.os_file_name = "output.txt"
        expected = "task_description=Test description|task_language=it|task_custom_id=customid|is_audio_file_head_length=20|is_audio_file_process_length=100|os_task_file_format=txt|os_task_file_name=output.txt"
        self.assertEqual(taskconf.config_string(), expected)

    def test_constructor_from_config_string_01(self):
        config_string = "task_description=Test description|task_language=it|task_custom_id=customid|is_audio_file_head_length=20|is_audio_file_process_length=100|os_task_file_format=smil|os_task_file_name=output.smil|os_task_file_smil_audio_ref=../audio/audio001.mp3|os_task_file_smil_page_ref=../text/page001.xhtml"
        taskconf = TaskConfiguration(config_string)
        self.assertEqual(taskconf.language, Language.IT)
        self.assertEqual(taskconf.description, "Test description")
        self.assertEqual(taskconf.custom_id, "customid")
        self.assertEqual(taskconf.is_audio_file_head_length, "20")
        self.assertEqual(taskconf.is_audio_file_process_length, "100")
        self.assertEqual(taskconf.os_file_format, SyncMapFormat.SMIL)
        self.assertEqual(taskconf.os_file_name, "output.smil")
        self.assertEqual(taskconf.os_file_smil_audio_ref, "../audio/audio001.mp3")
        self.assertEqual(taskconf.os_file_smil_page_ref, "../text/page001.xhtml")

    def test_constructor_from_config_string_02(self):
        config_string = "task_description=Test description|task_language=it|task_custom_id=customid|is_audio_file_head_length=20|is_audio_file_process_length=100|os_task_file_format=txt|os_task_file_name=output.txt"
        taskconf = TaskConfiguration(config_string)
        self.assertEqual(taskconf.language, Language.IT)
        self.assertEqual(taskconf.description, "Test description")
        self.assertEqual(taskconf.custom_id, "customid")
        self.assertEqual(taskconf.is_audio_file_head_length, "20")
        self.assertEqual(taskconf.is_audio_file_process_length, "100")
        self.assertEqual(taskconf.os_file_format, SyncMapFormat.TXT)
        self.assertEqual(taskconf.os_file_name, "output.txt")
        self.assertEqual(taskconf.os_file_smil_audio_ref, None)
        self.assertEqual(taskconf.os_file_smil_page_ref, None)

    def test_constructor_from_config_string_03(self):
        config_string = "task_description=Test description|not_a_valid_key=foo|task_language=it"
        taskconf = TaskConfiguration(config_string)
        self.assertEqual(taskconf.language, Language.IT)
        self.assertEqual(taskconf.description, "Test description")

    def test_constructor_from_config_string_04(self):
        config_string = "task_description=Test description|not_a_valid_key=|task_language=it"
        taskconf = TaskConfiguration(config_string)
        self.assertEqual(taskconf.language, Language.IT)
        self.assertEqual(taskconf.description, "Test description")

    def test_constructor_from_config_string_05(self):
        config_string = "task_description=Test description|=foo|task_language=it"
        taskconf = TaskConfiguration(config_string)
        self.assertEqual(taskconf.language, Language.IT)
        self.assertEqual(taskconf.description, "Test description")

    def test_constructor_from_config_string_06(self):
        config_string = "task_description=Test description|task_language=it|"
        taskconf = TaskConfiguration(config_string)
        self.assertEqual(taskconf.language, Language.IT)
        self.assertEqual(taskconf.description, "Test description")

    def test_constructor_from_config_string_07(self):
        config_string = "|task_description=Test description|task_language=it"
        taskconf = TaskConfiguration(config_string)
        self.assertEqual(taskconf.language, Language.IT)
        self.assertEqual(taskconf.description, "Test description")

    def test_constructor_from_config_string_08(self):
        config_string = "|task_description=Test description|task_language=it|"
        taskconf = TaskConfiguration(config_string)
        self.assertEqual(taskconf.language, Language.IT)
        self.assertEqual(taskconf.description, "Test description")

    def test_constructor_from_config_string_09(self):
        config_string = "task_description=|task_language=it"
        taskconf = TaskConfiguration(config_string)
        self.assertEqual(taskconf.language, Language.IT)
        self.assertEqual(taskconf.description, None)

    def test_constructor_from_config_string_10(self):
        config_string = "task_description=Test description=|task_language=it"
        taskconf = TaskConfiguration(config_string)
        self.assertEqual(taskconf.language, Language.IT)
        self.assertEqual(taskconf.description, None)

    def test_constructor_from_config_string_11(self):
        config_string = "task_description=Test with space |task_language=it"
        taskconf = TaskConfiguration(config_string)
        self.assertEqual(taskconf.language, Language.IT)
        self.assertEqual(taskconf.description, "Test with space ")

    def test_constructor_from_config_string_12(self):
        config_string = " task_description=Test with space|task_language=it"
        taskconf = TaskConfiguration(config_string)
        self.assertEqual(taskconf.language, Language.IT)
        self.assertEqual(taskconf.description, None)

    def test_constructor_from_config_string_13(self):
        config_string = " task_description=Test with space |task_language=it"
        taskconf = TaskConfiguration(config_string)
        self.assertEqual(taskconf.language, Language.IT)
        self.assertEqual(taskconf.description, None)

    def test_constructor_from_config_string_14(self):
        config_string = "task_description=Test description|foo=|=bar|foo=bar|||task_language=it"
        taskconf = TaskConfiguration(config_string)
        self.assertEqual(taskconf.language, Language.IT)
        self.assertEqual(taskconf.description, "Test description")


    def test_constructor_01(self):
        task = Task()
        self.assertNotEqual(task.identifier, None)
        self.assertEqual(task.configuration, None)
        self.assertEqual(task.audio_file_path, None)
        self.assertEqual(task.text_file_path, None)

    def test_constructor_02(self):
        config_string = "task_description=Test description|task_language=it|task_custom_id=customid|is_audio_file_head_length=20|is_audio_file_process_length=100|os_task_file_format=smil|os_task_file_name=output.smil|os_task_file_smil_audio_ref=../audio/audio001.mp3|os_task_file_smil_page_ref=../text/page001.xhtml"
        task = Task(config_string)
        self.assertNotEqual(task.identifier, None)
        self.assertNotEqual(task.configuration, None)
        self.assertEqual(task.audio_file_path, None)
        self.assertEqual(task.text_file_path, None)

    def test_constructor_03(self):
        config_string = "task_description=Test description|task_language=it|task_custom_id=customid|is_audio_file_head_length=20|is_audio_file_process_length=100|os_task_file_format=txt|os_task_file_name=output.txt"
        task = Task(config_string)
        self.assertNotEqual(task.identifier, None)
        self.assertNotEqual(task.configuration, None)
        self.assertEqual(task.audio_file_path, None)
        self.assertEqual(task.text_file_path, None)

    def test_set_configuration(self):
        task = Task()
        taskconf = TaskConfiguration()
        task.configuration = taskconf
        self.assertNotEqual(task.configuration, None)

    def test_set_audio_file_path_absolute_01(self):
        task = Task()
        task.audio_file_path_absolute = get_abs_path("res/container/job/assets/p001.mp3")
        self.assertNotEqual(task.audio_file, None)
        self.assertEqual(task.audio_file.file_size, 426735)
        self.assertEqual(int(task.audio_file.audio_length), 53)

    def test_set_audio_file_path_absolute_02(self):
        task = Task()
        with self.assertRaises(OSError):
            task.audio_file_path_absolute = get_abs_path("not/existing.mp3")

    def test_set_text_file_path_absolute_01(self):
        task = Task()
        task.configuration = TaskConfiguration()
        task.configuration.language = Language.EN
        task.configuration.is_text_file_format = TextFileFormat.UNPARSED
        task.configuration.is_text_unparsed_id_regex = "f[0-9]+"
        task.configuration.is_text_unparsed_id_sort = IDSortingAlgorithm.NUMERIC
        task.text_file_path_absolute = get_abs_path("res/inputtext/sonnet_unparsed_id.xhtml")
        self.assertNotEqual(task.text_file, None)
        self.assertEqual(len(task.text_file), 15)

    def test_set_text_file_path_absolute_02(self):
        task = Task()
        task.configuration = TaskConfiguration()
        task.configuration.language = Language.EN
        task.configuration.is_text_file_format = TextFileFormat.UNPARSED
        task.configuration.is_text_unparsed_id_regex = "f[0-9]+"
        task.configuration.is_text_unparsed_class_regex = "ra"
        task.configuration.is_text_unparsed_id_sort = IDSortingAlgorithm.NUMERIC
        task.text_file_path_absolute = get_abs_path("res/inputtext/sonnet_unparsed_class.xhtml")
        self.assertNotEqual(task.text_file, None)
        self.assertEqual(len(task.text_file), 0)

    def test_set_text_file_path_absolute_03(self):
        task = Task()
        task.configuration = TaskConfiguration()
        task.configuration.language = Language.EN
        task.configuration.is_text_file_format = TextFileFormat.UNPARSED
        task.configuration.is_text_unparsed_class_regex = "ra"
        task.configuration.is_text_unparsed_id_sort = IDSortingAlgorithm.NUMERIC
        task.text_file_path_absolute = get_abs_path("res/inputtext/sonnet_unparsed_class_id.xhtml")
        self.assertNotEqual(task.text_file, None)
        self.assertEqual(len(task.text_file), 15)

    def test_set_text_file_path_absolute_04(self):
        task = Task()
        task.configuration = TaskConfiguration()
        task.configuration.language = Language.EN
        task.configuration.is_text_file_format = TextFileFormat.PLAIN
        task.text_file_path_absolute = get_abs_path("res/inputtext/sonnet_plain.txt")
        self.assertNotEqual(task.text_file, None)
        self.assertEqual(len(task.text_file), 15)

    def test_set_text_file_path_absolute_05(self):
        task = Task()
        task.configuration = TaskConfiguration()
        task.configuration.language = Language.EN
        task.configuration.is_text_file_format = TextFileFormat.PARSED
        task.text_file_path_absolute = get_abs_path("res/inputtext/sonnet_parsed.txt")
        self.assertNotEqual(task.text_file, None)
        self.assertEqual(len(task.text_file), 15)

    def test_output_sync_map_01(self):
        task = Task()
        task.configuration = TaskConfiguration()
        task.configuration.language = Language.EN
        task.configuration.os_file_format = SyncMapFormat.TXT
        task.sync_map = self.dummy_sync_map()
        handler, output_file_path = tempfile.mkstemp(suffix=".txt")
        task.sync_map_file_path_absolute = output_file_path
        path = task.output_sync_map_file()
        self.assertNotEqual(path, None)
        self.assertEqual(path, output_file_path)
        os.close(handler)
        os.remove(output_file_path)

    def test_output_sync_map_02(self):
        task = Task()
        task.configuration = TaskConfiguration()
        task.configuration.language = Language.EN
        task.configuration.os_file_format = SyncMapFormat.TXT
        task.sync_map = self.dummy_sync_map()
        handler, output_file_path = tempfile.mkstemp(suffix=".txt")
        task.sync_map_file_path = output_file_path
        output_path = tempfile.mkdtemp()
        path = task.output_sync_map_file(container_root_path=output_path)
        self.assertNotEqual(path, None)
        self.assertEqual(path, os.path.join(output_path, output_file_path))
        os.close(handler)
        os.remove(output_file_path)
        shutil.rmtree(output_path)

    def test_output_sync_map_03(self):
        task = Task()
        task.configuration = TaskConfiguration()
        task.configuration.language = Language.EN
        task.configuration.os_file_format = SyncMapFormat.TXT
        task.sync_map = self.dummy_sync_map()
        output_path = tempfile.mkdtemp()
        path = task.output_sync_map_file(container_root_path=output_path)
        self.assertEqual(path, None)
        shutil.rmtree(output_path)

    def test_output_sync_map_04(self):
        task = Task()
        task.configuration = TaskConfiguration()
        task.configuration.language = Language.EN
        task.configuration.os_file_format = SyncMapFormat.SMIL
        task.configuration.os_file_smil_page_ref = "Text/page.xhtml"
        task.configuration.os_file_smil_audio_ref = "Audio/audio.mp3"
        task.sync_map = self.dummy_sync_map()
        handler, output_file_path = tempfile.mkstemp(suffix=".smil")
        task.sync_map_file_path_absolute = output_file_path
        path = task.output_sync_map_file()
        self.assertNotEqual(path, None)
        os.close(handler)
        os.remove(output_file_path)

    def test_output_sync_map_05(self):
        task = Task()
        task.configuration = TaskConfiguration()
        task.configuration.language = Language.EN
        task.configuration.os_file_format = SyncMapFormat.SMIL
        task.configuration.os_file_smil_audio_ref = "Audio/audio.mp3"
        task.sync_map = self.dummy_sync_map()
        handler, output_file_path = tempfile.mkstemp(suffix=".smil")
        task.sync_map_file_path_absolute = output_file_path
        path = task.output_sync_map_file()
        self.assertEqual(path, None)
        os.close(handler)
        os.remove(output_file_path)

    def test_output_sync_map_06(self):
        task = Task()
        task.configuration = TaskConfiguration()
        task.configuration.language = Language.EN
        task.configuration.os_file_format = SyncMapFormat.SMIL
        task.configuration.os_file_smil_page_ref = "Text/page.xhtml"
        task.sync_map = self.dummy_sync_map()
        handler, output_file_path = tempfile.mkstemp(suffix=".smil")
        task.sync_map_file_path_absolute = output_file_path
        path = task.output_sync_map_file()
        self.assertEqual(path, None)
        os.close(handler)
        os.remove(output_file_path)

if __name__ == '__main__':
    unittest.main()



