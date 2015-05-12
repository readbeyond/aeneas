#!/usr/bin/env python
# coding=utf-8

import os
import sys
import tempfile
import unittest

from aeneas.container import ContainerFormat
from aeneas.hierarchytype import HierarchyType
from aeneas.idsortingalgorithm import IDSortingAlgorithm
from aeneas.job import Job, JobConfiguration
from aeneas.language import Language
from aeneas.task import Task
from aeneas.task import Task
from aeneas.textfile import TextFileFormat

class TestJob(unittest.TestCase):

    def test_setter_01(self):
        value = Language.IT
        jobconf = JobConfiguration()
        jobconf.language = value
        self.assertEqual(jobconf.language, value)

    def test_setter_02(self):
        value = "Test description"
        jobconf = JobConfiguration()
        jobconf.description = value
        self.assertEqual(jobconf.description, value)

    def test_setter_03(self):
        value = "*.mp3"
        jobconf = JobConfiguration()
        jobconf.is_audio_file_name_regex = value
        self.assertEqual(jobconf.is_audio_file_name_regex, value)

    def test_setter_04(self):
        value = "."
        jobconf = JobConfiguration()
        jobconf.is_audio_file_relative_path = value
        self.assertEqual(jobconf.is_audio_file_relative_path, value)

    def test_setter_05(self):
        value = "OEBPS/"
        jobconf = JobConfiguration()
        jobconf.is_hierarchy_prefix = value
        self.assertEqual(jobconf.is_hierarchy_prefix, value)

    def test_setter_06(self):
        value = HierarchyType.FLAT
        jobconf = JobConfiguration()
        jobconf.is_hierarchy_type = value
        self.assertEqual(jobconf.is_hierarchy_type, value)

    def test_setter_07(self):
        value = "[0-9]*"
        jobconf = JobConfiguration()
        jobconf.is_task_directory_name_regex = value
        self.assertEqual(jobconf.is_task_directory_name_regex, value)

    def test_setter_08(self):
        value = TextFileFormat.UNPARSED
        jobconf = JobConfiguration()
        jobconf.is_text_file_format = value
        self.assertEqual(jobconf.is_text_file_format, value)

    def test_setter_09(self):
        value = "*.txt"
        jobconf = JobConfiguration()
        jobconf.is_text_file_name_regex = value
        self.assertEqual(jobconf.is_text_file_name_regex, value)

    def test_setter_10(self):
        value = "."
        jobconf = JobConfiguration()
        jobconf.is_text_file_relative_path = value
        self.assertEqual(jobconf.is_text_file_relative_path, value)

    def test_setter_11(self):
        value = "ra"
        jobconf = JobConfiguration()
        jobconf.is_text_unparsed_class_regex = value
        self.assertEqual(jobconf.is_text_unparsed_class_regex, value)

    def test_setter_12(self):
        value = "f[0-9]*"
        jobconf = JobConfiguration()
        jobconf.is_text_unparsed_id_regex = value
        self.assertEqual(jobconf.is_text_unparsed_id_regex, value)

    def test_setter_13(self):
        value = IDSortingAlgorithm.NUMERIC
        jobconf = JobConfiguration()
        jobconf.is_text_unparsed_id_sort = value
        self.assertEqual(jobconf.is_text_unparsed_id_sort, value)

    def test_setter_14(self):
        value = ContainerFormat.ZIP
        jobconf = JobConfiguration()
        jobconf.os_container_format = value
        self.assertEqual(jobconf.os_container_format, value)

    def test_setter_15(self):
        value = "OEBPS/mo/"
        jobconf = JobConfiguration()
        jobconf.os_hierarchy_prefix = value
        self.assertEqual(jobconf.os_hierarchy_prefix, value)

    def test_setter_16(self):
        value = HierarchyType.FLAT
        jobconf = JobConfiguration()
        jobconf.os_hierarchy_type = value
        self.assertEqual(jobconf.os_hierarchy_type, value)

    def test_setter_17(self):
        value = "test_output.zip"
        jobconf = JobConfiguration()
        jobconf.os_file_name = value
        self.assertEqual(jobconf.os_file_name, value)

    def test_config_string_full(self):
        jobconf = JobConfiguration()
        jobconf.language = Language.IT
        jobconf.description = "Test description"
        jobconf.is_audio_file_name_regex = "*.mp3"
        jobconf.is_audio_file_relative_path = "."
        jobconf.is_hierarchy_prefix = "OEBPS/"
        jobconf.is_hierarchy_type = HierarchyType.FLAT
        jobconf.is_task_directory_name_regex = "[0-9]*"
        jobconf.is_text_file_format = TextFileFormat.UNPARSED
        jobconf.is_text_file_name_regex = "*.txt"
        jobconf.is_text_file_relative_path = "."
        jobconf.is_text_unparsed_class_regex = "ra"
        jobconf.is_text_unparsed_id_regex = "f[0-9]*"
        jobconf.is_text_unparsed_id_sort = IDSortingAlgorithm.NUMERIC
        jobconf.os_container_format = ContainerFormat.ZIP
        jobconf.os_hierarchy_prefix = "OEBPS/mo/"
        jobconf.os_hierarchy_type = HierarchyType.FLAT
        jobconf.os_file_name = "test_output.zip"
        expected = "job_description=Test description|job_language=it|is_audio_file_name_regex=*.mp3|is_audio_file_relative_path=.|is_hierarchy_prefix=OEBPS/|is_hierarchy_type=flat|is_task_dir_name_regex=[0-9]*|is_text_type=unparsed|is_text_file_name_regex=*.txt|is_text_file_relative_path=.|is_text_unparsed_class_regex=ra|is_text_unparsed_id_regex=f[0-9]*|is_text_unparsed_id_sort=numeric|os_job_file_name=test_output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=OEBPS/mo/"
        self.assertEqual(jobconf.config_string(), expected)

    def test_config_string_missing_keys(self):
        jobconf = JobConfiguration()
        jobconf.language = Language.IT
        jobconf.description = "Test description"
        jobconf.is_audio_file_name_regex = "*.mp3"
        jobconf.is_audio_file_relative_path = "."
        jobconf.is_hierarchy_prefix = "OEBPS/"
        jobconf.is_hierarchy_type = HierarchyType.FLAT
        jobconf.is_task_directory_name_regex = "[0-9]*"
        jobconf.is_text_file_format = TextFileFormat.PLAIN
        jobconf.is_text_file_name_regex = "*.txt"
        jobconf.is_text_file_relative_path = "."
        jobconf.os_container_format = ContainerFormat.ZIP
        jobconf.os_hierarchy_prefix = "OEBPS/mo/"
        jobconf.os_hierarchy_type = HierarchyType.FLAT
        jobconf.os_file_name = "test_output.zip"
        expected = "job_description=Test description|job_language=it|is_audio_file_name_regex=*.mp3|is_audio_file_relative_path=.|is_hierarchy_prefix=OEBPS/|is_hierarchy_type=flat|is_task_dir_name_regex=[0-9]*|is_text_type=plain|is_text_file_name_regex=*.txt|is_text_file_relative_path=.|os_job_file_name=test_output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=OEBPS/mo/"
        self.assertEqual(jobconf.config_string(), expected)

    def test_constructor_from_config_string_01(self):
        config_string = "job_description=Test description|job_language=it|is_audio_file_name_regex=*.mp3|is_audio_file_relative_path=.|is_hierarchy_prefix=OEBPS/|is_hierarchy_type=flat|is_task_dir_name_regex=[0-9]*|is_text_type=unparsed|is_text_file_name_regex=*.txt|is_text_file_relative_path=.|is_text_unparsed_class_regex=ra|is_text_unparsed_id_regex=f[0-9]*|is_text_unparsed_id_sort=numeric|os_job_file_name=test_output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=OEBPS/mo/"
        jobconf = JobConfiguration(config_string)
        self.assertEqual(jobconf.language, Language.IT)
        self.assertEqual(jobconf.description, "Test description")
        self.assertEqual(jobconf.is_text_file_format, TextFileFormat.UNPARSED)
        self.assertEqual(jobconf.is_text_unparsed_class_regex, "ra")
        self.assertEqual(jobconf.is_text_unparsed_id_regex, "f[0-9]*")
        self.assertEqual(jobconf.is_text_unparsed_id_sort, IDSortingAlgorithm.NUMERIC)

    def test_constructor_from_config_string_02(self):
        config_string = "job_description=Test description|job_language=it|is_audio_file_name_regex=*.mp3|is_audio_file_relative_path=.|is_hierarchy_prefix=OEBPS/|is_hierarchy_type=flat|is_task_dir_name_regex=[0-9]*|is_text_type=plain|is_text_file_name_regex=*.txt|is_text_file_relative_path=.|os_job_file_name=test_output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=OEBPS/mo/"
        jobconf = JobConfiguration(config_string)
        self.assertEqual(jobconf.language, Language.IT)
        self.assertEqual(jobconf.description, "Test description")
        self.assertEqual(jobconf.is_text_file_format, TextFileFormat.PLAIN)
        self.assertEqual(jobconf.is_text_unparsed_class_regex, None)
        self.assertEqual(jobconf.is_text_unparsed_id_regex, None)
        self.assertEqual(jobconf.is_text_unparsed_id_sort, None)

    def test_constructor_01(self):
        job = Job()
        self.assertEqual(len(job), 0)
        self.assertNotEqual(job.identifier, None)
        self.assertEqual(job.configuration, None)

    def test_constructor_02(self):
        config_string = "job_description=Test description|job_language=it|is_audio_file_name_regex=*.mp3|is_audio_file_relative_path=.|is_hierarchy_prefix=OEBPS/|is_hierarchy_type=flat|is_task_dir_name_regex=[0-9]*|is_text_type=unparsed|is_text_file_name_regex=*.txt|is_text_file_relative_path=.|is_text_unparsed_class_regex=ra|is_text_unparsed_id_regex=f[0-9]*|is_text_unparsed_id_sort=numeric|os_job_file_name=test_output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=OEBPS/mo/"
        job = Job(config_string)
        self.assertEqual(len(job), 0)
        self.assertNotEqual(job.identifier, None)
        self.assertNotEqual(job.configuration, None)

    def test_constructor_03(self):
        config_string = "job_description=Test description|job_language=it|is_audio_file_name_regex=*.mp3|is_audio_file_relative_path=.|is_hierarchy_prefix=OEBPS/|is_hierarchy_type=flat|is_task_dir_name_regex=[0-9]*|is_text_type=plain|is_text_file_name_regex=*.txt|is_text_file_relative_path=.|os_job_file_name=test_output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=OEBPS/mo/"
        job = Job(config_string)
        self.assertEqual(len(job), 0)
        self.assertNotEqual(job.identifier, None)
        self.assertNotEqual(job.configuration, None)

    def test_set_configuration(self):
        job = Job()
        jobconf = JobConfiguration()
        job.configuration = jobconf
        self.assertNotEqual(job.configuration, None)

    def test_append_task(self):
        job = Job()
        self.assertEqual(len(job), 0)
        task1 = Task()
        task2 = Task()
        task3 = Task()
        job.tasks.append(task1)
        self.assertEqual(len(job), 1)
        job.tasks.append(task2)
        job.tasks.append(task3)
        self.assertEqual(len(job), 3)

    def test_delete_tasks(self):
        job = Job()
        task1 = Task()
        job.tasks.append(task1)
        self.assertEqual(len(job), 1)
        job.tasks = []
        self.assertEqual(len(job), 0)

if __name__ == '__main__':
    unittest.main()



