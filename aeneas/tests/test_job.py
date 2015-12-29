#!/usr/bin/env python
# coding=utf-8

import unittest

from aeneas.container import ContainerFormat
from aeneas.hierarchytype import HierarchyType
from aeneas.idsortingalgorithm import IDSortingAlgorithm
from aeneas.job import Job
from aeneas.job import JobConfiguration
from aeneas.language import Language
from aeneas.logger import Logger
from aeneas.task import Task
from aeneas.textfile import TextFileFormat

class TestJob(unittest.TestCase):

    def setter(self, attribute, value):
        jobconf = JobConfiguration()
        setattr(jobconf, attribute, value)
        self.assertEqual(getattr(jobconf, attribute), value)

    def test_job_logger(self):
        logger = Logger()
        job = Job(logger=logger)

    def test_job_identifier(self):
        job = Job()
        self.assertEqual(len(job.identifier), 36)

    def test_job_empty_on_creation(self):
        job = Job()
        self.assertEqual(len(job), 0)

    def test_job_empty_configuration(self):
        job = Job()
        self.assertEqual(job.configuration, None)

    def test_job_string_configuration_invalid(self):
        with self.assertRaises(TypeError):
            job = Job(1)

    def test_job_string_configuration_bytes(self):
        with self.assertRaises(TypeError):
            job = Job(b"job_language=en")

    def test_job_string_configuration_unicode(self):
        job = Job(u"job_language=en")
        self.assertNotEqual(job.configuration, None)

    def test_job_set_configuration(self):
        job = Job()
        jobconf = JobConfiguration()
        job.configuration = jobconf
        self.assertNotEqual(job.configuration, None)

    def test_job_append_task(self):
        job = Job()
        self.assertEqual(len(job), 0)
        task1 = Task()
        job.append_task(task1)
        self.assertEqual(len(job), 1)
        task2 = Task()
        job.append_task(task2)
        self.assertEqual(len(job), 2)
        task3 = Task()
        job.append_task(task3)
        self.assertEqual(len(job), 3)

    def test_job_clear_tasks(self):
        job = Job()
        task1 = Task()
        job.tasks.append(task1)
        self.assertEqual(len(job), 1)
        job.clear_tasks()
        self.assertEqual(len(job), 0)

    def test_jc_language(self):
        # NOTE the string parameter is "job_language"
        self.setter("language", Language.IT)

    def test_jc_description_str(self):
        # NOTE the string parameter is "job_description"
        self.setter("description", u"Test description")

    def test_jc_description_unicode_ascii(self):
        self.setter("description", u"Test description")

    def test_jc_description_unicode_unicode(self):
        self.setter("description", u"Test àèìòù")

    def test_jc_is_audio_file_name_regex(self):
        self.setter("is_audio_file_name_regex", u"*.mp3")

    def test_jc_is_audio_file_relative_path(self):
        self.setter("is_audio_file_relative_path", u".")

    def test_jc_is_hierarchy_prefix(self):
        self.setter("is_hierarchy_prefix", u"OEBPS/")

    def test_jc_is_hierarchy_type(self):
        self.setter("is_hierarchy_type", HierarchyType.FLAT)

    def test_jc_is_task_directory_name_regex(self):
        self.setter("is_task_directory_name_regex", u"[0-9]*")

    def test_jc_is_text_file_format(self):
        self.setter("is_text_file_format", TextFileFormat.UNPARSED)

    def test_jc_is_text_file_name_regex(self):
        self.setter("is_text_file_name_regex", u"*.txt")

    def test_jc_is_text_file_relative_path(self):
        self.setter("is_text_file_relative_path", u".")

    def test_jc_is_text_unparsed_class_regex(self):
        self.setter("is_text_unparsed_class_regex", u"ra")

    def test_jc_is_text_unparsed_id_regex(self):
        self.setter("is_text_unparsed_id_regex", u"f[0-9]*")

    def test_jc_is_text_unparsed_id_sort(self):
        self.setter("is_text_unparsed_id_sort", IDSortingAlgorithm.NUMERIC)

    def test_jc_os_container_format(self):
        self.setter("os_container_format", ContainerFormat.ZIP)

    def test_jc_os_hierarchy_prefix(self):
        self.setter("os_hierarchy_prefix", u"OEBPS/mo/")

    def test_jc_os_hierarchy_type(self):
        self.setter("os_hierarchy_type", HierarchyType.FLAT)

    def test_jc_os_file_name(self):
        self.setter("os_file_name", u"test_output.zip")

    def test_config_string_full(self):
        jobconf = JobConfiguration()
        jobconf.language = Language.IT
        jobconf.description = u"Test description"
        jobconf.is_audio_file_name_regex = u"*.mp3"
        jobconf.is_audio_file_relative_path = u"."
        jobconf.is_hierarchy_prefix = u"OEBPS/"
        jobconf.is_hierarchy_type = HierarchyType.FLAT
        jobconf.is_task_directory_name_regex = u"[0-9]*"
        jobconf.is_text_file_format = TextFileFormat.UNPARSED
        jobconf.is_text_file_name_regex = u"*.txt"
        jobconf.is_text_file_relative_path = u"."
        jobconf.is_text_unparsed_class_regex = u"ra"
        jobconf.is_text_unparsed_id_regex = u"f[0-9]*"
        jobconf.is_text_unparsed_id_sort = IDSortingAlgorithm.NUMERIC
        jobconf.os_container_format = ContainerFormat.ZIP
        jobconf.os_hierarchy_prefix = u"OEBPS/mo/"
        jobconf.os_hierarchy_type = HierarchyType.FLAT
        jobconf.os_file_name = u"test_output.zip"
        expected = u"job_description=Test description|job_language=it|is_audio_file_name_regex=*.mp3|is_audio_file_relative_path=.|is_hierarchy_prefix=OEBPS/|is_hierarchy_type=flat|is_task_dir_name_regex=[0-9]*|is_text_type=unparsed|is_text_file_name_regex=*.txt|is_text_file_relative_path=.|is_text_unparsed_class_regex=ra|is_text_unparsed_id_regex=f[0-9]*|is_text_unparsed_id_sort=numeric|os_job_file_name=test_output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=OEBPS/mo/"
        self.assertEqual(jobconf.config_string(), expected)

    def test_config_string_missing_keys(self):
        jobconf = JobConfiguration()
        jobconf.language = Language.IT
        jobconf.description = u"Test description"
        jobconf.is_audio_file_name_regex = u"*.mp3"
        jobconf.is_audio_file_relative_path = u"."
        jobconf.is_hierarchy_prefix = u"OEBPS/"
        jobconf.is_hierarchy_type = HierarchyType.FLAT
        jobconf.is_task_directory_name_regex = u"[0-9]*"
        jobconf.is_text_file_format = TextFileFormat.PLAIN
        jobconf.is_text_file_name_regex = u"*.txt"
        jobconf.is_text_file_relative_path = u"."
        jobconf.os_container_format = ContainerFormat.ZIP
        jobconf.os_hierarchy_prefix = u"OEBPS/mo/"
        jobconf.os_hierarchy_type = HierarchyType.FLAT
        jobconf.os_file_name = u"test_output.zip"
        expected = u"job_description=Test description|job_language=it|is_audio_file_name_regex=*.mp3|is_audio_file_relative_path=.|is_hierarchy_prefix=OEBPS/|is_hierarchy_type=flat|is_task_dir_name_regex=[0-9]*|is_text_type=plain|is_text_file_name_regex=*.txt|is_text_file_relative_path=.|os_job_file_name=test_output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=OEBPS/mo/"
        self.assertEqual(jobconf.config_string(), expected)

    def test_constructor_from_config_string_unparsed(self):
        config_string = u"job_description=Test description|job_language=it|is_audio_file_name_regex=*.mp3|is_audio_file_relative_path=.|is_hierarchy_prefix=OEBPS/|is_hierarchy_type=flat|is_task_dir_name_regex=[0-9]*|is_text_type=unparsed|is_text_file_name_regex=*.txt|is_text_file_relative_path=.|is_text_unparsed_class_regex=ra|is_text_unparsed_id_regex=f[0-9]*|is_text_unparsed_id_sort=numeric|os_job_file_name=test_output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=OEBPS/mo/"
        jobconf = JobConfiguration(config_string)
        self.assertEqual(jobconf.language, Language.IT)
        self.assertEqual(jobconf.description, u"Test description")
        self.assertEqual(jobconf.is_text_file_format, TextFileFormat.UNPARSED)
        self.assertEqual(jobconf.is_text_unparsed_class_regex, u"ra")
        self.assertEqual(jobconf.is_text_unparsed_id_regex, u"f[0-9]*")
        self.assertEqual(jobconf.is_text_unparsed_id_sort, IDSortingAlgorithm.NUMERIC)

    def test_constructor_from_config_string_plain(self):
        config_string = u"job_description=Test description|job_language=it|is_audio_file_name_regex=*.mp3|is_audio_file_relative_path=.|is_hierarchy_prefix=OEBPS/|is_hierarchy_type=flat|is_task_dir_name_regex=[0-9]*|is_text_type=plain|is_text_file_name_regex=*.txt|is_text_file_relative_path=.|os_job_file_name=test_output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=OEBPS/mo/"
        jobconf = JobConfiguration(config_string)
        self.assertEqual(jobconf.language, Language.IT)
        self.assertEqual(jobconf.description, u"Test description")
        self.assertEqual(jobconf.is_text_file_format, TextFileFormat.PLAIN)
        self.assertEqual(jobconf.is_text_unparsed_class_regex, None)
        self.assertEqual(jobconf.is_text_unparsed_id_regex, None)
        self.assertEqual(jobconf.is_text_unparsed_id_sort, None)

if __name__ == '__main__':
    unittest.main()



