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
        jobconf[attribute] = value
        read_value = jobconf[attribute]
        if value is None:
            self.assertIsNone(read_value)
        else:
            self.assertEqual(read_value, value)

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
        self.assertIsNone(job.configuration)

    def test_job_string_configuration_invalid(self):
        with self.assertRaises(TypeError):
            job = Job(1)

    def test_job_string_configuration_bytes(self):
        with self.assertRaises(TypeError):
            job = Job(b"job_language=eng")

    def test_job_string_configuration_unicode(self):
        job = Job(u"job_language=eng")
        self.assertIsNotNone(job.configuration)

    def test_job_set_configuration(self):
        job = Job()
        jobconf = JobConfiguration()
        job.configuration = jobconf
        self.assertIsNotNone(job.configuration)

    def test_job_add_task(self):
        job = Job()
        self.assertEqual(len(job), 0)
        task1 = Task()
        job.add_task(task1)
        self.assertEqual(len(job), 1)
        task2 = Task()
        job.add_task(task2)
        self.assertEqual(len(job), 2)
        task3 = Task()
        job.add_task(task3)
        self.assertEqual(len(job), 3)

    def test_job_clear_tasks(self):
        job = Job()
        task1 = Task()
        job.tasks.append(task1)
        self.assertEqual(len(job), 1)
        job.clear_tasks()
        self.assertEqual(len(job), 0)

    def test_jc_language(self):
        self.setter("language", Language.ITA)

    def test_jc_description_unicode_ascii(self):
        self.setter("description", u"Test description")

    def test_jc_description_unicode_unicode(self):
        self.setter("description", u"Test àèìòù")

    def test_jc_is_audio_file_name_regex(self):
        self.setter("i_a_name_regex", u"*.mp3")

    def test_jc_is_audio_file_relative_path(self):
        self.setter("i_a_relative_path", u".")

    def test_jc_is_hierarchy_prefix(self):
        self.setter("i_hierarchy_prefix", u"OEBPS/")

    def test_jc_is_hierarchy_type(self):
        self.setter("i_hierarchy_type", HierarchyType.FLAT)

    def test_jc_is_task_directory_name_regex(self):
        self.setter("i_task_directory_name_regex", u"[0-9]*")

    def test_jc_is_text_file_name_regex(self):
        self.setter("i_t_name_regex", u"*.txt")

    def test_jc_is_text_file_relative_path(self):
        self.setter("i_t_relative_path", u".")

    def test_jc_os_container_format(self):
        self.setter("o_container_format", ContainerFormat.ZIP)

    def test_jc_os_hierarchy_prefix(self):
        self.setter("o_hierarchy_prefix", u"OEBPS/mo/")

    def test_jc_os_hierarchy_type(self):
        self.setter("o_hierarchy_type", HierarchyType.FLAT)

    def test_jc_os_file_name(self):
        self.setter("o_name", u"test_output.zip")

    def test_config_string_full(self):
        jobconf = JobConfiguration()
        jobconf["language"] = Language.ITA
        jobconf["description"] = u"Test description"
        jobconf["i_a_name_regex"] = u"*.mp3"
        jobconf["i_a_relative_path"] = u"."
        jobconf["i_hierarchy_prefix"] = u"OEBPS/"
        jobconf["i_hierarchy_type"] = HierarchyType.FLAT
        jobconf["i_task_directory_name_regex"] = u"[0-9]*"
        jobconf["i_t_name_regex"] = u"*.txt"
        jobconf["i_t_relative_path"] = u"."
        jobconf["o_container_format"] = ContainerFormat.ZIP
        jobconf["o_hierarchy_prefix"] = u"OEBPS/mo/"
        jobconf["o_hierarchy_type"] = HierarchyType.FLAT
        jobconf["o_name"] = u"test_output.zip"
        expected = u"is_audio_file_name_regex=*.mp3|is_audio_file_relative_path=.|is_hierarchy_prefix=OEBPS/|is_hierarchy_type=flat|is_task_dir_name_regex=[0-9]*|is_text_file_name_regex=*.txt|is_text_file_relative_path=.|job_description=Test description|job_language=ita|os_job_file_container=zip|os_job_file_hierarchy_prefix=OEBPS/mo/|os_job_file_hierarchy_type=flat|os_job_file_name=test_output.zip"
        self.assertEqual(jobconf.config_string, expected)

    def test_config_string_missing_keys(self):
        jobconf = JobConfiguration()
        jobconf["language"] = Language.ITA
        jobconf["description"] = u"Test description"
        jobconf["i_a_name_regex"] = u"*.mp3"
        jobconf["i_a_relative_path"] = u"."
        jobconf["i_hierarchy_prefix"] = u"OEBPS/"
        jobconf["i_hierarchy_type"] = HierarchyType.FLAT
        jobconf["i_task_directory_name_regex"] = u"[0-9]*"
        jobconf["i_t_name_regex"] = u"*.txt"
        jobconf["i_t_relative_path"] = u"."
        jobconf["o_container_format"] = ContainerFormat.ZIP
        jobconf["o_hierarchy_prefix"] = u"OEBPS/mo/"
        jobconf["o_hierarchy_type"] = HierarchyType.FLAT
        jobconf["o_name"] = u"test_output.zip"
        expected = u"is_audio_file_name_regex=*.mp3|is_audio_file_relative_path=.|is_hierarchy_prefix=OEBPS/|is_hierarchy_type=flat|is_task_dir_name_regex=[0-9]*|is_text_file_name_regex=*.txt|is_text_file_relative_path=.|job_description=Test description|job_language=ita|os_job_file_container=zip|os_job_file_hierarchy_prefix=OEBPS/mo/|os_job_file_hierarchy_type=flat|os_job_file_name=test_output.zip"
        self.assertEqual(jobconf.config_string, expected)

    def test_constructor_from_config_string_unparsed(self):
        config_string = u"job_description=Test description|job_language=ita|is_audio_file_name_regex=*.mp3|is_audio_file_relative_path=.|is_hierarchy_prefix=OEBPS/|is_hierarchy_type=flat|is_task_dir_name_regex=[0-9]*|is_text_type=unparsed|is_text_file_name_regex=*.txt|is_text_file_relative_path=.|is_text_unparsed_class_regex=ra|is_text_unparsed_id_regex=f[0-9]*|is_text_unparsed_id_sort=numeric|os_job_file_name=test_output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=OEBPS/mo/"
        jobconf = JobConfiguration(config_string)
        self.assertEqual(jobconf["language"], Language.ITA)
        self.assertEqual(jobconf["description"], u"Test description")

    def test_constructor_from_config_string_plain(self):
        config_string = u"job_description=Test description|job_language=ita|is_audio_file_name_regex=*.mp3|is_audio_file_relative_path=.|is_hierarchy_prefix=OEBPS/|is_hierarchy_type=flat|is_task_dir_name_regex=[0-9]*|is_text_type=plain|is_text_file_name_regex=*.txt|is_text_file_relative_path=.|os_job_file_name=test_output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=OEBPS/mo/"
        jobconf = JobConfiguration(config_string)
        self.assertEqual(jobconf["language"], Language.ITA)
        self.assertEqual(jobconf["description"], u"Test description")


if __name__ == "__main__":
    unittest.main()
