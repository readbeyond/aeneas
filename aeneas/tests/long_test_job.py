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

from aeneas.executejob import ExecuteJob
import aeneas.globalfunctions as gf


class TestExecuteJob(unittest.TestCase):

    def execute(self, path):
        input_path = gf.absolute_path(path, __file__)
        output_path = gf.tmp_directory()
        executor = ExecuteJob(job=None)
        executor.load_job_from_container(input_path)
        self.assertIsNotNone(executor.job)
        executor.execute()
        result_path = executor.write_output_container(output_path)
        self.assertIsNotNone(result_path)
        self.assertTrue(gf.file_exists(result_path))
        executor.clean()
        gf.delete_directory(output_path)

    def test_epub(self):
        self.execute("res/container/job.epub")

    def test_example1(self):
        self.execute("res/example_jobs/example1")

    def test_example2(self):
        self.execute("res/example_jobs/example2")

    def test_example3(self):
        self.execute("res/example_jobs/example3")

    def test_example4(self):
        self.execute("res/example_jobs/example4")

    def test_example5(self):
        self.execute("res/example_jobs/example5")

    def test_example6(self):
        self.execute("res/example_jobs/example6")

    def test_example7(self):
        self.execute("res/example_jobs/example7")

    def test_txt_config_paged_1(self):
        self.execute("res/validator/job_txt_config_paged_1")

    def test_txt_config_not_root_nested(self):
        self.execute("res/validator/job_txt_config_not_root_nested")

    def test_txt_config(self):
        self.execute("res/validator/job_txt_config")

    def test_xml_config_not_root_nested(self):
        self.execute("res/validator/job_xml_config_not_root_nested")

    def test_xml_config(self):
        self.execute("res/validator/job_xml_config")

    def test_zip(self):
        self.execute("res/container/job.zip")


if __name__ == '__main__':
    unittest.main()
