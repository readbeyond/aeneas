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
import os

from aeneas.tools.execute_job import ExecuteJobCLI
import aeneas.globalfunctions as gf


class TestExecuteJobCLI(unittest.TestCase):

    def execute(self, parameters, expected_exit_code):
        output_path = gf.tmp_directory()
        params = ["placeholder"]
        for p_type, p_value in parameters:
            if p_type == "in":
                params.append(gf.absolute_path(p_value, __file__))
            elif p_type == "out":
                params.append(os.path.join(output_path, p_value))
            else:
                params.append(p_value)
        exit_code = ExecuteJobCLI(use_sys=False).run(arguments=params)
        gf.delete_directory(output_path)
        self.assertEqual(exit_code, expected_exit_code)

    def test_exec_tool_example(self):
        self.execute([
            ("in", "../tools/res/job.zip"),
            ("out", "")
        ], 0)

    def test_exec_tool_wizard(self):
        self.execute([
            ("in", "../tools/res/job_no_config.zip"),
            ("out", ""),
            ("", "is_hierarchy_type=flat|is_hierarchy_prefix=assets/|is_text_file_relative_path=.|is_text_file_name_regex=.*\.xhtml|is_text_type=unparsed|is_audio_file_relative_path=.|is_audio_file_name_regex=.*\.mp3|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric|os_job_file_name=demo_sync_job_output|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=assets/|os_task_file_name=\$PREFIX.xhtml.smil|os_task_file_format=smil|os_task_file_smil_page_ref=\$PREFIX.xhtml|os_task_file_smil_audio_ref=../Audio/\$PREFIX.mp3|job_language=eng|job_description=Demo Sync Job")
        ], 0)

    def test_epub(self):
        self.execute([
            ("in", "res/container/job.epub"),
            ("out", "")
        ], 0)

    def test_example2(self):
        self.execute([
            ("in", "res/example_jobs/example2"),
            ("out", "")
        ], 0)

    def test_example3(self):
        self.execute([
            ("in", "res/example_jobs/example3"),
            ("out", "")
        ], 0)

    def test_example4(self):
        self.execute([
            ("in", "res/example_jobs/example4"),
            ("out", "")
        ], 0)

    def test_example5(self):
        self.execute([
            ("in", "res/example_jobs/example5"),
            ("out", "")
        ], 0)

    def test_example6(self):
        self.execute([
            ("in", "res/example_jobs/example6"),
            ("out", "")
        ], 0)

    def test_example7(self):
        self.execute([
            ("in", "res/example_jobs/example7"),
            ("out", "")
        ], 0)

    def test_txt_config_paged_1(self):
        self.execute([
            ("in", "res/validator/job_txt_config_paged_1"),
            ("out", "")
        ], 0)

    def test_txt_config_not_root_nested(self):
        self.execute([
            ("in", "res/validator/job_txt_config_not_root_nested"),
            ("out", "")
        ], 0)

    def test_txt_config(self):
        self.execute([
            ("in", "res/validator/job_txt_config"),
            ("out", "")
        ], 0)

    def test_xml_config_not_root_nested(self):
        self.execute([
            ("in", "res/validator/job_xml_config_not_root_nested"),
            ("out", "")
        ], 0)

    def test_xml_config(self):
        self.execute([
            ("in", "res/validator/job_xml_config"),
            ("out", "")
        ], 0)

    def test_zip(self):
        self.execute([
            ("in", "res/container/job.zip"),
            ("out", "")
        ], 0)


if __name__ == "__main__":
    unittest.main()
