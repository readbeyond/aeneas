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

import os
import unittest

from aeneas.tools.validate import ValidateCLI
import aeneas.globalfunctions as gf


class TestValidateCLI(unittest.TestCase):

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
        exit_code = ValidateCLI(use_sys=False).run(arguments=params)
        gf.delete_directory(output_path)
        self.assertEqual(exit_code, expected_exit_code)

    def test_help(self):
        self.execute([], 2)
        self.execute([("", "-h")], 2)
        self.execute([("", "--help")], 2)
        self.execute([("", "--help-rconf")], 2)
        self.execute([("", "--version")], 2)

    def test_bad_type(self):
        self.execute([
            ("", "foo"),
            ("in", "../tools/res/config.txt")
        ], 2)

    def test_config_txt(self):
        self.execute([
            ("", "config"),
            ("in", "../tools/res/config.txt")
        ], 0)

    def test_config_txt_bad(self):
        self.execute([
            ("", "config"),
            ("in", "../tools/res/config.bad.txt")
        ], 1)

    def test_config_xml(self):
        self.execute([
            ("", "config"),
            ("in", "../tools/res/config.xml")
        ], 0)

    def test_config_xml_bad(self):
        self.execute([
            ("", "config"),
            ("in", "../tools/res/config.bad.xml")
        ], 1)

    def test_container(self):
        self.execute([
            ("", "container"),
            ("in", "../tools/res/job.zip")
        ], 0)

    def test_container_bad(self):
        self.execute([
            ("", "container"),
            ("in", "../tools/res/job_no_config.zip")
        ], 1)

    def test_container_too_many_tasks(self):
        self.execute([
            ("", "container"),
            ("in", "../tools/res/job.zip"),
            ("", "-r=\"job_max_tasks=1\"")
        ], 1)

    def test_job(self):
        self.execute([
            ("", "job"),
            ("", "job_language=it|os_job_file_name=output.zip|os_job_file_container=zip|is_hierarchy_type=flat")
        ], 0)

    def test_job_bad(self):
        self.execute([
            ("", "job"),
            ("", "os_job_file_name=output.zip|os_job_file_container=zip|is_hierarchy_type=flat")
        ], 1)

    def test_task(self):
        self.execute([
            ("", "task"),
            ("", "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt")
        ], 0)

    def test_task_bad(self):
        self.execute([
            ("", "task"),
            ("", "task_language=it|is_text_type=plain|os_task_file_name=output.txt")
        ], 1)

    def test_wizard(self):
        self.execute([
            ("", "wizard"),
            ("", "is_hierarchy_type=flat|is_hierarchy_prefix=assets/|is_text_file_relative_path=.|is_text_file_name_regex=.*\.xhtml|is_text_type=unparsed|is_audio_file_relative_path=.|is_audio_file_name_regex=.*\.mp3|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric|os_job_file_name=demo_sync_job_output|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=assets/|os_task_file_name=\$PREFIX.xhtml.smil|os_task_file_format=smil|os_task_file_smil_page_ref=\$PREFIX.xhtml|os_task_file_smil_audio_ref=../Audio/\$PREFIX.mp3|job_language=en|job_description=Demo Sync Job"),
            ("in", "../tools/res/job_no_config.zip")
        ], 0)

    def test_wizard_bad(self):
        self.execute([
            ("", "wizard"),
            ("", "job_language=it|invalid=string"),
            ("in", "../tools/res/job_no_config.zip")
        ], 1)

    def test_read_missing_1(self):
        self.execute([
            ("", "config")
        ], 2)

    def test_read_missing_2(self):
        self.execute([
            ("in", "../tools/res/config.txt")
        ], 2)

    def test_read_missing_3(self):
        self.execute([
            ("", "job_language=it|invalid=string"),
            ("in", "../tools/res/job_no_config.zip")
        ], 2)

    def test_read_missing_4(self):
        self.execute([
            ("", "wizard"),
            ("in", "../tools/res/job_no_config.zip")
        ], 2)

    def test_read_missing_5(self):
        self.execute([
            ("", "wizard"),
            ("", "job_language=it|invalid=string")
        ], 2)

    def test_read_cannot_read_1(self):
        self.execute([
            ("", "config"),
            ("", "/foo/bar/baz.txt")
        ], 1)

    def test_read_cannot_read_2(self):
        self.execute([
            ("", "container"),
            ("", "/foo/bar/baz.txt")
        ], 1)

    def test_read_cannot_read_3(self):
        self.execute([
            ("", "config"),
            ("", "../tools/res/parsed.txt")
        ], 1)

    def test_read_cannot_read_4(self):
        self.execute([
            ("", "container"),
            ("", "../tools/res/config.txt")
        ], 1)


if __name__ == "__main__":
    unittest.main()
