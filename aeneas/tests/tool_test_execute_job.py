#!/usr/bin/env python
# coding=utf-8

import os
import unittest

from aeneas.tools.execute_job import ExecuteJobCLI
import aeneas.globalfunctions as gf

class TestExecuteJobCLI(unittest.TestCase):

    def execute(self, parameters, expected_exit_code):
        output_path = gf.tmp_directory()
        params = ["placeholder"]
        for p_type, p_value in parameters:
            if p_type == "in":
                params.append(gf.get_abs_path(p_value, __file__))
            elif p_type == "out":
                params.append(os.path.join(output_path, p_value))
            else:
                params.append(p_value)
        exit_code = ExecuteJobCLI(use_sys=False).run(arguments=params)
        gf.delete_directory(output_path)
        self.assertEqual(exit_code, expected_exit_code)

    def test_help(self):
        self.execute([], 2)
        self.execute([("", "-h")], 2)
        self.execute([("", "--help")], 2)
        self.execute([("", "--version")], 2)

    def test_exec_container(self):
        self.execute([
            ("in", "../tools/res/job.zip"),
            ("out", "")
        ], 0)

    def test_exec_container_bad_1(self):
        self.execute([
            ("in", "../tools/res/job_no_config.zip"),
            ("out", "")
        ], 1)

    def test_exec_container_bad_2(self):
        self.execute([
            ("in", "../tools/res/job.bad.zip"),
            ("out", "")
        ], 1)

    def test_exec_container_skip_validator_1(self):
        self.execute([
            ("in", "../tools/res/job.zip"),
            ("out", ""),
            ("", "--skip-validator")
        ], 0)

    def test_exec_container_skip_validator_2(self):
        self.execute([
            ("in", "../tools/res/job_no_config.zip"),
            ("out", ""),
            ("", "--skip-validator")
        ], 1)

    def test_exec_container_wizard(self):
        self.execute([
            ("in", "../tools/res/job_no_config.zip"),
            ("out", ""),
            ("", "is_hierarchy_type=flat|is_hierarchy_prefix=assets/|is_text_file_relative_path=.|is_text_file_name_regex=.*\.xhtml|is_text_type=unparsed|is_audio_file_relative_path=.|is_audio_file_name_regex=.*\.mp3|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric|os_job_file_name=demo_sync_job_output|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=assets/|os_task_file_name=\$PREFIX.xhtml.smil|os_task_file_format=smil|os_task_file_smil_page_ref=\$PREFIX.xhtml|os_task_file_smil_audio_ref=../Audio/\$PREFIX.mp3|job_language=en|job_description=Demo Sync Job")
        ], 0)

    def test_exec_missing_1(self):
        self.execute([
            ("in", "../tools/res/job.zip")
        ], 2)

    def test_exec_missing_2(self):
        self.execute([
            ("out", "")
        ], 2)

    def test_exec_cannot_read(self):
        self.execute([
            ("in", "/foo/bar/baz"),
            ("out", "")
        ], 1)

    def test_exec_cannot_write(self):
        self.execute([
            ("in", "../tools/res/job.zip"),
            ("", "/foo/bar/baz.txt")
        ], 1)



if __name__ == '__main__':
    unittest.main()



