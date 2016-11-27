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

    def test_help(self):
        self.execute([], 2)
        self.execute([("", "-h")], 2)
        self.execute([("", "--help")], 2)
        self.execute([("", "--help-rconf")], 2)
        self.execute([("", "--version")], 2)

    def test_exec_container_too_many_jobs(self):
        self.execute([
            ("in", "../tools/res/job.zip"),
            ("out", ""),
            ("", "-r=\"job_max_tasks=1\"")
        ], 1)

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

    def test_exec_container_skip_validator_2(self):
        self.execute([
            ("in", "../tools/res/job_no_config.zip"),
            ("out", ""),
            ("", "--skip-validator")
        ], 1)

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


if __name__ == "__main__":
    unittest.main()
