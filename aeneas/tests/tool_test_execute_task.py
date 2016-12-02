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

from aeneas.tools.execute_task import ExecuteTaskCLI
import aeneas.globalfunctions as gf


class TestExecuteTaskCLI(unittest.TestCase):

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
        exit_code = ExecuteTaskCLI(use_sys=False).run(arguments=params)
        gf.delete_directory(output_path)
        self.assertEqual(exit_code, expected_exit_code)

    def test_help(self):
        self.execute([], 2)
        self.execute([("", "-h")], 2)
        self.execute([("", "--help")], 2)
        self.execute([("", "--help-rconf")], 2)
        self.execute([("", "--version")], 2)

    def test_examples(self):
        self.execute([("", "-e")], 2)
        self.execute([("", "--examples")], 2)

    def test_examples_all(self):
        self.execute([("", "-e")], 2)
        self.execute([("", "--examples-all")], 2)

    def test_list_parameters(self):
        self.execute([("", "--list-parameters")], 2)

    def test_list_values_help(self):
        self.execute([("", "--list-values=?")], 2)

    def test_list_values_bad(self):
        self.execute([("", "--list-values=foo")], 2)

    def test_list_values_aws(self):
        self.execute([("", "--list-values=aws")], 2)

    def test_list_values_espeak(self):
        self.execute([("", "--list-values=espeak")], 2)

    def test_list_values_espeakng(self):
        self.execute([("", "--list-values=espeak-ng")], 2)

    def test_list_values_festival(self):
        self.execute([("", "--list-values=festival")], 2)

    def test_list_values_nuance(self):
        self.execute([("", "--list-values=nuance")], 2)

    def test_list_values_task_language(self):
        self.execute([("", "--list-values=task_language")], 2)

    def test_list_values_is_text_type(self):
        self.execute([("", "--list-values=is_text_type")], 2)

    def test_list_values_is_text_unparsed_id_sort(self):
        self.execute([("", "--list-values=is_text_unparsed_id_sort")], 2)

    def test_list_values_os_task_file_format(self):
        self.execute([("", "--list-values=os_task_file_format")], 2)

    def test_list_values_os_task_file_head_tail_format(self):
        self.execute([("", "--list-values=os_task_file_head_tail_format")], 2)

    def test_list_values_task_adjust_boundary_algorithm(self):
        self.execute([("", "--list-values=task_adjust_boundary_algorithm")], 2)

    def test_exec_srt_max_audio_length(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt"),
            ("", "-r=\"task_max_audio_length=5.0\"")
        ], 1)

    def test_exec_srt_max_text_length(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt"),
            ("", "-r=\"task_max_text_length=5\"")
        ], 1)

    def test_exec_cannot_read_1(self):
        self.execute([
            ("", "/foo/bar/baz.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt")
        ], 1)

    def test_exec_cannot_read_2(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "/foo/bar/baz.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt")
        ], 1)

    def test_exec_cannot_write(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt"),
            ("", "/foo/bar/baz.srt")
        ], 1)

    def test_exec_missing_1(self):
        self.execute([
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt")
        ], 2)

    def test_exec_missing_2(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt")
        ], 2)

    def test_exec_missing_3(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("out", "sonnet.srt")
        ], 2)

    def test_exec_missing_4(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt")
        ], 2)


if __name__ == "__main__":
    unittest.main()
