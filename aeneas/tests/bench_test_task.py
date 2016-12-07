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

import multiprocessing
import os
import unittest

from aeneas.tools.execute_task import ExecuteTaskCLI
import aeneas.globalfunctions as gf


# TODO actually parse this file to know what extras
#      (festival, speect, etc.) are available to test
EXTRA_TESTS = os.path.exists(os.path.join(os.path.expanduser("~"), ".aeneas.conf"))
BENCH_DIR = os.path.join(os.path.expanduser("~"), ".aeneas", "benchmark_input")
BENCH_TESTS = os.path.exists(BENCH_DIR)


class TestBenchmarkExecuteTaskCLI(unittest.TestCase):

    def bench_execute(self, parameters, expected_exit_code, timeout):
        if not BENCH_TESTS:
            return
        args = (parameters, expected_exit_code)
        p = multiprocessing.Process(target=self.execute, name="funcExecute", args=args)
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            self.assertTrue(False)

    def execute(self, parameters, expected_exit_code):
        output_path = gf.tmp_directory()
        params = ["placeholder"]
        for p_type, p_value in parameters:
            if p_type == "in":
                params.append(os.path.join(BENCH_DIR, p_value))
            elif p_type == "out":
                params.append(os.path.join(output_path, p_value))
            else:
                params.append(p_value)
        exit_code = ExecuteTaskCLI(use_sys=False).run(arguments=params)
        gf.delete_directory(output_path)
        self.assertEqual(exit_code, expected_exit_code)

    def test_001_mplain(self):
        self.bench_execute([
            ("in", "001m.mp3"),
            ("in", "001m.mplain.txt"),
            ("", "task_language=eng|is_text_type=mplain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 20)

    def test_001_plain_sentence(self):
        self.bench_execute([
            ("in", "001m.mp3"),
            ("in", "001m.plain.sentence.txt"),
            ("", "task_language=eng|is_text_type=plain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 5)

    def test_001_plain_word(self):
        self.bench_execute([
            ("in", "001m.mp3"),
            ("in", "001m.plain.word.txt"),
            ("", "task_language=eng|is_text_type=plain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 5)

    def test_005_mplain(self):
        self.bench_execute([
            ("in", "005m.mp3"),
            ("in", "005m.mplain.txt"),
            ("", "task_language=eng|is_text_type=mplain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 100)

    def test_005_plain_sentence(self):
        self.bench_execute([
            ("in", "005m.mp3"),
            ("in", "005m.plain.sentence.txt"),
            ("", "task_language=eng|is_text_type=plain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 20)

    def test_005_plain_word(self):
        self.bench_execute([
            ("in", "005m.mp3"),
            ("in", "005m.plain.word.txt"),
            ("", "task_language=eng|is_text_type=plain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 20)

    def test_010_mplain(self):
        self.bench_execute([
            ("in", "010m.mp3"),
            ("in", "010m.mplain.txt"),
            ("", "task_language=eng|is_text_type=mplain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 200)

    def test_010_plain_sentence(self):
        self.bench_execute([
            ("in", "010m.mp3"),
            ("in", "010m.plain.sentence.txt"),
            ("", "task_language=eng|is_text_type=plain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 40)

    def test_010_plain_word(self):
        self.bench_execute([
            ("in", "010m.mp3"),
            ("in", "010m.plain.word.txt"),
            ("", "task_language=eng|is_text_type=plain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 40)

    def test_020_mplain(self):
        self.bench_execute([
            ("in", "020m.mp3"),
            ("in", "020m.mplain.txt"),
            ("", "task_language=eng|is_text_type=mplain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 400)

    def test_020_plain_sentence(self):
        self.bench_execute([
            ("in", "020m.mp3"),
            ("in", "020m.plain.sentence.txt"),
            ("", "task_language=eng|is_text_type=plain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 60)

    def test_020_plain_word(self):
        self.bench_execute([
            ("in", "020m.mp3"),
            ("in", "020m.plain.word.txt"),
            ("", "task_language=eng|is_text_type=plain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 60)

    def test_030_mplain(self):
        self.bench_execute([
            ("in", "030m.mp3"),
            ("in", "030m.mplain.txt"),
            ("", "task_language=eng|is_text_type=mplain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 600)

    def test_030_plain_sentence(self):
        self.bench_execute([
            ("in", "030m.mp3"),
            ("in", "030m.plain.sentence.txt"),
            ("", "task_language=eng|is_text_type=plain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 90)

    def test_030_plain_word(self):
        self.bench_execute([
            ("in", "030m.mp3"),
            ("in", "030m.plain.word.txt"),
            ("", "task_language=eng|is_text_type=plain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 90)

    def test_060_mplain(self):
        self.bench_execute([
            ("in", "060m.mp3"),
            ("in", "060m.mplain.txt"),
            ("", "task_language=eng|is_text_type=mplain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 1200)

    def test_060_plain_sentence(self):
        self.bench_execute([
            ("in", "060m.mp3"),
            ("in", "060m.plain.sentence.txt"),
            ("", "task_language=eng|is_text_type=plain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 180)

    def test_060_plain_word(self):
        self.bench_execute([
            ("in", "060m.mp3"),
            ("in", "060m.plain.word.txt"),
            ("", "task_language=eng|is_text_type=plain|os_task_file_format=json"),
            ("out", "out.json"),
        ], 0, 180)


if __name__ == "__main__":
    unittest.main()
