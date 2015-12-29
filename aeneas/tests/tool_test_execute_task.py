#!/usr/bin/env python
# coding=utf-8

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
                params.append(gf.get_abs_path(p_value, __file__))
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
        self.execute([("", "--version")], 2)

    def test_examples(self):
        self.execute([("", "-e")], 2)
        self.execute([("", "--examples")], 2)

    def test_list_parameters(self):
        self.execute([("", "--list-parameters")], 2)

    def test_list_values(self):
        self.execute([("", "--list-values=is_text_type")], 2)

    def test_list_values_bad(self):
        self.execute([("", "--list-values=foo")], 2)

    def test_exec_json(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/plain.txt"),
            ("", "task_language=en|is_text_type=plain|os_task_file_format=json"),
            ("out", "sonnet.json")
        ], 0)

    def test_exec_smil(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/page.xhtml"),
            ("", "task_language=en|is_text_type=unparsed|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric|os_task_file_format=smil|os_task_file_smil_audio_ref=p001.mp3|os_task_file_smil_page_ref=p001.xhtml"),
            ("out", "sonnet.smil")
        ], 0)

    def test_exec_srt(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_srt_html(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt"),
            ("", "--output-html")
        ], 0)

    def test_exec_srt_skip_validator(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt"),
            ("", "--skip-validator")
        ], 0)

    def test_exec_srt_allow_unlisted_language(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en-zz|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt"),
            ("", "--skip-validator"),
            ("", "--allow-unlisted-language")
        ], 0)

    def test_exec_cannot_read_1(self):
        self.execute([
            ("", "/foo/bar/baz.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt")
        ], 1)

    def test_exec_cannot_read_2(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "/foo/bar/baz.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt")
        ], 1)

    def test_exec_cannot_write(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
            ("", "/foo/bar/baz.srt")
        ], 1)

    def test_exec_missing_1(self):
        self.execute([
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt")
        ], 2)

    def test_exec_missing_2(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
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
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt")
        ], 2)

    # NOTE disabling these ones as they require a network connection
    def zzz_test_exec_youtube(self):
        self.execute([
            ("", "https://www.youtube.com/watch?v=rU4a7AA8wM0"),
            ("in", "../tools/res/plain.txt"),
            ("", "task_language=en|is_text_type=plain|os_task_file_format=txt"),
            ("out", "sonnet.txt"),
            ("", "-y")
        ], 0)

    def zzz_test_exec_youtube_largest_audio(self):
        self.execute([
            ("", "https://www.youtube.com/watch?v=rU4a7AA8wM0"),
            ("in", "../tools/res/plain.txt"),
            ("", "task_language=en|is_text_type=plain|os_task_file_format=txt"),
            ("out", "sonnet.txt"),
            ("", "-y"),
            ("", "--largest-audio")
        ], 0)



if __name__ == '__main__':
    unittest.main()



