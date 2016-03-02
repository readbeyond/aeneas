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
            ("", "-r=\"allow_unlisted_languages=True\"")
        ], 0)

    def test_exec_srt_pure(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt"),
            ("", "-r=\"c_extensions=False\"")
        ], 0)

    def test_exec_srt_cew_subprocess(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt"),
            ("", "-r=\"cew_subprocess_enabled=True\"")
        ], 0)

    def test_exec_srt_head(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt|is_audio_file_head_length=5.000"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_srt_tail(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt|is_audio_file_tail_length=5.000"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_srt_process(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt|is_audio_file_process_length=40.000"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_srt_head_process(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt|is_audio_file_process_length=40.000|is_audio_file_head_length=5.000"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_srt_detect_head(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt|is_audio_file_detect_head_min=0|is_audio_file_detect_head_max=10.000"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_srt_detect_tail(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt|is_audio_file_detect_tail_min=0|is_audio_file_detect_tail_max=10.000"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_srt_detect_head_tail(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt|is_audio_file_detect_head_min=0|is_audio_file_detect_head_max=10.000|is_audio_file_detect_tail_min=0|is_audio_file_detect_tail_max=10.000"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_srt_aba_aftercurrent(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt|task_adjust_boundary_algorithm=aftercurrent|task_adjust_boundary_aftercurrent_value=0.200"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_srt_aba_beforenext(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt|task_adjust_boundary_algorithm=beforenext|task_adjust_boundary_beforenext_value=0.200"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_srt_aba_offset(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt|task_adjust_boundary_algorithm=offset|task_adjust_boundary_offset_value=0.200"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_srt_aba_percent(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt|task_adjust_boundary_algorithm=percent|task_adjust_boundary_percent_value=50"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_srt_aba_rate(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt|task_adjust_boundary_algorithm=rate|task_adjust_boundary_rate_value=21.0"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_srt_aba_rateaggressive(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt|task_adjust_boundary_algorithm=rateaggressive|task_adjust_boundary_rate_value=21.0"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_srt_ignore_regex(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt|is_text_file_ignore_regex=\\[.*?\\]"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_json_id_regex(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=json|os_task_file_id_regex=Word%03d"),
            ("out", "sonnet.json")
        ], 0)

    def test_exec_srt_transmap(self):
        path = gf.absolute_path("res/transliteration/transliteration.map", __file__)
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt|is_text_file_transliterate_map=%s" % path),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_srt_dtw_margin(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt"),
            ("", "-r=\"dtw_margin=30\"")
        ], 0)

    def test_exec_srt_dtw_algorithm(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt"),
            ("", "-r=\"c_extensions=False|dtw_algorithm=exact\"")
        ], 0)

    def test_exec_srt_mfcc_window_shift(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt"),
            ("", "-r=\"mfcc_window_length=0.250|mfcc_window_shift=0.100\"")
        ], 0)

    def test_exec_srt_path(self):
        home = os.path.expanduser("~")
        espeak_path = os.path.join(home, ".bin/myespeak")
        ffmpeg_path = os.path.join(home, ".bin/myffmpeg")
        ffprobe_path = os.path.join(home, ".bin/myffprobe")
        if gf.file_exists(espeak_path) and gf.file_exists(ffmpeg_path) and gf.file_exists(ffprobe_path):
            self.execute([
                ("in", "../tools/res/audio.mp3"),
                ("in", "../tools/res/subtitles.txt"),
                ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
                ("out", "sonnet.srt"),
                ("", "-r=\"espeak_path=%s|ffmpeg_path=%s|ffprobe_path=%s\"" % (espeak_path, ffmpeg_path, ffprobe_path))
            ], 0)

    def test_exec_srt_tmp_path(self):
        tmp_path = gf.tmp_directory()
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt"),
            ("", "-r=\"tmp_path=%s\"" % (tmp_path))
        ], 0)
        gf.delete_directory(tmp_path)

    def test_exec_srt_max_audio_length(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt"),
            ("", "-r=\"task_max_audio_length=5.0\"")
        ], 1)

    def test_exec_srt_max_text_length(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=en|is_text_type=subtitles|os_task_file_format=srt"),
            ("out", "sonnet.srt"),
            ("", "-r=\"task_max_text_length=5\"")
        ], 1)

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



