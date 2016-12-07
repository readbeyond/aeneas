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


# TODO actually parse this file to know what extras
#      (festival, speect, etc.) are available to test
EXTRA_TESTS = os.path.exists(os.path.join(os.path.expanduser("~"), ".aeneas.conf"))


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

    def test_exec_is_audio_file_detect_head_max(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|is_audio_file_detect_head_max=5.000"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_is_audio_file_detect_head_min(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|is_audio_file_detect_head_min=1.000"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_is_audio_file_detect_tail_max(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|is_audio_file_detect_tail_max=5.000"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_is_audio_file_detect_tail_min(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|is_audio_file_detect_tail_min=1.000"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_is_audio_file_head_length(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|is_audio_file_head_length=0.200"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_is_audio_file_process_length(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|is_audio_file_process_length=52.000"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_is_audio_file_tail_length(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|is_audio_file_tail_length=0.200"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_is_text_file_ignore_regex(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|is_text_file_ignore_regex=\\[.*?\\]"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_is_text_file_transliterate_map(self):
        path = gf.absolute_path("res/transliteration/transliteration.map", __file__)
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|is_text_file_transliterate_map=%s" % path),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_is_text_type_plain(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/plain.txt"),
            ("", "task_language=eng|is_text_type=plain|os_task_file_format=json"),
            ("out", "sonnet.json")
        ], 0)

    def test_exec_is_text_type_subtitles(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=json"),
            ("out", "sonnet.json")
        ], 0)

    def test_exec_is_text_type_parsed(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/parsed.txt"),
            ("", "task_language=eng|is_text_type=parsed|os_task_file_format=json"),
            ("out", "sonnet.json")
        ], 0)

    def test_exec_is_text_type_unparsed(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/unparsed.xhtml"),
            ("", "task_language=eng|is_text_type=unparsed|os_task_file_format=json|is_text_unparsed_class_regex=ra|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric"),
            ("out", "sonnet.json")
        ], 0)

    def test_exec_is_text_type_mplain(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/mplain.txt"),
            ("", "task_language=eng|is_text_type=mplain|os_task_file_format=json"),
            ("out", "sonnet.json")
        ], 0)

    def test_exec_is_text_type_munparsed(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/munparsed.xhtml"),
            ("", "task_language=eng|is_text_type=munparsed|os_task_file_format=json|is_text_munparsed_l1_id_regex=p[0-9]+|is_text_munparsed_l2_id_regex=p[0-9]+s[0-9]+|is_text_munparsed_l3_id_regex=p[0-9]+s[0-9]+w[0-9]+"),
            ("out", "sonnet.json")
        ], 0)

    def test_exec_os_task_file_eaf_audio_ref(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=eaf|os_task_file_eaf_audio_ref=audio.mp3"),
            ("out", "sonnet.eaf")
        ], 0)

    def test_exec_os_task_file_head_tail_format_add(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|os_task_file_head_tail_format=add"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_os_task_file_head_tail_format_hidden(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|os_task_file_head_tail_format=hidden"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_os_task_file_head_tail_format_stretch(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|os_task_file_head_tail_format=stretch"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_os_task_file_id_regex(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=xml|os_task_file_id_regex=foo%06d"),
            ("out", "sonnet.xml")
        ], 0)

    def test_exec_os_task_file_smil_audio_ref_os_task_file_smil_page_ref(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=smil|os_task_file_smil_audio_ref=audio.mp3|os_task_file_smil_page_ref=page.xhtml"),
            ("out", "sonnet.smil")
        ], 0)

    def test_exec_task_adjust_boundary_aftercurrent_value(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|task_adjust_boundary_algorithm=aftercurrent|task_adjust_boundary_aftercurrent_value=0.500"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_task_adjust_boundary_beforenext_value(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|task_adjust_boundary_algorithm=beforenext|task_adjust_boundary_beforenext_value=0.500"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_task_adjust_boundary_no_zero(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|task_adjust_boundary_no_zero=True"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_task_adjust_boundary_nonspeech_min_task_adjust_boundary_nonspeech_string(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|task_adjust_boundary_nonspeech_min=0.250|task_adjust_boundary_nonspeech_string=(sil)"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_task_adjust_boundary_offset_value(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|task_adjust_boundary_algorithm=offset|task_adjust_boundary_offset_value=0.500"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_task_adjust_boundary_percent_value(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|task_adjust_boundary_algorithm=percent|task_adjust_boundary_percent_value=90"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_rate_task_adjust_boundary_rate_value(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|task_adjust_boundary_algorithm=rate|task_adjust_boundary_rate_value=21.000"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_rateaggressive_task_adjust_boundary_rate_value(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|task_adjust_boundary_algorithm=rateaggressive|task_adjust_boundary_rate_value=21.000"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_task_custom_id(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|task_custom_id=mytask"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_task_description(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/subtitles.txt"),
            ("", "task_language=eng|is_text_type=subtitles|os_task_file_format=srt|task_description=bla bla bla"),
            ("out", "sonnet.srt")
        ], 0)

    def test_exec_os_task_file_levels_1(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/mplain.txt"),
            ("", "task_language=eng|is_text_type=mplain|os_task_file_format=json|os_task_file_levels=1"),
            ("out", "sonnet.json")
        ], 0)

    def test_exec_os_task_file_levels_2(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/mplain.txt"),
            ("", "task_language=eng|is_text_type=mplain|os_task_file_format=json|os_task_file_levels=2"),
            ("out", "sonnet.json")
        ], 0)

    def test_exec_os_task_file_levels_3(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/mplain.txt"),
            ("", "task_language=eng|is_text_type=mplain|os_task_file_format=json|os_task_file_levels=3"),
            ("out", "sonnet.json")
        ], 0)

    def test_exec_os_task_file_levels_12(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/mplain.txt"),
            ("", "task_language=eng|is_text_type=mplain|os_task_file_format=json|os_task_file_levels=12"),
            ("out", "sonnet.json")
        ], 0)

    def test_exec_os_task_file_levels_23(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/mplain.txt"),
            ("", "task_language=eng|is_text_type=mplain|os_task_file_format=json|os_task_file_levels=23"),
            ("out", "sonnet.json")
        ], 0)

    def test_exec_os_task_file_levels_13(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/mplain.txt"),
            ("", "task_language=eng|is_text_type=mplain|os_task_file_format=json|os_task_file_levels=13"),
            ("out", "sonnet.json")
        ], 0)

    def test_exec_os_task_file_levels_123(self):
        self.execute([
            ("in", "../tools/res/audio.mp3"),
            ("in", "../tools/res/mplain.txt"),
            ("", "task_language=eng|is_text_type=mplain|os_task_file_format=json|os_task_file_levels=123"),
            ("out", "sonnet.json")
        ], 0)


if __name__ == "__main__":
    unittest.main()
