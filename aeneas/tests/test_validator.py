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

from aeneas.logger import Logger
from aeneas.validator import Validator
import aeneas.globalfunctions as gf


class TestValidator(unittest.TestCase):

    def string_well_formed(self, bstring, expected):
        validator = Validator()
        validator.check_raw_string(bstring, is_bstring=True)
        self.assertEqual(validator.result.passed, expected)

    def file_encoding(self, path, expected):
        validator = Validator()
        result = validator.check_file_encoding(gf.absolute_path(path, __file__))
        self.assertEqual(result.passed, expected)

    def jc(self, string, expected):
        validator = Validator()
        result = validator.check_configuration_string(
            string,
            is_job=True,
            external_name=False
        )
        self.assertEqual(result.passed, expected)
        if expected:
            self.assertEqual(len(result.errors), 0)
        else:
            self.assertGreater(len(result.errors), 0)

    def tc(self, string, expected):
        validator = Validator()
        result = validator.check_configuration_string(
            string,
            is_job=False,
            external_name=False
        )
        self.assertEqual(result.passed, expected)
        if expected:
            self.assertEqual(len(result.errors), 0)
        else:
            self.assertGreater(len(result.errors), 0)

    def container(self, path, expected):
        validator = Validator()
        result = validator.check_container(gf.absolute_path(path, __file__))
        self.assertEqual(result.passed, expected)
        if expected:
            self.assertEqual(len(result.errors), 0)
        else:
            self.assertGreater(len(result.errors), 0)

    def test_string_well_formed_none(self):
        self.string_well_formed(None, False)

    def test_string_well_formed_unicode(self):
        self.string_well_formed(u"Unicode string should fail", False)

    def test_string_well_formed_zero_length(self):
        self.string_well_formed(b"", False)

    def test_string_well_formed_latin(self):
        self.string_well_formed(u"abcdé".encode("latin-1"), False)

    def test_string_well_formed_uft8(self):
        self.string_well_formed(u"abcdé".encode("utf-8"), True)

    def test_string_reserved_character_yes(self):
        self.string_well_formed(b"string with ~ reserved char", False)

    def test_string_reserved_character_no(self):
        self.string_well_formed(b"string without reserved char", True)

    def test_check_file_none(self):
        self.file_encoding(None, False)

    def test_check_file_empty(self):
        self.file_encoding("res/validator/empty.txt", True)

    def test_check_file_encoding_iso8859(self):
        self.file_encoding("res/validator/encoding_iso8859.txt", False)

    def test_check_file_encoding_utf32(self):
        self.file_encoding("res/validator/encoding_utf32.xhtml", False)

    def test_check_file_encoding_utf8(self):
        self.file_encoding("res/validator/encoding_utf8.xhtml", True)

    def test_check_file_encoding_utf8_bom(self):
        self.file_encoding("res/validator/encoding_utf8_bom.xhtml", True)

    def test_check_jc_reserved_characters(self):
        self.jc(u"dummy config string with ~ reserved characters", False)

    def test_check_jc_bad_encoding(self):
        self.jc(u"config string with bad encoding: à".encode("latin-1"), False)

    def test_check_jc_malformed_string(self):
        self.jc(u"malformed config string", False)

    def test_check_jc_no_key(self):
        self.jc(u"=malformed", False)

    def test_check_jc_no_value(self):
        self.jc(u"malformed=", False)

    def test_check_jc_invalid_keys(self):
        self.jc(u"not=relevant|config=string", False)

    def test_check_jc_valid(self):
        self.jc(u"job_language=it|os_job_file_name=output.zip|os_job_file_container=zip", True)

    def test_check_jc_missing_required_job_language(self):
        self.jc(u"os_job_file_name=output.zip|os_job_file_container=zip", False)

    def test_check_jc_missing_required_os_job_file_container(self):
        self.jc(u"job_language=it|os_job_file_name=output.zip", False)

    def test_check_jc_missing_required_os_job_file_name(self):
        self.jc(u"job_language=it|os_job_file_container=zip", False)

    # language is no longer checked
    # COMMENTED def test_check_jc_invalid_value_job_language(self):
    # COMMENTED   self.jc(u"job_language=zzzz|os_job_file_name=output.zip|os_job_file_container=zip", False)

    def test_check_jc_invalid_value_os_job_file_container(self):
        self.jc(u"job_language=it|os_job_file_name=output.zip|os_job_file_container=zzzzzz", False)

    def test_check_jc_invalid_value_is_hierarchy_type(self):
        self.jc(u"job_language=it|os_job_file_name=output.zip|os_job_file_container=zip|is_hierarchy_type=zzzzzz", False)

    def test_check_jc_valid_flat(self):
        self.jc(u"job_language=it|os_job_file_name=output.zip|os_job_file_container=zip|is_hierarchy_type=flat", True)

    def test_check_jc_valid_paged_with_required(self):
        self.jc(u"job_language=it|os_job_file_name=output.zip|os_job_file_container=zip|is_hierarchy_type=paged|is_task_dir_name_regex=[0-9]*", True)

    def test_check_jc_missing_paged_required_is_task_dir_name_regex(self):
        self.jc(u"job_language=it|os_job_file_name=output.zip|os_job_file_container=zip|is_hierarchy_type=paged", False)

    def test_check_jc_invalid_value_os_job_file_hierarchy_type(self):
        self.jc(u"job_language=it|os_job_file_name=output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=zzzzzz", False)

    def test_check_jc_valid_os_flat(self):
        self.jc(u"job_language=it|os_job_file_name=output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=flat", True)

    def test_check_jc_valid_os_paged(self):
        self.jc(u"job_language=it|os_job_file_name=output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=paged", True)

    def test_check_tc_bad_encoding(self):
        self.tc(u"config string with bad encoding: à".encode("latin-1"), False)

    def test_check_tc_reserved_characters(self):
        self.tc(u"dummy config string with ~ reserved characters", False)

    def test_check_tc_malformed(self):
        self.tc(u"malformed config string", False)

    def test_check_tc_no_key(self):
        self.tc(u"=malformed", False)

    def test_check_tc_no_value(self):
        self.tc(u"malformed=", False)

    def test_check_tc_invalid_keys(self):
        self.tc(u"not=relevant|config=string", False)

    def test_check_tc_valid(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt", True)

    def test_check_tc_missing_required_task_language(self):
        self.tc(u"is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt", False)

    def test_check_tc_missing_required_is_text_type(self):
        self.tc(u"task_language=it|os_task_file_name=output.txt|os_task_file_format=txt", False)

    def test_check_tc_missing_required_os_task_file_name(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_format=txt", False)

    def test_check_tc_missing_required_os_task_file_format(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt", False)

    # language is no longer checked
    # COMMENTED def test_check_tc_invalid_value_task_language(self):
    # COMMENTED   self.tc(u"task_language=zzzz|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt", False)

    def test_check_tc_invalid_value_is_text_type(self):
        self.tc(u"task_language=it|is_text_type=zzzzzz|os_task_file_name=output.txt|os_task_file_format=txt", False)

    def test_check_tc_invalid_value_os_task_file_format(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=zzzzzz", False)

    def test_check_tc_missing_unparsed_required_is_text_unparsed_class_regex(self):
        self.tc(u"task_language=it|is_text_type=unparsed|os_task_file_name=output.txt|os_task_file_format=txt", False)

    def test_check_tc_valid_unparsed_is_text_unparsed_class_regex(self):
        self.tc(u"task_language=it|is_text_type=unparsed|is_text_unparsed_class_regex=ra|is_text_unparsed_id_sort=numeric|os_task_file_name=output.txt|os_task_file_format=txt", True)

    def test_check_tc_valid_unparsed_is_text_unparsed_id_regex(self):
        self.tc(u"task_language=it|is_text_type=unparsed|is_text_unparsed_id_regex=f[0-9]*|is_text_unparsed_id_sort=numeric|os_task_file_name=output.txt|os_task_file_format=txt", True)

    def test_check_tc_valid_unparsed_both(self):
        self.tc(u"task_language=it|is_text_type=unparsed|is_text_unparsed_class_regex=ra|is_text_unparsed_id_regex=f[0-9]*|is_text_unparsed_id_sort=numeric|os_task_file_name=output.txt|os_task_file_format=txt", True)

    def test_check_tc_invalid_value_is_text_unparsed_id_sort(self):
        self.tc(u"task_language=it|is_text_type=unparsed|is_text_unparsed_class_regex=ra|is_text_unparsed_id_regex=f[0-9]*|is_text_unparsed_id_sort=foo|os_task_file_name=output.txt|os_task_file_format=txt", False)

    def test_check_tc_missing_required_is_text_unparsed_id_sort(self):
        self.tc(u"task_language=it|is_text_type=unparsed|is_text_unparsed_class_regex=ra|is_text_unparsed_id_regex=f[0-9]*|os_task_file_name=output.txt|os_task_file_format=txt", False)

    def test_check_tc_valid_smil(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=smil|os_task_file_smil_page_ref=page.xhtml|os_task_file_smil_audio_ref=../Audio/audio.mp3", True)

    def test_check_tc_missing_smil_required_os_task_file_smil_audio_ref(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=smil|os_task_file_smil_page_ref=page.xhtml", False)

    def test_check_tc_missing_smil_required_os_task_file_smil_page_ref(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=smil|os_task_file_smil_audio_ref=../Audio/audio.mp3", False)

    def test_check_tc_missing_smil_required_both(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=smil", False)

    def test_check_tc_valid_aba_auto(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=auto", True)

    def test_check_tc_invalid_aba_value_task_adjust_boundary_algorithm(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=foo", False)

    def test_check_tc_missing_aba_rate_required_task_adjust_boundary_rate_value(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=rate", False)

    def test_check_tc_valid_aba_rate(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=rate|task_adjust_boundary_rate_value=21", True)

    def test_check_tc_missing_aba_percent_required_task_adjust_boundary_percent_value(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=percent", False)

    def test_check_tc_valid_aba_percent(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=percent|task_adjust_boundary_percent_value=50", True)

    def test_check_tc_missing_aba_aftercurrent_required_task_adjust_boundary_aftercurrent_value(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=aftercurrent", False)

    def test_check_tc_valid_aba_aftercurrent(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=aftercurrent|task_adjust_boundary_aftercurrent_value=0.200", True)

    def test_check_tc_missing_aba_beforenext_required_task_adjust_boundary_beforenext_value(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=beforenext", False)

    def test_check_tc_valid_aba_beforenext(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=beforenext|task_adjust_boundary_beforenext_value=0.200", True)

    def test_check_tc_missing_aba_rateaggressive_required_task_adjust_boundary_rate_value(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=rateagressive", False)

    def test_check_tc_valid_aba_rateaggressive(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=rateaggressive|task_adjust_boundary_rate_value=21", True)

    def test_check_tc_missing_aba_offset_required_task_adjust_boundary_offset_value(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=offset", False)

    def test_check_tc_valid_aba_offset(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=offset|task_adjust_boundary_offset_value=0.200", True)

    def test_check_tc_invalid_value_os_task_file_head_tail_format(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|os_task_file_head_tail_format=foo", False)

    def test_check_tc_valid_head_tail_format(self):
        self.tc(u"task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|os_task_file_head_tail_format=add", True)

    def test_check_container_txt_valid(self):
        self.container("res/validator/job_txt_config", True)

    def test_check_container_none(self):
        self.container(None, False)

    def test_check_container_not_existing(self):
        self.container("res/validator/x/y/z/not_existing", False)

    def test_check_container_empty(self):
        self.container("res/validator/job_empty", False)

    def test_check_container_empty_zip(self):
        self.container("res/validator/empty.zip", False)

    def test_check_container_txt_no_config(self):
        self.container("res/validator/job_no_config", False)

    def test_check_container_txt_bad_config_01(self):
        self.container("res/validator/job_txt_config_bad_1", False)

    def test_check_container_txt_bad_config_02(self):
        self.container("res/validator/job_txt_config_bad_2", False)

    def test_check_container_txt_bad_config_03(self):
        self.container("res/validator/job_txt_config_bad_3", False)

    def test_check_container_txt_not_root(self):
        self.container("res/validator/job_txt_config_not_root", True)

    def test_check_container_txt_not_root_nested(self):
        self.container("res/validator/job_txt_config_not_root_nested", True)

    def test_check_container_xml_valid(self):
        self.container("res/validator/job_xml_config", True)

    def test_check_container_xml_bad_config_01(self):
        self.container("res/validator/job_xml_config_bad_1", False)

    def test_check_container_xml_bad_config_02(self):
        self.container("res/validator/job_xml_config_bad_2", False)

    def test_check_container_xml_bad_config_03(self):
        self.container("res/validator/job_xml_config_bad_3", False)

    def test_check_container_xml_bad_config_04(self):
        self.container("res/validator/job_xml_config_bad_4", False)

    def test_check_container_xml_not_root(self):
        self.container("res/validator/job_xml_config_not_root", True)

    def test_check_container_xml_not_root_nested(self):
        self.container("res/validator/job_xml_config_not_root_nested", True)


if __name__ == "__main__":
    unittest.main()
