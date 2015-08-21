#!/usr/bin/env python
# coding=utf-8

import os
import sys
import unittest

from . import get_abs_path

from aeneas.logger import Logger
from aeneas.validator import Validator

class TestValidator(unittest.TestCase):

    def test_check_string_encoding_false(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = u"abcdé".encode("latin-1")
        self.assertFalse(validator._check_string_encoding(string))

    def test_check_string_encoding_true(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = u"abcdé".encode("utf-8")
        self.assertTrue(validator._check_string_encoding(string))

    def test_check_reserved_characters_false(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "string with ~ reserved char"
        self.assertFalse(validator._check_reserved_characters(string))

    def test_check_reserved_characters_true(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "string without reserved char"
        self.assertTrue(validator._check_reserved_characters(string))

    def test_check_file_encoding_false_01(self):
        logger = Logger()
        validator = Validator(logger=logger)
        input_file_path = get_abs_path("res/validator/encoding_iso8859.txt")
        result = validator.check_file_encoding(input_file_path)
        self.assertFalse(result.passed)

    def test_check_file_encoding_false_02(self):
        logger = Logger()
        validator = Validator(logger=logger)
        input_file_path = get_abs_path("res/validator/encoding_utf32.xhtml")
        result = validator.check_file_encoding(input_file_path)
        self.assertFalse(result.passed)

    def test_check_file_encoding_true_01(self):
        logger = Logger()
        validator = Validator(logger=logger)
        input_file_path = get_abs_path("res/validator/encoding_utf8.xhtml")
        result = validator.check_file_encoding(input_file_path)
        self.assertTrue(result.passed)

    def test_check_file_encoding_true_02(self):
        logger = Logger()
        validator = Validator(logger=logger)
        input_file_path = get_abs_path("res/validator/encoding_utf8_bom.xhtml")
        result = validator.check_file_encoding(input_file_path)
        self.assertTrue(result.passed)

    def test_check_job_configuration_01(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = u"dummy config string with bad encoding é".encode("latin-1")
        result = validator.check_job_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_job_configuration_02(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "dummy config string with ~ reserved characters"
        result = validator.check_job_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_job_configuration_03(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "malformed config string"
        result = validator.check_job_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_job_configuration_04(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "=malformed"
        result = validator.check_job_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_job_configuration_05(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "malformed="
        result = validator.check_job_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_job_configuration_06(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "not=relevant|config=string"
        result = validator.check_job_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_job_configuration_07(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "job_language=it|missing=other"
        result = validator.check_job_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_job_configuration_08(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "job_language=it|os_job_file_name=output.zip"
        result = validator.check_job_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_job_configuration_09(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "job_language=it|os_job_file_name=output.zip|os_job_file_container=zip"
        result = validator.check_job_configuration(string)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)

    def test_check_job_configuration_10(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "job_language=zzzz|os_job_file_name=output.zip|os_job_file_container=zip"
        result = validator.check_job_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_job_configuration_11(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "job_language=it|os_job_file_name=output.zip|os_job_file_container=zzzzzz"
        result = validator.check_job_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_job_configuration_12(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "job_language=it|os_job_file_name=output.zip|os_job_file_container=zip|is_hierarchy_type=zzzzzz"
        result = validator.check_job_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_job_configuration_13(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "job_language=it|os_job_file_name=output.zip|os_job_file_container=zip|is_hierarchy_type=flat"
        result = validator.check_job_configuration(string)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)

    def test_check_job_configuration_14(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "job_language=it|os_job_file_name=output.zip|os_job_file_container=zip|is_hierarchy_type=paged"
        result = validator.check_job_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_job_configuration_15(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "job_language=it|os_job_file_name=output.zip|os_job_file_container=zip|is_hierarchy_type=paged|is_task_dir_name_regex=[0-9]*"
        result = validator.check_job_configuration(string)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)

    def test_check_job_configuration_16(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "job_language=it|os_job_file_name=output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=zzzzzz"
        result = validator.check_job_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_job_configuration_17(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "job_language=it|os_job_file_name=output.zip|os_job_file_container=zip|os_job_file_hierarchy_type=flat"
        result = validator.check_job_configuration(string)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)

    def test_check_task_configuration_01(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = u"dummy config string with bad encoding é".encode("latin-1")
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_02(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "dummy config string with ~ reserved characters"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_03(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "malformed config string"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_04(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "=malformed"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_05(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "malformed="
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_06(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "not=relevant|config=string"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_07(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|missing=other"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_08(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|missing=other"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_09(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|missing=other"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_10(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt"
        result = validator.check_task_configuration(string)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)

    def test_check_task_configuration_11(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=zzzz|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_12(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=zzzzzz|os_task_file_name=output.txt|os_task_file_format=txt"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_13(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=zzzzzz"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_14(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=unparsed|os_task_file_name=output.txt|os_task_file_format=txt"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_15(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=unparsed|is_text_unparsed_class_regex=ra|is_text_unparsed_id_sort=numeric|os_task_file_name=output.txt|os_task_file_format=txt"
        result = validator.check_task_configuration(string)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)

    def test_check_task_configuration_16(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=unparsed|is_text_unparsed_id_regex=f[0-9]*|is_text_unparsed_id_sort=numeric|os_task_file_name=output.txt|os_task_file_format=txt"
        result = validator.check_task_configuration(string)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)

    def test_check_task_configuration_17(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=unparsed|is_text_unparsed_class_regex=ra|is_text_unparsed_id_regex=f[0-9]*|is_text_unparsed_id_sort=numeric|os_task_file_name=output.txt|os_task_file_format=txt"
        result = validator.check_task_configuration(string)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)

    def test_check_task_configuration_18(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=unparsed|is_text_unparsed_id_sort=numeric|os_task_file_name=output.txt|os_task_file_format=txt"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_19(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=unparsed|is_text_unparsed_class_regex=ra|is_text_unparsed_id_regex=f[0-9]*|os_task_file_name=output.txt|os_task_file_format=txt"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_20(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=smil|os_task_file_smil_page_ref=page.xhtml|os_task_file_smil_audio_ref=../Audio/audio.mp3"
        result = validator.check_task_configuration(string)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)

    def test_check_task_configuration_21(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=smil|os_task_file_smil_page_ref=page.xhtml"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_22(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=smil|os_task_file_smil_audio_ref=../Audio/audio.mp3"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_23(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=smil"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_24(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=auto"
        result = validator.check_task_configuration(string)
        self.assertTrue(result.passed)

    def test_check_task_configuration_25(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=foo"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_26(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=rate"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_27(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=rate|task_adjust_boundary_rate_value=21"
        result = validator.check_task_configuration(string)
        self.assertTrue(result.passed)

    def test_check_task_configuration_28(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=percent"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_29(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=percent|task_adjust_boundary_percent_value=50"
        result = validator.check_task_configuration(string)
        self.assertTrue(result.passed)

    def test_check_task_configuration_30(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=aftercurrent"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_31(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=aftercurrent|task_adjust_boundary_aftercurrent_value=0.200"
        result = validator.check_task_configuration(string)
        self.assertTrue(result.passed)

    def test_check_task_configuration_32(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=beforenext"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_33(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=beforenext|task_adjust_boundary_beforenext_value=0.200"
        result = validator.check_task_configuration(string)
        self.assertTrue(result.passed)

    def test_check_task_configuration_34(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=rateagressive"
        result = validator.check_task_configuration(string)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_task_configuration_35(self):
        logger = Logger()
        validator = Validator(logger=logger)
        string = "task_language=it|is_text_type=plain|os_task_file_name=output.txt|os_task_file_format=txt|task_adjust_boundary_algorithm=rateaggressive|task_adjust_boundary_rate_value=21"
        result = validator.check_task_configuration(string)
        self.assertTrue(result.passed)

    def test_check_container_txt_01(self):
        logger = Logger()
        validator = Validator(logger=logger)
        container_path = get_abs_path("res/validator/job_txt_config")
        result = validator.check_container(container_path)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)

    def test_check_container_txt_02(self):
        logger = Logger()
        validator = Validator(logger=logger)
        container_path = get_abs_path("res/validator/job_no_config")
        result = validator.check_container(container_path)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_container_txt_03(self):
        logger = Logger()
        validator = Validator(logger=logger)
        container_path = get_abs_path("res/validator/job_txt_config_bad_1")
        result = validator.check_container(container_path)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_container_txt_04(self):
        logger = Logger()
        validator = Validator(logger=logger)
        container_path = get_abs_path("res/validator/job_txt_config_bad_2")
        result = validator.check_container(container_path)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_container_txt_05(self):
        logger = Logger()
        validator = Validator(logger=logger)
        container_path = get_abs_path("res/validator/job_txt_config_bad_3")
        result = validator.check_container(container_path)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)
    
    def test_check_container_txt_06(self):
        logger = Logger()
        validator = Validator(logger=logger)
        container_path = get_abs_path("res/validator/job_txt_config_not_root")
        result = validator.check_container(container_path)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)
    
    def test_check_container_txt_07(self):
        logger = Logger()
        validator = Validator(logger=logger)
        container_path = get_abs_path("res/validator/job_txt_config_not_root_nested")
        result = validator.check_container(container_path)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)

    def test_check_container_xml_01(self):
        logger = Logger()
        validator = Validator(logger=logger)
        container_path = get_abs_path("res/validator/job_xml_config")
        result = validator.check_container(container_path)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)

    def test_check_container_xml_02(self):
        logger = Logger()
        validator = Validator(logger=logger)
        container_path = get_abs_path("res/validator/job_xml_config_bad_1")
        result = validator.check_container(container_path)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_container_xml_03(self):
        logger = Logger()
        validator = Validator(logger=logger)
        container_path = get_abs_path("res/validator/job_xml_config_bad_2")
        result = validator.check_container(container_path)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_container_xml_04(self):
        logger = Logger()
        validator = Validator(logger=logger)
        container_path = get_abs_path("res/validator/job_xml_config_bad_3")
        result = validator.check_container(container_path)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_container_xml_05(self):
        logger = Logger()
        validator = Validator(logger=logger)
        container_path = get_abs_path("res/validator/job_xml_config_bad_4")
        result = validator.check_container(container_path)
        self.assertFalse(result.passed)
        self.assertGreater(len(result.errors), 0)

    def test_check_container_xml_06(self):
        logger = Logger()
        validator = Validator(logger=logger)
        container_path = get_abs_path("res/validator/job_xml_config_not_root")
        result = validator.check_container(container_path)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)

    def test_check_container_xml_07(self):
        logger = Logger()
        validator = Validator(logger=logger)
        container_path = get_abs_path("res/validator/job_xml_config_not_root_nested")
        result = validator.check_container(container_path)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.errors), 0)

if __name__ == '__main__':
    unittest.main()



