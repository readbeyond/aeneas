#!/usr/bin/env python
# coding=utf-8

import unittest

from aeneas.analyzecontainer import AnalyzeContainer
from aeneas.container import Container
import aeneas.tests as at

class TestAnalyzeContainer(unittest.TestCase):

    FILES = [
        {
            "path": "res/validator/job_txt_config",
            "length": 3
        },
        {
            "path": "res/validator/job_txt_config_not_root",
            "length": 3
        },
        {
            "path": "res/validator/job_txt_config_not_root_nested",
            "length": 3
        },
        {
            "path": "res/validator/job_txt_config_paged_1",
            "length": 3
        },
        {
            "path": "res/validator/job_txt_config_paged_2",
            "length": 3
        },
        {
            "path": "res/validator/job_txt_config_paged_3",
            "length": 3
        },
        {
            "path": "res/validator/job_xml_config",
            "length": 3
        },
        {
            "path": "res/validator/job_xml_config_not_root",
            "length": 3
        },
        {
            "path": "res/validator/job_xml_config_not_root_nested",
            "length": 3
        },
    ]

    CONFIG_STRING = "is_hierarchy_type=flat|is_hierarchy_prefix=assets/|is_text_file_relative_path=.|is_text_file_name_regex=.*\.xhtml|is_text_type=unparsed|is_audio_file_relative_path=.|is_audio_file_name_regex=.*\.mp3|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric|os_job_file_name=demo_sync_job_output|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=assets/|os_task_file_name=$PREFIX.xhtml.smil|os_task_file_format=smil|os_task_file_smil_page_ref=$PREFIX.xhtml|os_task_file_smil_audio_ref=../Audio/$PREFIX.mp3|job_language=en|job_description=Demo Sync Job"

    EMPTY_CONTAINERS = [
        "res/container/empty_dir",
        "res/container/empty_file.epub",
        "res/container/empty_file.tar",
        "res/container/empty_file.tar.bz2",
        "res/container/empty_file.tar.gz",
        "res/container/empty_file.zip"
    ]

    INVALID_CONTAINERS = [
        "res/validator/job_empty",
        "res/validator/job_no_config",
        "res/validator/job_no_task_assets",
        "res/validator/job_txt_config_bad_1",
        "res/validator/job_txt_config_bad_2",
        "res/validator/job_txt_config_bad_3",
        "res/validator/job_xml_config_bad_1",
        "res/validator/job_xml_config_bad_2",
        "res/validator/job_xml_config_bad_3",
        "res/validator/job_xml_config_bad_4"
    ]

    NOT_EXISTING_PATH = at.get_abs_path("res/validator/x/y/z/not_existing")

    def test_none(self):
        with self.assertRaises(TypeError):
            analyzer = AnalyzeContainer(None)

    def test_not_container(self):
        with self.assertRaises(TypeError):
            analyzer = AnalyzeContainer(self.NOT_EXISTING_PATH)

    def test_container_not_existing(self):
        analyzer = AnalyzeContainer(Container(self.NOT_EXISTING_PATH))
        job = analyzer.analyze()
        self.assertEqual(job, None)

    def test_analyze_empty_container(self):
        for f in self.EMPTY_CONTAINERS:
            analyzer = AnalyzeContainer(Container(f))
            job = analyzer.analyze()
            self.assertEqual(job, None)

    def test_analyze(self):
        for f in self.FILES:
            analyzer = AnalyzeContainer(Container(at.get_abs_path(f["path"])))
            job = analyzer.analyze()
            self.assertEqual(len(job), f["length"])

    def test_wizard_container_not_existing(self):
        analyzer = AnalyzeContainer(Container(self.NOT_EXISTING_PATH))
        job = analyzer.analyze_from_wizard("foo")
        self.assertEqual(job, None)

    def test_wizard_analyze_empty_container(self):
        for f in self.EMPTY_CONTAINERS:
            analyzer = AnalyzeContainer(Container(f))
            job = analyzer.analyze_from_wizard("foo")
            self.assertEqual(job, None)

    def test_wizard_analyze_none(self):
        for f in self.FILES:
            analyzer = AnalyzeContainer(Container(at.get_abs_path(f["path"])))
            job = analyzer.analyze_from_wizard(None)
            self.assertEqual(job, None)

    def test_wizard_analyze_valid(self):
        f = self.FILES[0]
        analyzer = AnalyzeContainer(Container(at.get_abs_path(f["path"])))
        job = analyzer.analyze_from_wizard(self.CONFIG_STRING)
        self.assertNotEqual(job, None)
        self.assertEqual(len(job), f["length"])

if __name__ == '__main__':
    unittest.main()



