#!/usr/bin/env python
# coding=utf-8

import os
import sys
import unittest

from . import get_abs_path

from aeneas.analyzecontainer import AnalyzeContainer
from aeneas.container import Container
from aeneas.logger import Logger

class TestAnalyzeContainer(unittest.TestCase):

    def test_check_container_txt_01(self):
        container_path = get_abs_path("res/validator/job_txt_config")
        logger = Logger()
        analyzer = AnalyzeContainer(Container(container_path), logger=logger)
        job = analyzer.analyze()
        self.assertEqual(len(job), 3)

    def test_check_container_txt_02(self):
        container_path = get_abs_path("res/validator/job_txt_config_not_root")
        logger = Logger()
        analyzer = AnalyzeContainer(Container(container_path), logger=logger)
        job = analyzer.analyze()
        self.assertEqual(len(job), 3)

    def test_check_container_txt_03(self):
        container_path = get_abs_path("res/validator/job_txt_config_not_root_nested")
        logger = Logger()
        analyzer = AnalyzeContainer(Container(container_path), logger=logger)
        job = analyzer.analyze()
        self.assertEqual(len(job), 3)

    def test_check_container_txt_paged_01(self):
        container_path = get_abs_path("res/validator/job_txt_config_paged_1")
        logger = Logger()
        analyzer = AnalyzeContainer(Container(container_path), logger=logger)
        job = analyzer.analyze()
        self.assertEqual(len(job), 3)

    def test_check_container_txt_paged_02(self):
        container_path = get_abs_path("res/validator/job_txt_config_paged_2")
        logger = Logger()
        analyzer = AnalyzeContainer(Container(container_path), logger=logger)
        job = analyzer.analyze()
        self.assertEqual(len(job), 3)

    def test_check_container_txt_paged_03(self):
        container_path = get_abs_path("res/validator/job_txt_config_paged_3")
        logger = Logger()
        analyzer = AnalyzeContainer(Container(container_path), logger=logger)
        job = analyzer.analyze()
        self.assertEqual(len(job), 3)

    def test_check_container_xml_01(self):
        container_path = get_abs_path("res/validator/job_xml_config")
        logger = Logger()
        analyzer = AnalyzeContainer(Container(container_path), logger=logger)
        job = analyzer.analyze()
        self.assertEqual(len(job), 3)

    def test_check_container_xml_02(self):
        container_path = get_abs_path("res/validator/job_xml_config_not_root")
        logger = Logger()
        analyzer = AnalyzeContainer(Container(container_path), logger=logger)
        job = analyzer.analyze()
        self.assertEqual(len(job), 3)

    def test_check_container_xml_03(self):
        container_path = get_abs_path("res/validator/job_xml_config_not_root_nested")
        logger = Logger()
        analyzer = AnalyzeContainer(Container(container_path), logger=logger)
        job = analyzer.analyze()
        self.assertEqual(len(job), 3)

if __name__ == '__main__':
    unittest.main()



