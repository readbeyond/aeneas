#!/usr/bin/env python
# coding=utf-8

import unittest

from . import get_abs_path

from aeneas.analyzecontainer import AnalyzeContainer
from aeneas.container import Container

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

    def test_analyze(self):
        for f in self.FILES:
            analyzer = AnalyzeContainer(Container(get_abs_path(f["path"])))
            job = analyzer.analyze()
            self.assertEqual(len(job), f["length"])

    #TODO analyze_from_wizard

if __name__ == '__main__':
    unittest.main()



