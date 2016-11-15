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

from aeneas.analyzecontainer import AnalyzeContainer
from aeneas.container import Container
import aeneas.globalfunctions as gf


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

    CONFIG_STRING = u"is_hierarchy_type=flat|is_hierarchy_prefix=assets/|is_text_file_relative_path=.|is_text_file_name_regex=.*\.xhtml|is_text_type=unparsed|is_audio_file_relative_path=.|is_audio_file_name_regex=.*\.mp3|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric|os_job_file_name=demo_sync_job_output|os_job_file_container=zip|os_job_file_hierarchy_type=flat|os_job_file_hierarchy_prefix=assets/|os_task_file_name=$PREFIX.xhtml.smil|os_task_file_format=smil|os_task_file_smil_page_ref=$PREFIX.xhtml|os_task_file_smil_audio_ref=../Audio/$PREFIX.mp3|job_language=en|job_description=Demo Sync Job"

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

    NOT_EXISTING_PATH = gf.absolute_path("res/validator/x/y/z/not_existing", __file__)

    def test_none(self):
        with self.assertRaises(TypeError):
            analyzer = AnalyzeContainer(None)

    def test_not_container(self):
        with self.assertRaises(TypeError):
            analyzer = AnalyzeContainer(self.NOT_EXISTING_PATH)

    def test_container_not_existing(self):
        analyzer = AnalyzeContainer(Container(self.NOT_EXISTING_PATH))
        job = analyzer.analyze()
        self.assertIsNone(job)

    def test_analyze_empty_container(self):
        for f in self.EMPTY_CONTAINERS:
            analyzer = AnalyzeContainer(Container(f))
            job = analyzer.analyze()
            self.assertIsNone(job)

    def test_analyze(self):
        for f in self.FILES:
            analyzer = AnalyzeContainer(Container(gf.absolute_path(f["path"], __file__)))
            job = analyzer.analyze()
            self.assertEqual(len(job), f["length"])

    def test_wizard_container_not_existing(self):
        analyzer = AnalyzeContainer(Container(self.NOT_EXISTING_PATH))
        job = analyzer.analyze(config_string=u"foo")
        self.assertIsNone(job)

    def test_wizard_analyze_empty_container(self):
        for f in self.EMPTY_CONTAINERS:
            analyzer = AnalyzeContainer(Container(f))
            job = analyzer.analyze(config_string=u"foo")
            self.assertIsNone(job)

    def test_wizard_analyze_valid(self):
        f = self.FILES[0]
        analyzer = AnalyzeContainer(Container(gf.absolute_path(f["path"], __file__)))
        job = analyzer.analyze(config_string=self.CONFIG_STRING)
        self.assertIsNotNone(job)
        self.assertEqual(len(job), f["length"])


if __name__ == "__main__":
    unittest.main()
