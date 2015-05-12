#!/usr/bin/env python
# coding=utf-8

import os
import sys
import unittest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
sys.path.append(PROJECT_DIR)

from aeneas.executejob import ExecuteJob
from aeneas.logger import Logger

class TestExecuteJob(unittest.TestCase):

    def test_execute(self):
        input_path = "../aeneas/tests/res/validator/job_txt_config_not_root_nested"
        output_path = "/tmp/"

        logger = Logger(tee=True)
        executor = ExecuteJob(job=None, logger=logger)
        executor.load_job_from_container(input_path)
        self.assertNotEqual(executor.job, None)
        result = executor.execute()
        self.assertTrue(result)
        result, path = executor.write_output_container(output_path)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(path))
        executor.clean()



if __name__ == '__main__':
    unittest.main()
