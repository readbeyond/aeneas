#!/usr/bin/env python
# coding=utf-8

import os
import sys
import unittest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
sys.path.append(PROJECT_DIR)

from aeneas.executetask import ExecuteTask
from aeneas.logger import Logger
from aeneas.task import Task

class TestExecuteTask(unittest.TestCase):

    def test_execute(self):
        config_string = "task_language=en|os_task_file_format=txt|os_task_file_name=output_head.txt|is_text_type=plain|is_audio_file_head_length=11.960|is_audio_file_process_length=31.640"
        task = Task(config_string)
        task.audio_file_path_absolute = "../aeneas/tests/res/container/job/assets/p001.mp3"
        task.text_file_path_absolute = "../aeneas/tests/res/inputtext/sonnet_plain_head_length.txt"
        logger = Logger(tee=True)
        executor = ExecuteTask(task, logger=logger)
        result = executor.execute()
        self.assertTrue(result)
        task.sync_map_file_path_absolute = "/tmp/output_head_length.txt"
        path = task.output_sync_map_file()
        self.assertNotEqual(path, None)



if __name__ == '__main__':
    unittest.main()
