#!/usr/bin/env python
# coding=utf-8

import unittest

from aeneas.executetask import ExecuteTask
from aeneas.syncmap import SyncMapFormat
from aeneas.task import Task
import aeneas.globalfunctions as gf

class TestExecuteTask(unittest.TestCase):

    def execute(self, config_string, audio_path, text_path):
        handler, tmp_path = gf.tmp_file()
        task = Task(config_string)
        task.audio_file_path_absolute = gf.absolute_path(audio_path, __file__)
        task.text_file_path_absolute = gf.absolute_path(text_path, __file__)
        executor = ExecuteTask(task)
        executor.execute()
        task.sync_map_file_path_absolute = tmp_path
        result_path = task.output_sync_map_file()
        self.assertIsNotNone(result_path)
        self.assertEqual(result_path, tmp_path)
        self.assertGreater(len(gf.read_file_bytes(result_path)), 0)
        gf.delete_file(handler, tmp_path)

    def test_head_length(self):
        self.execute(
            u"task_language=en|os_task_file_format=txt|os_task_file_name=output|is_text_type=plain|is_audio_file_head_length=11.960|is_audio_file_process_length=31.640",
            "res/container/job/assets/p001.mp3",
            "res/inputtext/sonnet_plain_head_length.txt"
        )

    def test_head(self):
        self.execute(
            u"task_language=en|os_task_file_format=txt|os_task_file_name=output|is_text_type=plain|is_audio_file_head_length=11.960",
            "res/container/job/assets/p001.mp3",
            "res/inputtext/sonnet_plain_head.txt"
        )

    def test_tail_length(self):
        self.execute(
            u"task_language=en|os_task_file_format=txt|os_task_file_name=output|is_text_type=plain|is_audio_file_tail_length=1.000|is_audio_file_process_length=52.320",
            "res/container/job/assets/p001.mp3",
            "res/inputtext/sonnet_plain_head_length.txt"
        )

    def test_tail(self):
        self.execute(
            u"task_language=en|os_task_file_format=txt|os_task_file_name=output|is_text_type=plain|is_audio_file_tail_length=1.000",
            "res/container/job/assets/p001.mp3",
            "res/inputtext/sonnet_plain_head.txt"
        )

    def test_phrase_level(self):
        self.execute(
            u"task_language=en|os_task_file_format=smil|os_task_file_name=p001.smil|os_task_file_smil_audio_ref=p001.mp3|os_task_file_smil_page_ref=p001.xhtml|is_text_type=unparsed|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric",
            "res/container/job/assets/p001.mp3",
            "res/container/job/assets/p001.xhtml"
        )

    def test_formats(self):
        for fmt in SyncMapFormat.ALLOWED_VALUES:
            config_string = u"task_language=en|os_task_file_format=%s|os_task_file_name=output|is_text_type=plain|os_task_file_smil_audio_ref=p001.mp3|os_task_file_smil_page_ref=p001.xhtml" % fmt
            self.execute(
                config_string,
                "res/container/job/assets/p001.mp3",
                "res/inputtext/sonnet_plain.txt"
            )

    # TODO more tests

if __name__ == '__main__':
    unittest.main()



