#!/usr/bin/env python
# coding=utf-8

"""
Execute a task, that is, a pair of audio/text files
and a configuration string.
"""

import sys

import aeneas.globalfunctions as gf
from aeneas.executetask import ExecuteTask
from aeneas.logger import Logger
from aeneas.task import Task
from aeneas.tools import get_rel_path

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.1.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

def usage():
    """ Print usage message """
    name = "aeneas.tools.execute_task"
    dir_path_1 = get_rel_path("../tests/res/example_jobs/example1/OEBPS/Resources")
    config_string_1 = "task_language=en|os_task_file_format=srt|is_text_type=parsed"
    dir_path_2 = get_rel_path("../tests/res/container/job/assets/")
    config_string_2 = "task_language=en|os_task_file_format=smil|os_task_file_smil_audio_ref=p001.mp3|os_task_file_smil_page_ref=p001.xhtml|is_text_type=unparsed|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric"
    print ""
    print "Usage:"
    print "  $ python -m %s path/to/audio.mp3 path/to/text.txt config_string /path/to/output/file.smil [-v]" % name
    print ""
    print "Example 1 (input: parsed text, output: SRT)"
    print "  $ DIR=\"%s\"" % dir_path_1
    print "  $ CONFIG_STRING=\"%s\"" % config_string_1
    print "  $ python -m %s $DIR/sonnet001.mp3 $DIR/sonnet001.txt \"$CONFIG_STRING\" /tmp/sonnet001.srt" % name
    print ""
    print "Example 2 (input: unparsed text, output: SMIL)"
    print "  $ DIR=\"%s\"" % dir_path_2
    print "  $ CONFIG_STRING=\"%s\"" % config_string_2
    print "  $ python -m %s $DIR/p001.mp3 $DIR/p001.xhtml \"$CONFIG_STRING\" /tmp/p001.smil" % name
    print ""

def main():
    """ Entry point """
    if len(sys.argv) < 5:
        usage()
        return
    audio_file_path = sys.argv[1]
    text_file_path = sys.argv[2]
    config_string = sys.argv[3]
    sync_map_file_path = sys.argv[4]
    verbose = (sys.argv[-1] == "-v")

    if not gf.can_run_c_extension():
        print "[WARN] Unable to load Python C Extensions"
        print "[WARN] Running the slower pure Python code"
        print "[WARN] See the README file for directions to compile the Python C Extensions"

    print "[INFO] Creating task..."
    task = Task(config_string)
    task.audio_file_path_absolute = audio_file_path
    task.text_file_path_absolute = text_file_path
    task.sync_map_file_path_absolute = sync_map_file_path
    print "[INFO] Creating task... done"

    print "[INFO] Executing task..."
    logger = Logger(tee=verbose)
    executor = ExecuteTask(task=task, logger=logger)
    result = executor.execute()
    print "[INFO] Executing task... done"

    if not result:
        print "[ERRO] An error occurred while executing the task"
        return

    print "[INFO] Creating output sync map file..."
    path = task.output_sync_map_file()
    print "[INFO] Creating output sync map file... done"

    if path is not None:
        print "[INFO] Created %s" % path
    else:
        print "[ERRO] An error occurred while writing the output sync map file"

if __name__ == '__main__':
    main()



