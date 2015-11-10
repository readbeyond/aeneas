#!/usr/bin/env python
# coding=utf-8

"""
Execute a task, that is, a pair of audio/text files
and a configuration string.
"""

import sys

from aeneas.adjustboundaryalgorithm import AdjustBoundaryAlgorithm
from aeneas.downloader import Downloader
from aeneas.executetask import ExecuteTask
from aeneas.idsortingalgorithm import IDSortingAlgorithm
from aeneas.language import Language
from aeneas.logger import Logger
from aeneas.syncmap import SyncMapFormat
from aeneas.syncmap import SyncMapHeadTailFormat
from aeneas.task import Task
from aeneas.textfile import TextFileFormat
from aeneas.validator import Validator
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.3.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

NAME = "aeneas.tools.execute_task"

AUDIO_FILE = gf.get_rel_path("res/audio.mp3")

CONFIG_STRING_JSON = "task_language=en|is_text_type=plain|os_task_file_format=json"
SYNC_MAP_JSON = "output/sonnet.json"
TEXT_FILE_PLAIN = gf.get_rel_path("res/plain.txt")

CONFIG_STRING_SRT = "task_language=en|is_text_type=subtitles|os_task_file_format=srt"
SYNC_MAP_SRT = "output/sonnet.srt"
TEXT_FILE_SUBTITLES = gf.get_rel_path("res/subtitles.txt")

CONFIG_STRING_SMIL = "task_language=en|is_text_type=unparsed|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric|os_task_file_format=smil|os_task_file_smil_audio_ref=p001.mp3|os_task_file_smil_page_ref=p001.xhtml"
SYNC_MAP_SMIL = "output/sonnet.smil"
TEXT_FILE_UNPARSED = gf.get_rel_path("res/page.xhtml")

CONFIG_STRING_YOUTUBE = "task_language=en|is_text_type=plain|os_task_file_format=json"
SYNC_MAP_YOUTUBE = "output/sonnet.json"
TEXT_FILE_YOUTUBE = gf.get_rel_path("res/plain.txt")
URL_YOUTUBE = "https://www.youtube.com/watch?v=rU4a7AA8wM0"

EXAMPLES = {
    "--example-json" : {
        "audio": AUDIO_FILE,
        "text": TEXT_FILE_PLAIN,
        "config": CONFIG_STRING_JSON,
        "syncmap": SYNC_MAP_JSON
    },
    "--example-srt" : {
        "audio": AUDIO_FILE,
        "text": TEXT_FILE_SUBTITLES,
        "config": CONFIG_STRING_SRT,
        "syncmap": SYNC_MAP_SRT
    },
    "--example-smil" : {
        "audio": AUDIO_FILE,
        "text": TEXT_FILE_UNPARSED,
        "config": CONFIG_STRING_SMIL,
        "syncmap": SYNC_MAP_SMIL
    }
}

LIST_VALUES = {
    "task_language" : Language,
    "is_text_type" : TextFileFormat,
    "is_text_unparsed_id_sort" : IDSortingAlgorithm,
    "os_task_file_format" : SyncMapFormat,
    "os_task_file_head_tail_format" : SyncMapHeadTailFormat,
    "task_adjust_boundary_algorithm" : AdjustBoundaryAlgorithm,
}

def get_values(class_name):
    return ", ".join(class_name.ALLOWED_VALUES)

def get_parameters():
    print "  task_language                           : language (*)"
    print ""
    print "  is_audio_file_detect_head_max           : detect audio head, at most this many seconds"
    print "  is_audio_file_detect_head_min           : detect audio head, at least this many seconds"
    print "  is_audio_file_detect_tail_max           : detect audio tail, at most this many seconds"
    print "  is_audio_file_detect_tail_min           : detect audio tail, at least this many seconds"
    print "  is_audio_file_head_length               : ignore this many seconds from the begin of the audio file"
    print "  is_audio_file_process_length            : process this many seconds of the audio file"
    print "  is_audio_file_tail_length               : ignore this many seconds from the end of the audio file"
    print ""
    print "  is_text_type                            : input text format (*)"
    print "  is_text_unparsed_class_regex            : regex matching class attributes (unparsed)"
    print "  is_text_unparsed_id_regex               : regex matching id attributes (unparsed)"
    print "  is_text_unparsed_id_sort                : sort matched elements by id (unparsed) (*)"
    print "  is_text_file_ignore_regex               : ignore text matched by regex for audio alignment purposes"
    print "  is_text_file_transliterate_map          : apply the given transliteration map for audio alignment purposes"
    print ""
    print "  os_task_file_format                     : output sync map format (*)"
    print "  os_task_file_id_regex                   : id regex for the output sync map (subtitles, plain)"
    print "  os_task_file_head_tail_format           : format audio head/tail (*)"
    print "  os_task_file_smil_audio_ref             : value for the audio ref (smil, smilh, smilm)"
    print "  os_task_file_smil_page_ref              : value for the text ref (smil, smilh, smilm)"
    print ""
    print "  task_adjust_boundary_algorithm          : adjust sync map fragments using algorithm (*)"
    print "  task_adjust_boundary_aftercurrent_value : offset value, in seconds (aftercurrent)"
    print "  task_adjust_boundary_beforenext_value   : offset value, in seconds (beforenext)"
    print "  task_adjust_boundary_offset_value       : offset value, in seconds (offset)"
    print "  task_adjust_boundary_percent_value      : percent value, in [0..100], (percent)"
    print "  task_adjust_boundary_rate_value         : max rate, in characters/s (rate, rateaggressive)"


def usage(examples=False, full_help=False):
    """ Print usage message """
    print ""
    print "Usage:"
    print "  $ python -m %s -e" % NAME
    print "  $ python -m %s -h" % NAME
    print "  $ python -m %s --list-parameters" % NAME
    print "  $ python -m %s --list-values PARAM" % NAME
    print "  $ python -m %s path/to/audio.mp3 path/to/text.txt CONFIG_STRING /path/to/output.file [options]" % NAME
    print "  $ python -m %s URL path/to/text.txt CONFIG_STRING /path/to/output.file -y [options]" % NAME
    print ""
    print "Options:"
    print "  -e, --examples      : print examples"
    print "  -h, --help          : print full help"
    print "  -v                  : verbose output"
    print "  -y                  : download audio from YouTube video located at URL"
    print "  --list-parameters   : list all parameters"
    print "  --list-values PARAM : list all allowed values for the parameter PARAM"
    if full_help:
        print "  --best-audio        : download best audio (-y)"
        print "  --example-json      : run example with JSON output"
        print "  --example-smil      : run example with SMIL output"
        print "  --example-srt       : run example with SRT output"
        print "  --keep-audio        : do not delete the audio file downloaded from YouTube (-y)"
        print "  --output-html       : output HTML file for fine tuning"
        print "  --skip-validator    : do not validate CONFIG_STRING"
        print ""
        print "Documentation:"
        print "  Please visit http://www.readbeyond.it/aeneas/docs/"
        print ""
        print "Parameters:"
        get_parameters()
    if full_help or examples:
        print ""
        print "Example 1 (input: plain text, output: JSON)"
        print "  $ CONFIG_STRING=\"%s\"" % CONFIG_STRING_JSON
        print "  $ python -m %s %s %s \"$CONFIG_STRING\" %s" % (NAME, AUDIO_FILE, TEXT_FILE_PLAIN, SYNC_MAP_JSON)
        print "  or"
        print "  $ python -m %s --example-json" % NAME
        print ""
        print "Example 2 (input: subtitles text, output: SRT)"
        print "  $ CONFIG_STRING=\"%s\"" % CONFIG_STRING_SRT
        print "  $ python -m %s %s %s \"$CONFIG_STRING\" %s" % (NAME, AUDIO_FILE, TEXT_FILE_SUBTITLES, SYNC_MAP_SRT)
        print "  or"
        print "  $ python -m %s --example-srt" % NAME
        print ""
        print "Example 3 (input: unparsed text, output: SMIL)"
        print "  $ CONFIG_STRING=\"%s\"" % CONFIG_STRING_SMIL
        print "  $ python -m %s %s %s \"$CONFIG_STRING\" %s" % (NAME, AUDIO_FILE, TEXT_FILE_UNPARSED, SYNC_MAP_SMIL)
        print "  or"
        print "  $ python -m %s --example-smil" % NAME
        print ""
        print "Example 4 (download audio from YouTube)"
        print "  $ CONFIG_STRING=\"%s\"" % CONFIG_STRING_YOUTUBE
        print "  $ python -m %s %s %s \"$CONFIG_STRING\" %s -y" % (NAME, URL_YOUTUBE, TEXT_FILE_YOUTUBE, SYNC_MAP_YOUTUBE)
    print ""
    sys.exit(2)

def run(argv):
    """ Execute task. argv are the arguments """

    if ("-e" in argv) or ("--examples" in argv):
        # show examples
        usage(examples=True)

    if ("-h" in argv) or ("--help" in argv):
        # show full help
        usage(full_help=True)

    if ("--list-parameters" in argv):
        # list parameters
        print "[INFO] You can use --list-values on parameters marked by (*)" 
        print "[INFO] Available parameters:" 
        print ""
        get_parameters()
        print ""
        sys.exit(2)

    if ("--list-values" in argv):
        # list values
        if len(argv) > 2:
            parameter = argv[2]
            if parameter in LIST_VALUES:
                print "[INFO] Allowed values for parameter '%s'" % parameter
                print "[INFO] %s" % get_values(LIST_VALUES[parameter])
                sys.exit(2)
            else:
                usage()
        else:
            usage()

    # shall we run a demo?
    demo = None
    for example in EXAMPLES:
        if example in argv:
            demo = example

    # if no demo, check we have enough arguments
    if (demo is None) and (len(argv) < 5):
        usage()

    verbose = False
    download_from_youtube = False
    best_audio = False
    keep_audio = False
    output_html = False
    if demo is None:
        # no demo, read arguments
        audio_file_path = argv[1]
        text_file_path = argv[2]
        config_string = argv[3]
        sync_map_file_path = argv[4]
        validate = True
        for i in range(5, len(argv)):
            arg = argv[i]
            if arg == "-v":
                verbose = True
            if arg == "--skip-validator":
                validate = False
            if arg == "-y":
                download_from_youtube = True
                youtube_url = audio_file_path
            if arg == "--keep-audio":
                keep_audio = True
            if arg == "--best-audio":
                best_audio = True
            if arg == "--output-html":
                output_html = True
    else:
        # demo, set arguments
        validate = False
        demo_arguments = EXAMPLES[demo]
        audio_file_path = demo_arguments["audio"]
        text_file_path = demo_arguments["text"]
        config_string = demo_arguments["config"]
        sync_map_file_path = demo_arguments["syncmap"]

    if not gf.can_run_c_extension():
        print "[WARN] Unable to load Python C Extensions"
        print "[WARN] Running the slower pure Python code"
        print "[WARN] See the README file for directions to compile the Python C Extensions"

    if output_html:
        keep_audio = True
        html_file_path = sync_map_file_path + ".html"

    logger = Logger(tee=verbose)

    if validate:
        print "[INFO] Validating config string (specify --skip-validator to bypass)..."
        validator = Validator(logger=logger)
        result = validator.check_task_configuration(config_string, external_name=True)
        if not result.passed:
            print "[ERRO] The given config string is not valid:"
            print result.pretty_print()
            sys.exit(1)
        print "[INFO] Validating config string... done"

    if download_from_youtube:
        try:
            print "[INFO] Downloading audio from '%s' ..." % youtube_url
            downloader = Downloader(logger=logger)
            audio_file_path = downloader.audio_from_youtube(
                youtube_url,
                best_audio=best_audio
            )
            print "[INFO] Downloading audio from '%s' ... done" % youtube_url
        except ImportError:
            print "[ERRO] You need to install Pythom module pafy to download audio from YouTube. Run:"
            print "[ERRO] $ sudo pip install pafy"
            sys.exit(1)
        except Exception as exc:
            print "[ERRO] The following error occurred while downloading audio from YouTube:"
            print "[ERRO] %s" % str(exc)
            sys.exit(1)

    if demo is not None:
        print "[INFO] Running example task with arguments:"
        print "[INFO]   Audio file:    %s" % audio_file_path
        print "[INFO]   Text file:     %s" % text_file_path
        print "[INFO]   Config string: %s" % config_string
        print "[INFO]   Sync map file: %s" % sync_map_file_path

    try:
        print "[INFO] Creating task..."
        task = Task(config_string, logger=logger)
        task.audio_file_path_absolute = audio_file_path
        task.text_file_path_absolute = text_file_path
        task.sync_map_file_path_absolute = sync_map_file_path
        print "[INFO] Creating task... done"
    except Exception as exc:
        print "[ERRO] The following error occurred while creating the task:"
        print "[ERRO] %s" % str(exc)
        sys.exit(1)

    try:
        print "[INFO] Executing task..."
        executor = ExecuteTask(task=task, logger=logger)
        executor.execute()
        print "[INFO] Executing task... done"
    except Exception as exc:
        print "[ERRO] The following error occurred while executing the task:"
        print "[ERRO] %s" % str(exc)
        sys.exit(1)

    try:
        print "[INFO] Creating output sync map file..."
        path = task.output_sync_map_file()
        print "[INFO] Creating output sync map file... done"
        print "[INFO] Created %s" % path
    except Exception as exc:
        print "[ERRO] The following error occurred while writing the sync map file:"
        print "[ERRO] %s" % str(exc)
        sys.exit(1)

    if output_html:
        try:
            parameters = {}
            parameters[gc.PPN_TASK_OS_FILE_FORMAT] = task.configuration.os_file_format
            parameters[gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF] = task.configuration.os_file_smil_page_ref
            parameters[gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF] = task.configuration.os_file_smil_audio_ref
            print "[INFO] Creating output HTML file..."
            task.sync_map.output_html_for_tuning(audio_file_path, html_file_path, parameters)
            print "[INFO] Creating output HTML file... done"
            print "[INFO] Created %s" % html_file_path
        except Exception as exc:
            print "[ERRO] The following error occurred while writing the HTML file:"
            print "[ERRO] %s" % str(exc)
            sys.exit(1)

    if download_from_youtube:
        if keep_audio:
            print "[INFO] Option --keep-audio set: keeping downloaded file '%s'" % audio_file_path
        else:
            gf.delete_file(None, audio_file_path)

    sys.exit(0)

def main():
    """ Entry point """
    run(sys.argv)

if __name__ == '__main__':
    main()



