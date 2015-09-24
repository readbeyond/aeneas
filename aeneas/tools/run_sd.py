#!/usr/bin/env python
# coding=utf-8

"""
Detect the time interval of the given audio file
containing the given text,
that is, detect the audio head and the audio length.
"""

import os
import sys
import tempfile

import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf
from aeneas.audiofile import AudioFile
from aeneas.ffmpegwrapper import FFMPEGWrapper
from aeneas.logger import Logger
from aeneas.synthesizer import Synthesizer
from aeneas.textfile import TextFile
from aeneas.tools import get_rel_path
from aeneas.vad import VAD
from aeneas.sd import SD

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.2.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

def usage():
    """ Print usage message """
    name = "aeneas.tools.run_sd"
    dir_path = get_rel_path("../tests/res/example_jobs/example1/OEBPS/Resources")
    print ""
    print "Usage:"
    print "  $ python -m %s language 'fragment 1|fragment 2|...|fragment N' list [parameters] /path/to/audio_file" % name
    print "  $ python -m %s language /path/to/text_file [parsed|plain|subtitles|unparsed] [parameters] /path/to/audio_file " % name
    print ""
    print "Examples:"
    print "  $ DIR=\"%s\"" % dir_path
    print "  $ python -m %s en 'From fairest creatures we desire increase' list $DIR/sonnet001.mp3" % (name)
    print "  $ python -m %s en $DIR/sonnet001.txt parsed $DIR/sonnet001.mp3" % (name)
    print "  $ python -m %s en $DIR/sonnet001.txt parsed min_head_length=0.0 max_head_length=5.0 $DIR/sonnet001.mp3" % (name)
    print "  $ python -m %s en $DIR/sonnet001.txt parsed min_tail_length=1.0 max_tail_length=5.0 $DIR/sonnet001.mp3" % (name)
    print "  $ python -m %s en $DIR/sonnet001.txt parsed min_head_length=0.0 max_head_length=5.0 min_tail_length=1.0 max_tail_length=5.0 $DIR/sonnet001.mp3" % (name)
    print ""

def cleanup(handler, path):
    """ Remove temporary hadler/file """
    if handler is not None:
        try:
            os.close(handler)
        except:
            pass
    if path is not None:
        try:
            os.remove(path)
        except:
            pass

def main():
    """ Entry point """
    if len(sys.argv) < 5:
        usage()
        return
    language = sys.argv[1]
    text_file_path = sys.argv[2]
    text_format = sys.argv[3]
    audio_file_path = sys.argv[-1]
    verbose = False
    parameters = {}

    for i in range(4, len(sys.argv)-1):
        args = sys.argv[i].split("=")
        if len(args) == 1:
            verbose = (args[0] in ["v", "-v", "verbose", "--verbose"])
        if len(args) == 2:
            key, value = args
            if key == "id_regex":
                parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = value
            if key == "class_regex":
                parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX] = value
            if key == "sort":
                parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT] = value
            if key == "min_head_length":
                parameters["min_head_length"] = float(value)
            if key == "max_head_length":
                parameters["max_head_length"] = float(value)
            if key == "min_tail_length":
                parameters["min_head_length"] = float(value)
            if key == "max_tail_length":
                parameters["max_tail_length"] = float(value)

    if not gf.can_run_c_extension():
        print "[WARN] Unable to load Python C Extensions"
        print "[WARN] Running the slower pure Python code"
        print "[WARN] See the README file for directions to compile the Python C Extensions"

    logger = Logger(tee=verbose)

    print "[INFO] Reading audio..."
    tmp_handler, tmp_file_path = tempfile.mkstemp(
        suffix=".wav",
        dir=gf.custom_tmp_dir()
    )
    converter = FFMPEGWrapper(logger=logger)
    converter.convert(audio_file_path, tmp_file_path)
    audio_file = AudioFile(tmp_file_path)
    print "[INFO] Reading audio... done"

    print "[INFO] Reading text..."
    if text_format == "list":
        text_file = TextFile()
        text_file.read_from_list(text_file_path.split("|"))
    else:
        text_file = TextFile(text_file_path, text_format, parameters)
    text_file.set_language(language)
    print "[INFO] Reading text... done"

    print "[INFO] Detecting audio interval..."
    sd = SD(audio_file, text_file, logger=logger)
    min_head_length = gc.SD_MIN_HEAD_LENGTH
    if "min_head_length" in parameters:
        min_head_length = parameters["min_head_length"]
    max_head_length = gc.SD_MAX_HEAD_LENGTH
    if "max_head_length" in parameters:
        max_head_length = parameters["max_head_length"]
    min_tail_length = gc.SD_MIN_TAIL_LENGTH
    if "min_tail_length" in parameters:
        min_tail_length = parameters["min_tail_length"]
    max_tail_length = gc.SD_MAX_TAIL_LENGTH
    if "max_tail_length" in parameters:
        max_tail_length = parameters["max_tail_length"]
    start, end = sd.detect_interval(
        min_head_length,
        max_head_length,
        min_tail_length,
        max_tail_length
    )
    zero = 0
    audio_len = audio_file.audio_length
    head_len = start
    text_len = end - start
    tail_len = audio_len - end
    print "[INFO] Detecting audio interval... done"
    print "[INFO] "
    print "[INFO] Head: %.3f %.3f (%.3f)" % (zero, start, head_len)
    print "[INFO] Text: %.3f %.3f (%.3f)" % (start, end, text_len)
    print "[INFO] Tail: %.3f %.3f (%.3f)" % (end, audio_len, tail_len)
    print "[INFO] "
    zero_h = gf.time_to_hhmmssmmm(0)
    start_h = gf.time_to_hhmmssmmm(start)
    end_h = gf.time_to_hhmmssmmm(end)
    audio_len_h = gf.time_to_hhmmssmmm(audio_len)
    head_len_h = gf.time_to_hhmmssmmm(head_len)
    text_len_h = gf.time_to_hhmmssmmm(text_len)
    tail_len_h = gf.time_to_hhmmssmmm(tail_len)
    print "[INFO] Head: %s %s (%s)" % (zero_h, start_h, head_len_h)
    print "[INFO] Text: %s %s (%s)" % (start_h, end_h, text_len_h)
    print "[INFO] Tail: %s %s (%s)" % (end_h, audio_len_h, tail_len_h)

    #print "[INFO]   Cleaning up..."
    cleanup(tmp_handler, tmp_file_path)
    #print "[INFO]   Cleaning up... done"

if __name__ == '__main__':
    main()



