#!/usr/bin/env python
# coding=utf-8

"""
Detect the time interval of the given audio file
containing the given text,
that is, detect the audio head and the audio length.
"""

import sys
import tempfile

from aeneas.audiofile import AudioFileMonoWAV
from aeneas.audiofile import AudioFileUnsupportedFormatError
from aeneas.ffmpegwrapper import FFMPEGWrapper
from aeneas.logger import Logger
from aeneas.sd import SD
from aeneas.tools import get_text_file_object
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

NAME = "aeneas.tools.run_sd"

AUDIO_FILE = gf.get_rel_path("res/audio.mp3")
PARAMETERS_BOTH = "min_head_length=0.0 max_head_length=5.0 min_tail_length=1.0 max_tail_length=5.0"
PARAMETERS_HEAD = "min_head_length=0.0 max_head_length=5.0"
PARAMETERS_TAIL = "min_tail_length=1.0 max_tail_length=5.0"
TEXT_FILE = gf.get_rel_path("res/parsed.txt")

def usage():
    """ Print usage message """
    print ""
    print "Usage:"
    print "  $ python -m %s language /path/to/text_file [parsed|plain|subtitles|unparsed] [parameters] /path/to/audio_file " % NAME
    print ""
    print "Parameters:"
    print "  max_head_length=VALUE : audio head of at most this many seconds"
    print "  max_tail_length=VALUE : audio tail of at most this many seconds"
    print "  min_head_length=VALUE : audio head of at least this many seconds"
    print "  min_tail_length=VALUE : audio tail of at least this many seconds"
    print ""
    print "Examples:"
    print "  $ python -m %s en %s parsed %s" % (NAME, TEXT_FILE, AUDIO_FILE)
    print "  $ python -m %s en %s parsed %s %s" % (NAME, TEXT_FILE, PARAMETERS_HEAD, AUDIO_FILE)
    print "  $ python -m %s en %s parsed %s %s" % (NAME, TEXT_FILE, PARAMETERS_TAIL, AUDIO_FILE)
    print "  $ python -m %s en %s parsed %s %s" % (NAME, TEXT_FILE, PARAMETERS_BOTH, AUDIO_FILE)
    print ""
    sys.exit(2)

def get_head_tail_length(parameters):
    """ Get head/tail min/max length from parameters """
    min_h = gc.SD_MIN_HEAD_LENGTH
    if "min_head_length" in parameters:
        min_h = parameters["min_head_length"]
    max_h = gc.SD_MAX_HEAD_LENGTH
    if "max_head_length" in parameters:
        max_h = parameters["max_head_length"]
    min_t = gc.SD_MIN_TAIL_LENGTH
    if "min_tail_length" in parameters:
        min_t = parameters["min_tail_length"]
    max_t = gc.SD_MAX_TAIL_LENGTH
    if "max_tail_length" in parameters:
        max_t = parameters["max_tail_length"]
    return (min_h, max_h, min_t, max_t)

def print_out(audio_len, start, end):
    zero = 0
    head_len = start
    text_len = end - start
    tail_len = audio_len - end
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

def main():
    """ Entry point """
    if len(sys.argv) < 5:
        usage()
    language = sys.argv[1]
    text_file_path = sys.argv[2]
    text_format = sys.argv[3]
    audio_file_path = sys.argv[-1]
    verbose = False
    parameters = {}
    for i in range(4, len(sys.argv)-1):
        arg = sys.argv[i]
        if arg == "-v":
            verbose = True
        else:
            args = arg.split("=")
            if len(args) == 2:
                key, value = args
                if key == "id_regex":
                    parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = value
                if key == "class_regex":
                    parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX] = value
                if key == "sort":
                    parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT] = value
                if key in [
                        "min_head_length",
                        "max_head_length",
                        "min_tail_length",
                        "max_tail_length"
                ]:
                    parameters[key] = float(value)
    (min_h, max_h, min_t, max_t) = get_head_tail_length(parameters)

    if not gf.can_run_c_extension():
        print "[WARN] Unable to load Python C Extensions"
        print "[WARN] Running the slower pure Python code"
        print "[WARN] See the README file for directions to compile the Python C Extensions"

    logger = Logger(tee=verbose)

    print "[INFO] Reading audio..."
    try:
        tmp_handler, tmp_file_path = tempfile.mkstemp(
            suffix=".wav",
            dir=gf.custom_tmp_dir()
        )
        converter = FFMPEGWrapper(logger=logger)
        converter.convert(audio_file_path, tmp_file_path)
    except IOError:
        print "[ERRO] Cannot convert audio file '%s'" % audio_file_path
        print "[ERRO] Check that it exists and that its path is written/escaped correctly."
        sys.exit(1)
    try:
        audio_file = AudioFileMonoWAV(tmp_file_path, logger=logger)
    except (AudioFileUnsupportedFormatError, IOError):
        print "[ERRO] Cannot read the converted WAV file '%s'" % tmp_file_path
        sys.exit(1)
    print "[INFO] Reading audio... done"

    print "[INFO] Reading text..."
    text_file = get_text_file_object(text_file_path, text_format, parameters, logger)
    text_file.set_language(language)
    print "[INFO] Reading text... done"

    print "[INFO] Detecting audio interval..."
    start_detector = SD(audio_file, text_file, logger=logger)
    start, end = start_detector.detect_interval(min_h, max_h, min_t, max_t)
    print "[INFO] Detecting audio interval... done"

    print_out(audio_file.audio_length, start, end)
    gf.delete_file(tmp_handler, tmp_file_path)
    sys.exit(0)

if __name__ == '__main__':
    main()



