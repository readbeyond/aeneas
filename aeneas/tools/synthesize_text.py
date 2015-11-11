#!/usr/bin/env python
# coding=utf-8

"""
Synthesize several text fragments,
read from file or from stdin,
producing a wav audio file.
"""

import sys

from aeneas.logger import Logger
from aeneas.synthesizer import Synthesizer
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

NAME = "aeneas.tools.synthesize_text"

AUDIO_FILE = "output/synthesized.wav"

TEXT_FILE_PARSED = gf.get_rel_path("res/parsed.txt")
TEXT_FILE_PLAIN = gf.get_rel_path("res/plain.txt")
TEXT_FILE_SUBTITLES = gf.get_rel_path("res/subtitles.txt")
TEXT_FILE_UNPARSED = gf.get_rel_path("res/unparsed.xhtml")

def usage():
    """ Print usage message """
    print ""
    print "Usage:"
    print "  $ python -m %s language 'fragment 1|fragment 2|...|fragment N' list [parameters] /path/to/audio_file " % NAME
    print "  $ python -m %s language /path/to/text_file [parsed|plain|subtitles|unparsed] [parameters] /path/to/audio_file " % NAME
    print ""
    print "Parameters:"
    print "  -v                : verbose output"
    print "  backwards         : synthesize from the last fragment to the first one"
    print "  class_regex=REGEX : extract text from elements with class attribute matching REGEX"
    print "  end=VALUE         : slice the text file until fragment VALUE"
    print "  id_regex=REGEX    : extract text from elements with id attribute matching REGEX"
    print "  quit_after=VALUE  : synthesize fragments until VALUE seconds or the end of text is reached"
    print "  sort=ALGORITHM    : sort the matched element id attributes using ALGORITHM (lexicographic, numeric, unsorted)"
    print "  start=VALUE       : slice the text file from fragment VALUE"
    print ""
    print "Examples:"
    print "  $ python -m %s en 'From fairest creatures|we desire|increase' list %s" % (NAME, AUDIO_FILE)
    print "  $ python -m %s en %s      plain            %s" % (NAME, TEXT_FILE_PLAIN, AUDIO_FILE)
    print "  $ python -m %s en %s     parsed           %s" % (NAME, TEXT_FILE_PARSED, AUDIO_FILE)
    print "  $ python -m %s en %s  subtitles        %s" % (NAME, TEXT_FILE_SUBTITLES, AUDIO_FILE)
    print "  $ python -m %s en %s unparsed  id_regex=f[0-9]*                    %s" % (NAME, TEXT_FILE_UNPARSED, AUDIO_FILE)
    print "  $ python -m %s en %s unparsed  class_regex=ra                      %s" % (NAME, TEXT_FILE_UNPARSED, AUDIO_FILE)
    print "  $ python -m %s en %s unparsed  id_regex=f[0-9]* sort=unsorted      %s" % (NAME, TEXT_FILE_UNPARSED, AUDIO_FILE)
    print "  $ python -m %s en %s unparsed  id_regex=f[0-9]* sort=numeric       %s" % (NAME, TEXT_FILE_UNPARSED, AUDIO_FILE)
    print "  $ python -m %s en %s unparsed  id_regex=f[0-9]* sort=lexicographic %s" % (NAME, TEXT_FILE_UNPARSED, AUDIO_FILE)
    print "  $ python -m %s en %s      plain     start=5                             %s" % (NAME, TEXT_FILE_PLAIN, AUDIO_FILE)
    print "  $ python -m %s en %s      plain     end=10                              %s" % (NAME, TEXT_FILE_PLAIN, AUDIO_FILE)
    print "  $ python -m %s en %s      plain     start=5 end=10                      %s" % (NAME, TEXT_FILE_PLAIN, AUDIO_FILE)
    print "  $ python -m %s en %s      plain     backwards                           %s" % (NAME, TEXT_FILE_PLAIN, AUDIO_FILE)
    print "  $ python -m %s en %s      plain     quit_after=10.0                     %s" % (NAME, TEXT_FILE_PLAIN, AUDIO_FILE)
    print "  $ python -m %s en %s      plain     backwards quit_after=10.0           %s" % (NAME, TEXT_FILE_PLAIN, AUDIO_FILE)
    print ""
    sys.exit(2)

def main():
    """ Entry point """
    if len(sys.argv) < 5:
        usage()
    language = sys.argv[1]
    text_file_path = sys.argv[2]
    text_format = sys.argv[3]
    audio_file_path = sys.argv[-1]
    verbose = False
    backwards = False
    quit_after = None
    parameters = {}

    for i in range(4, len(sys.argv)-1):
        arg = sys.argv[i]
        if arg == "backwards":
            backwards = True
        elif arg == "-v":
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
                if (key == "start") or (key == "end"):
                    try:
                        parameters[key] = int(value)
                    except:
                        pass
                if key == "quit_after":
                    quit_after = float(value)

    logger = Logger(tee=verbose)

    text_file = get_text_file_object(text_file_path, text_format, parameters, logger)
    text_file.set_language(language)
    start_fragment = None
    if "start" in parameters:
        start_fragment = parameters["start"]
    end_fragment = None
    if "end" in parameters:
        end_fragment = parameters["end"]
    print "[INFO] Read input text file with %d fragments" % (len(text_file))
    if start_fragment is not None:
        print "[INFO] Slicing from index %d" % (start_fragment)
    if end_fragment is not None:
        print "[INFO] Slicing to index %d" % (end_fragment)
    text_slice = text_file.get_slice(start_fragment, end_fragment)
    print "[INFO] Synthesizing %d fragments" % (len(text_slice))
    force_pure_python = False
    if quit_after is not None:
        print "[INFO] Stop synthesizing after reaching %.3f seconds" % (quit_after)
    if backwards:
        print "[INFO] Synthesizing backwards (=> forcing pure Python code)"
        force_pure_python = True

    try:
        synt = Synthesizer(logger=logger)
        synt.synthesize(text_slice, audio_file_path, quit_after, backwards, force_pure_python)
        print "[INFO] Created file '%s'" % audio_file_path
    except IOError:
        print "[ERRO] Cannot synthesize text to file '%s'" % (audio_file_path)
        print "[ERRO] Make sure the output file path is written/escaped correctly and that you have write permission on it"
        sys.exit(1)
    sys.exit(0)

if __name__ == '__main__':
    main()



