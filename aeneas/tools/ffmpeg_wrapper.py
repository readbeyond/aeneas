#!/usr/bin/env python
# coding=utf-8

"""
Use the wrapper around ``ffmpeg``
"""

import sys

from aeneas.ffmpegwrapper import FFMPEGWrapper
from aeneas.logger import Logger
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

NAME = "aeneas.tools.ffmpeg_wrapper"

INPUT_FILE = gf.get_rel_path("res/audio.mp3")
OUTPUT_FILE = "output/audio.wav"

def usage():
    """ Print usage message """
    print ""
    print "Usage:"
    print "  $ python -m %s /path/to/input_file /path/to/output_file [-v]" % NAME
    print ""
    print "Options:"
    print "  -v : verbose output"
    print ""
    print "Example:"
    print "  $ python -m %s %s %s" % (NAME, INPUT_FILE, OUTPUT_FILE)
    print ""
    sys.exit(2)

def main():
    """ Entry point """
    if len(sys.argv) < 3:
        usage()
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    verbose = False
    for i in range(3, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-v":
            verbose = True

    logger = Logger(tee=verbose)
    try:
        converter = FFMPEGWrapper(logger=logger)
        converter.convert(input_file_path, output_file_path)
        print "[INFO] Converted '%s' into '%s'" % (input_file_path, output_file_path)
    except IOError:
        print "[ERRO] Cannot convert file '%s' into '%s'" % (input_file_path, output_file_path)
        print "[ERRO] Make sure the input file has a format supported by ffmpeg and that its path is written/escaped correctly"
        print "[ERRO] Make sure the output file path is written/escaped correctly and that you have write permission on it"
        sys.exit(1)
    sys.exit(0)

if __name__ == '__main__':
    main()



