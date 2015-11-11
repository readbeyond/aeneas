#!/usr/bin/env python
# coding=utf-8

"""
Use the wrapper around ``ffprobe``
"""

import sys

from aeneas.ffprobewrapper import FFPROBEParsingError
from aeneas.ffprobewrapper import FFPROBEUnsupportedFormatError
from aeneas.ffprobewrapper import FFPROBEWrapper
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

NAME = "aeneas.tools.ffprobe_wrapper"

INPUT_FILE = gf.get_rel_path("res/audio.mp3")

def usage():
    """ Print usage message """
    print ""
    print "Usage:"
    print "  $ python -m %s /path/to/audio_file [-v]" % NAME
    print ""
    print "Options:"
    print "  -v : verbose output"
    print ""
    print "Example:"
    print "  $ python -m %s %s" % (NAME, INPUT_FILE)
    print ""
    sys.exit(2)

def main():
    """ Entry point """
    if len(sys.argv) < 2:
        usage()
    audio_file_path = sys.argv[1]
    verbose = False
    for i in range(2, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-v":
            verbose = True

    logger = Logger(tee=verbose)
    try:
        prober = FFPROBEWrapper(logger=logger)
        dictionary = prober.read_properties(audio_file_path)
        for key in sorted(dictionary.keys()):
            print "%s %s" % (key, dictionary[key])
    except IOError:
        print "[ERRO] Cannot read file '%s'" % (audio_file_path)
        print "[ERRO] Make sure the input file path is written/escaped correctly"
        sys.exit(1)
    except (FFPROBEUnsupportedFormatError, FFPROBEParsingError):
        print "[ERRO] Cannot read properties of file '%s'" % (audio_file_path)
        print "[ERRO] Make sure the input file has a format supported by ffprobe"
        sys.exit(1)
    sys.exit(0)

if __name__ == '__main__':
    main()



