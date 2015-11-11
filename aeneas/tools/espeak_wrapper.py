#!/usr/bin/env python
# coding=utf-8

"""
Use the wrapper around ``espeak``
"""

import sys

from aeneas.espeakwrapper import ESPEAKWrapper
from aeneas.logger import Logger
from aeneas.textfile import TextFile
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

NAME = "aeneas.tools.espeak_wrapper"

OUTPUT_FILE = "output/sonnet.wav"
TEXT = "From fairest creatures we desire increase"
TEXT_MULTI = "From|fairest|creatures|we|desire|increase"

def usage():
    """ Print usage message """
    print ""
    print "Usage:"
    print "  $ python -m %s TEXT LANGUAGE /path/to/output_file [-f] [-m] [-v]" % NAME
    print ""
    print "Options:"
    print "  -f : force using pure Python code, even if C extension is installed"
    print "  -m : text contains multiple fragments, separated by a '|' character"
    print "  -v : verbose output"
    print ""
    print "Example:"
    print "  $ python -m %s \"%s\" en %s" % (NAME, TEXT, OUTPUT_FILE)
    print "  $ python -m %s \"%s\" en %s -m" % (NAME, TEXT_MULTI, OUTPUT_FILE)
    print ""
    sys.exit(2)

def main():
    """ Entry point """
    if len(sys.argv) < 4:
        usage()
    text = gf.safe_unicode(sys.argv[1])
    if text is None:
        print "[ERRO] Unable to decode the given text to Unicode"
        sys.exit(1)
    language = sys.argv[2]
    output_file_path = sys.argv[3]
    verbose = False
    force = False
    multiple = False
    for i in range(4, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-v":
            verbose = True
        if arg == "-f":
            force = True
        if arg == "-m":
            multiple = True

    logger = Logger(tee=verbose)
    try:
        synt = ESPEAKWrapper(logger=logger)
        if multiple:
            tfl = TextFile()
            tfl.read_from_list(text.split("|"))
            tfl.set_language(language)
            synt.synthesize_multiple(tfl, output_file_path, force_pure_python=force)
        else:
            synt.synthesize_single(text, language, output_file_path, force_pure_python=force)
        print "[INFO] Created file '%s'" % output_file_path
    except IOError:
        print "[ERRO] Unable to create file '%s'" % output_file_path
        print "[ERRO] Make sure the output file path is written/escaped correctly and that you have write permission on it"
        sys.exit(1)
    sys.exit(0)

if __name__ == '__main__':
    main()



