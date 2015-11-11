#!/usr/bin/env python
# coding=utf-8

"""
Read text fragments, from stdin or from file
"""

import sys

from aeneas.logger import Logger
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

NAME = "aeneas.tools.read_text"

TEXT_FILE_PARSED = gf.get_rel_path("res/parsed.txt")
TEXT_FILE_PLAIN = gf.get_rel_path("res/plain.txt")
TEXT_FILE_SUBTITLES = gf.get_rel_path("res/subtitles.txt")
TEXT_FILE_UNPARSED = gf.get_rel_path("res/unparsed.xhtml")

def usage():
    """ Print usage message """
    print ""
    print "Usage:"
    print "  $ python -m %s 'fragment 1|fragment 2|...|fragment N' list" % NAME
    print "  $ python -m %s /path/to/text_file [parsed|plain|subtitles|unparsed] [parameters]" % NAME
    print ""
    print "Parameters:"
    print "  -v                : verbose output"
    print "  class_regex=REGEX : extract text from elements with class attribute matching REGEX"
    print "  id_regex=REGEX    : extract text from elements with id attribute matching REGEX (unparsed)"
    print "  id_regex=REGEX    : use REGEX for text id attributes (subtitles, plain)"
    print "  sort=ALGORITHM    : sort the matched element id attributes using ALGORITHM (lexicographic, numeric, unsorted)"
    print ""
    print "Examples:"
    print "  $ python -m %s 'From|fairest|creatures|we|desire|increase' list" % NAME
    print "  $ python -m %s %s parsed" % (NAME, TEXT_FILE_PARSED)
    print "  $ python -m %s %s plain" % (NAME, TEXT_FILE_PLAIN)
    print "  $ python -m %s %s plain id_regex=Word%%06d" % (NAME, TEXT_FILE_PLAIN)
    print "  $ python -m %s %s subtitles" % (NAME, TEXT_FILE_SUBTITLES)
    print "  $ python -m %s %s subtitles id_regex=Sub%%03d" % (NAME, TEXT_FILE_SUBTITLES)
    print "  $ python -m %s %s unparsed id_regex=f[0-9]*" % (NAME, TEXT_FILE_UNPARSED)
    print "  $ python -m %s %s unparsed class_regex=ra   sort=unsorted" % (NAME, TEXT_FILE_UNPARSED)
    print "  $ python -m %s %s unparsed id_regex=f[0-9]* sort=numeric" % (NAME, TEXT_FILE_UNPARSED)
    print "  $ python -m %s %s unparsed id_regex=f[0-9]* sort=lexicographic" % (NAME, TEXT_FILE_UNPARSED)
    print ""
    sys.exit(2)


def main():
    """ Entry point """
    if len(sys.argv) < 3:
        usage()
    text_file_path = sys.argv[1]
    text_format = sys.argv[2]
    parameters = {}
    verbose = False
    for i in range(3, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-v":
            verbose = True
        else:
            args = arg.split("=")
            if len(args) == 2:
                key, value = args
                if key == "id_regex":
                    parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = value
                    parameters[gc.PPN_TASK_OS_FILE_ID_REGEX] = value
                if key == "class_regex":
                    parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX] = value
                if key == "sort":
                    parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT] = value

    logger = Logger(tee=verbose)
    text_file = get_text_file_object(text_file_path, text_format, parameters, logger)
    print str(text_file)
    sys.exit(0)

if __name__ == '__main__':
    main()



