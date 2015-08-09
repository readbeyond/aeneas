#!/usr/bin/env python
# coding=utf-8

"""
Read text fragments, from stdin or from file
"""

import sys

import aeneas.globalconstants as gc
from aeneas.textfile import TextFile
from aeneas.tools import get_rel_path

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl (www.readbeyond.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.0.4"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

def usage():
    name = "aeneas.tools.read_text"
    file_path_1 = get_rel_path("../tests/res/inputtext/sonnet_parsed.txt")
    file_path_2 = get_rel_path("../tests/res/inputtext/sonnet_plain.txt")
    file_path_3 = get_rel_path("../tests/res/inputtext/sonnet_unparsed_class_id.xhtml")
    print ""
    print "Usage:"
    print "  $ python -m %s [list|parsed|plain|unparsed] /path/to/text_file [parameters]" % name 
    print ""
    print "Example:"
    print "  $ python -m %s list     'fragment 1|fragment 2|fragment 3'" % name
    print "  $ python -m %s parsed   %s" % (name, file_path_1)
    print "  $ python -m %s plain    %s" % (name, file_path_2)
    print "  $ python -m %s unparsed %s id_regex=f[0-9]*" % (name, file_path_3)
    print "  $ python -m %s unparsed %s class_regex=ra   sort=unsorted" % (name, file_path_3)
    print "  $ python -m %s unparsed %s id_regex=f[0-9]* sort=numeric" % (name, file_path_3)
    print "  $ python -m %s unparsed %s id_regex=f[0-9]* sort=lexicographic" % (name, file_path_3)
    print ""

def main():
    if len(sys.argv) < 3:
        usage()
        return

    text_format = sys.argv[1]
    file_path = sys.argv[2]

    parameters = {}
    for i in range(3, len(sys.argv)):
        key, value = sys.argv[i].split("=")
        if key == "id_regex":
            parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = value
        if key == "class_regex":
            parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX] = value
        if key == "sort":
            parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT] = value

    if text_format == "list":
        text_file = TextFile()
        text_file.read_from_list(file_path.split("|"))
    else:
        text_file = TextFile(file_path, text_format, parameters)
    print str(text_file)

if __name__ == '__main__':
    main()



