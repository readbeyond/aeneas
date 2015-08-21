#!/usr/bin/env python
# coding=utf-8

"""
Use the wrapper around ``ffprobe``
"""

import sys

from aeneas.ffprobewrapper import FFPROBEWrapper
from aeneas.tools import get_rel_path

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.1.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

def usage():
    """ Print usage message """
    name = "aeneas.tools.ffprobe_wrapper"
    file_path = get_rel_path("../tests/res/container/job/assets/p001.mp3")
    print ""
    print "Usage:"
    print "  $ python -m %s /path/to/audio_file" % name
    print ""
    print "Example:"
    print "  $ python -m %s %s" % (name, file_path)
    print ""

def main():
    """ Entry point """
    if len(sys.argv) < 2:
        usage()
        return
    audio_file_path = sys.argv[1]
    prober = FFPROBEWrapper()
    dictionary = prober.read_properties(audio_file_path)
    for key in sorted(dictionary.keys()):
        print "%s %s" % (key, dictionary[key])

if __name__ == '__main__':
    main()



