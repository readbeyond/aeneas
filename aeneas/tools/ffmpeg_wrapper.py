#!/usr/bin/env python
# coding=utf-8

"""
Use the wrapper around ``ffmpeg``
"""

import sys

from aeneas.ffmpegwrapper import FFMPEGWrapper
from aeneas.tools import get_rel_path

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl (www.readbeyond.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.0.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

def usage():
    name = "aeneas.tools.ffmpeg_wrapper"
    file_path = get_rel_path("../tests/res/container/job/assets/p001.mp3")
    print ""
    print "Usage:"
    print "  $ python -m %s /path/to/input_file /path/to/output_file" % name
    print ""
    print "Example:"
    print "  $ python -m %s %s /tmp/p001.wav" % (name, file_path)
    print ""

def main():
    if len(sys.argv) < 3:
        usage()
        return
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    converter = FFMPEGWrapper()
    converter.convert(input_file_path, output_file_path)
    print "Converted '%s' into '%s'" % (input_file_path, output_file_path)

if __name__ == '__main__':
    main()



