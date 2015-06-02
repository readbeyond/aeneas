#!/usr/bin/env python
# coding=utf-8

"""
Probe the properties of a given audio file
"""

import sys

from aeneas.audiofile import AudioFile
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
    name = "aeneas.tools.read_audio"
    file_path = get_rel_path("../tests/res/container/job/assets/p001.mp3")
    print ""
    print "Usage:"
    print "  $ python -m %s /path/to/audio_file" % name
    print ""
    print "Example:"
    print "  $ python -m %s %s" % (name, file_path)
    print ""

def main():
    if len(sys.argv) < 2:
        usage()
        return
    file_path = sys.argv[1]
    audiofile = AudioFile(file_path)
    print str(audiofile)

if __name__ == '__main__':
    main()



