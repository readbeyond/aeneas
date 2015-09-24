#!/usr/bin/env python
# coding=utf-8

"""
Probe the properties of a given audio file
"""

import numpy
import sys

import aeneas.globalfunctions as gf
from aeneas.audiofile import AudioFile
from aeneas.tools import get_rel_path

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.1.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

def usage():
    """ Print usage message """
    name = "aeneas.tools.extract_mfcc"
    file_path = get_rel_path("../tests/res/cmfcc/audio.wav")
    print ""
    print "Usage:"
    print "  $ python -m %s /path/to/audio_file.mono.wav /path/to/output.txt" % name
    print ""
    print "Example:"
    print "  $ python -m %s %s /tmp/audio.wav.mfcc.txt" % (name, file_path)
    print ""

def main():
    """ Entry point """
    if len(sys.argv) < 3:
        usage()
        return
    file_path = sys.argv[1]
    save_path = sys.argv[2]

    if not gf.can_run_c_extension():
        print "[WARN] Unable to load Python C Extensions"
        print "[WARN] Running the slower pure Python code"
        print "[WARN] See the README file for directions to compile the Python C Extensions"

    audiofile = AudioFile(file_path)
    audiofile.load_data()
    audiofile.extract_mfcc()
    audiofile.clear_data()
    numpy.savetxt(save_path, audiofile.audio_mfcc)
    print "[INFO] MFCCs saved to %s" % (save_path)

if __name__ == '__main__':
    main()



