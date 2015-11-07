#!/usr/bin/env python
# coding=utf-8

"""
Probe the properties of a given audio file
"""

import numpy
import sys

from aeneas.audiofile import AudioFileMonoWAV
from aeneas.audiofile import AudioFileUnsupportedFormatError
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

NAME = "aeneas.tools.extract_mfcc"

INPUT_FILE = gf.get_rel_path("res/audio.wav")
OUTPUT_FILE = "output/audio.wav.mfcc.txt"

def usage():
    """ Print usage message """
    print ""
    print "Usage:"
    print "  $ python -m %s /path/to/audio_file.mono.wav /path/to/output.txt [-v]" % NAME
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
    file_path = sys.argv[1]
    save_path = sys.argv[2]
    verbose = False
    for i in range(3, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-v":
            verbose = True

    if not gf.can_run_c_extension():
        print "[WARN] Unable to load Python C Extensions"
        print "[WARN] Running the slower pure Python code"
        print "[WARN] See the README file for directions to compile the Python C Extensions"

    if not gf.file_can_be_written(save_path):
        print "[ERRO] Unable to create file '%s'" % save_path
        print "[ERRO] Make sure the output file path is written/escaped correctly and that you have write permission on it"
        sys.exit(1)

    logger = Logger(tee=verbose)
    try:
        audiofile = AudioFileMonoWAV(file_path, logger=logger)
        audiofile.load_data()
        audiofile.extract_mfcc()
        audiofile.clear_data()
    except (AudioFileUnsupportedFormatError, IOError):
        print "[ERRO] Cannot read file '%s'" % file_path
        print "[ERRO] Make sure it is a mono WAV file and that its path is written/escaped correctly"
        sys.exit(1)
    numpy.savetxt(save_path, audiofile.audio_mfcc)
    print "[INFO] MFCCs saved to %s" % (save_path)
    sys.exit(0)

if __name__ == '__main__':
    main()



