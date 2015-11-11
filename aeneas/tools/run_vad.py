#!/usr/bin/env python
# coding=utf-8

"""
Extract a list of speech intervals from the given audio file,
using the MFCC energy-based VAD algorithm.
"""

import sys
import tempfile

from aeneas.audiofile import AudioFileMonoWAV
from aeneas.audiofile import AudioFileUnsupportedFormatError
from aeneas.ffmpegwrapper import FFMPEGWrapper
from aeneas.logger import Logger
from aeneas.vad import VAD
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

NAME = "aeneas.tools.run_vad"

INPUT_FILE = gf.get_rel_path("res/audio.mp3")
OUTPUT_BOTH = "output/both.txt"
OUTPUT_NONSPEECH = "output/nonspeech.txt"
OUTPUT_SPEECH = "output/speech.txt"

def usage():
    """ Print usage message """
    print ""
    print "Usage:"
    print "  $ python -m %s /path/to/audio_file [both|nonspeech|speech] /path/to/output_file [-v]" % NAME
    print ""
    print "Options:"
    print "  -v : verbose output"
    print ""
    print "Examples:"
    print "  $ python -m %s %s both      %s" % (NAME, INPUT_FILE, OUTPUT_BOTH)
    print "  $ python -m %s %s nonspeech %s" % (NAME, INPUT_FILE, OUTPUT_NONSPEECH)
    print "  $ python -m %s %s speech    %s" % (NAME, INPUT_FILE, OUTPUT_SPEECH)
    print ""
    sys.exit(2)

def write_to_file(output_file_path, intervals, labels=False):
    output_file = open(output_file_path, "w")
    for interval in intervals:
        if labels:
            output_file.write("%.3f\t%.3f\t%s\n" % (
                interval[0],
                interval[1],
                interval[2]
            ))
        else:
            output_file.write("%.3f\t%.3f\n" % (
                interval[0], interval[1]
            ))
    output_file.close()

def main():
    """ Entry point """
    if len(sys.argv) < 4:
        usage()
    audio_file_path = sys.argv[1]
    tmp_handler, tmp_file_path = tempfile.mkstemp(
        suffix=".wav",
        dir=gf.custom_tmp_dir()
    )
    mode = sys.argv[2]
    output_file_path = sys.argv[3]
    verbose = False
    for i in range(4, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-v":
            verbose = True
    if mode not in ["speech", "nonspeech", "both"]:
        usage()

    if not gf.can_run_c_extension("cmfcc"):
        print "[WARN] Unable to load Python C Extension cmfcc"
        print "[WARN] Running the (a bit slower) pure Python code"
        print "[WARN] See the README file for directions to compile the Python C Extensions"

    logger = Logger(tee=verbose)

    try:
        print "[INFO] Converting audio file to mono..."
        converter = FFMPEGWrapper(logger=logger)
        converter.convert(audio_file_path, tmp_file_path)
        print "[INFO] Converting audio file to mono... done"
    except IOError:
        print "[ERRO] Cannot convert audio file '%s'" % audio_file_path
        print "[ERRO] Check that it exists and that its path is written/escaped correctly"
        sys.exit(1)

    try:
        print "[INFO] Extracting MFCCs..."
        audiofile = AudioFileMonoWAV(tmp_file_path)
        audiofile.extract_mfcc()
        print "[INFO] Extracting MFCCs... done"
    except (AudioFileUnsupportedFormatError, IOError):
        print "[ERRO] Cannot read the converted WAV file '%s'" % tmp_file_path
        sys.exit(1)

    print "[INFO] Executing VAD..."
    vad = VAD(audiofile.audio_mfcc, audiofile.audio_length, logger=logger)
    vad.compute_vad()
    print "[INFO] Executing VAD... done"

    gf.delete_file(tmp_handler, tmp_file_path)

    if mode == "speech":
        write_to_file(output_file_path, vad.speech)
    elif mode == "nonspeech":
        write_to_file(output_file_path, vad.nonspeech)
    elif mode == "both":
        speech = [[x[0], x[1], "speech"] for x in vad.speech]
        nonspeech = [[x[0], x[1], "nonspeech"] for x in vad.nonspeech]
        both = sorted(speech + nonspeech)
        write_to_file(output_file_path, both, labels=True)
    print "[INFO] Created file %s" % output_file_path
    sys.exit(0)

if __name__ == '__main__':
    main()



