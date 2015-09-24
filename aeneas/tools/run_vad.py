#!/usr/bin/env python
# coding=utf-8

"""
Extract a list of speech intervals from the given audio file,
using the MFCC energy-based VAD algorithm.
"""

import os
import sys
import tempfile

import aeneas.globalfunctions as gf
from aeneas.ffmpegwrapper import FFMPEGWrapper
from aeneas.logger import Logger
from aeneas.tools import get_rel_path
from aeneas.vad import VAD
import aeneas.globalfunctions as gf

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
    name = "aeneas.tools.run_vad"
    dir_path = get_rel_path("../tests/res/example_jobs/example1/OEBPS/Resources")
    print ""
    print "Usage:"
    print "  $ python -m %s path/to/audio.mp3 speech    /path/to/speech.txt    [-v]" % name
    print "  $ python -m %s path/to/audio.mp3 nonspeech /path/to/nonspeech.txt [-v]" % name
    print "  $ python -m %s path/to/audio.mp3 both      /path/to/both.txt      [-v]" % name
    print ""
    print "Example:"
    print "  $ python -m %s %s/sonnet001.mp3 speech    /tmp/speech.txt" % (name, dir_path)
    print "  $ python -m %s %s/sonnet001.mp3 nonspeech /tmp/nonspeech.txt" % (name, dir_path)
    print "  $ python -m %s %s/sonnet001.mp3 both      /tmp/both.txt" % (name, dir_path)
    print ""

def cleanup(handler, path):
    """ Remove temporary hadler/file """
    if handler is not None:
        try:
            os.close(handler)
        except:
            pass
    if path is not None:
        try:
            os.remove(path)
        except:
            pass

def main():
    """ Entry point """
    if len(sys.argv) < 4:
        usage()
        return
    audio_file_path = sys.argv[1]
    tmp_handler, tmp_file_path = tempfile.mkstemp(
        suffix=".wav",
        dir=gf.custom_tmp_dir()
    )
    mode = sys.argv[2]
    output_file_path = sys.argv[3]
    verbose = (sys.argv[-1] == "-v")

    if mode not in ["speech", "nonspeech", "both"]:
        usage()
        return

    if not gf.can_run_c_extension():
        print "[WARN] Unable to load Python C Extensions"
        print "[WARN] Running the slower pure Python code"
        print "[WARN] See the README file for directions to compile the Python C Extensions"

    logger = Logger(tee=verbose)

    print "[INFO] Converting audio file to mono..."
    converter = FFMPEGWrapper(logger=logger)
    converter.convert(audio_file_path, tmp_file_path)
    print "[INFO] Converting audio file to mono... done"

    vad = VAD(tmp_file_path, logger=logger)
    print "[INFO] Extracting MFCCs..."
    vad.compute_mfcc()
    print "[INFO] Extracting MFCCs... done"
    print "[INFO] Executing VAD..."
    vad.compute_vad()
    print "[INFO] Executing VAD... done"

    print "[INFO] Cleaning up..."
    cleanup(tmp_handler, tmp_file_path)
    print "[INFO] Cleaning up... done"

    if mode == "speech":
        print "[INFO] Creating speech file..."
        output_file = open(output_file_path, "w")
        for interval in vad.speech:
            output_file.write("%.3f\t%.3f\n" % (interval[0], interval[1]))
        output_file.close()
        print "[INFO] Creating speech file... done"

    if mode == "nonspeech":
        print "[INFO] Creating nonspeech file..."
        output_file = open(output_file_path, "w")
        for interval in vad.nonspeech:
            output_file.write("%.3f\t%.3f\n" % (interval[0], interval[1]))
        output_file.close()
        print "[INFO] Creating nonspeech file... done"

    if mode == "both":
        print "[INFO] Creating speech and nonspeech file..."
        output_file = open(output_file_path, "w")
        speech = [[x[0], x[1], "speech"] for x in vad.speech]
        nonspeech = [[x[0], x[1], "nonspeech"] for x in vad.nonspeech]
        both = sorted(speech + nonspeech)
        for interval in both:
            output_file.write("%.3f\t%.3f\t%s\n" % (
                interval[0],
                interval[1],
                interval[2]
            ))
        output_file.close()
        print "[INFO] Creating speech and nonspeech file... done"

    print "[INFO] Created file %s" % output_file_path

if __name__ == '__main__':
    main()



