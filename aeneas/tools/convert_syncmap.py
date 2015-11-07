#!/usr/bin/env python
# coding=utf-8

"""
Convert a sync map from a format to another.
"""

import sys

from aeneas.logger import Logger
from aeneas.syncmap import SyncMap
from aeneas.syncmap import SyncMapFormat
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

NAME = "aeneas.tools.convert_syncmap"

AUDIO = gf.get_rel_path("res/audio.mp3")
SMIL_PARAMETERS = "os_task_file_smil_audio_ref=audio/sonnet001.mp3 os_task_file_smil_page_ref=text/sonnet001.xhtml"
SYNC_MAP_CSV = gf.get_rel_path("res/sonnet.csv")
SYNC_MAP_JSON = gf.get_rel_path("res/sonnet.json")
SYNC_MAP_ZZZ = gf.get_rel_path("res/sonnet.zzz")

OUTPUT_HTML = "output/sonnet.html"
OUTPUT_MAP_DAT = "output/syncmap.dat"
OUTPUT_MAP_JSON = "output/syncmap.json"
OUTPUT_MAP_SMIL = "output/syncmap.smil"
OUTPUT_MAP_SRT = "output/syncmap.srt"
OUTPUT_MAP_TXT = "output/syncmap.txt"

def usage():
    """ Print usage message """
    print ""
    print "Usage:"
    print "  $ python -m %s /path/to/input_sync_map /path/to/output_sync_map [parameters]" % NAME
    print "  $ python -m %s /path/to/input_sync_map /path/to/output_html     audio_file_path=/path/to/audio.file [parameters]" % NAME
    print ""
    print "Parameters:"
    print "  -v                              : verbose output"
    print "  audio_file_path=PATH            : create HTML file for fine tuning, reading audio from PATH"
    print "  input_format=FMT                : input sync map file has format FMT"
    print "  language=CODE                   : set language to CODE"
    print "  output_format=FMT               : output sync map file has format FMT"
    print "  os_task_file_smil_audio_ref=REF : use REF for the audio ref attribute (smil, smilh, smilm)"
    print "  os_task_file_smil_page_ref=REF  : use REF for the text ref attribute (smil, smilh, smilm)"
    print ""
    print "Examples:"
    print "  $ python -m %s %s %s" % (NAME, SYNC_MAP_JSON, OUTPUT_MAP_SRT)
    print "  $ python -m %s %s %s  output_format=txt" % (NAME, SYNC_MAP_JSON, OUTPUT_MAP_DAT)
    print "  $ python -m %s %s  %s  input_format=csv" % (NAME, SYNC_MAP_ZZZ, OUTPUT_MAP_TXT)
    print "  $ python -m %s %s  %s %s=en" % (NAME, SYNC_MAP_CSV, gc.PPN_SYNCMAP_LANGUAGE, OUTPUT_MAP_JSON)
    print "  $ python -m %s %s %s %s" % (NAME, SYNC_MAP_JSON, OUTPUT_MAP_SMIL, SMIL_PARAMETERS)
    print "  $ python -m %s %s %s audio_file_path=%s" % (NAME, SYNC_MAP_JSON, OUTPUT_HTML, AUDIO)
    # TODO a SMIL + text example
    print ""
    sys.exit(2)

def check_format(sm_format, string):
    """ Check that sm_format has an allowed SyncMapFormat value """
    if sm_format not in SyncMapFormat.ALLOWED_VALUES:
        print "[ERRO] %s sync map format '%s' is not allowed" % (string, sm_format)
        print "[INFO] Allowed formats: %s" % (" ".join(SyncMapFormat.ALLOWED_VALUES))
        sys.exit(1)

def get_format(parameters, key, file_path, string):
    """ Get sm_format from parameters or from file extension """
    if key in parameters:
        sm_format = parameters[key]
    else:
        sm_format = gf.file_extension(file_path)
    check_format(sm_format, string)
    return sm_format

def main():
    """ Entry point """
    if len(sys.argv) < 3:
        usage()
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
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
                if key in [
                        gc.PPN_SYNCMAP_LANGUAGE,
                        gc.PPN_TASK_OS_FILE_FORMAT,
                        gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF,
                        gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF,
                        "input_format",
                        "output_format",
                        "text_file",
                        "audio_file_path"
                ]:
                    parameters[key] = value
    input_sm_format = get_format(parameters, "input_format", input_file_path, "Input")
    output_html = ("audio_file_path" in parameters)
    if not output_html:
        output_sm_format = get_format(parameters, "output_format", output_file_path, "Output")

    logger = Logger(tee=verbose)
    try:
        print "[INFO] Reading sync map in %s format from file %s ..." % (input_sm_format, input_file_path)
        syncmap = SyncMap(logger=logger)
        syncmap.read(input_sm_format, input_file_path, parameters)
        print "[INFO] Reading sync map in %s format from file %s ... done" % (input_sm_format, input_file_path)
    except Exception as exc:
        print "[ERRO] The following error occurred while reading the input sync map:"
        print "[ERRO] %s" % str(exc)
        sys.exit(1)
    print "[INFO] Read %s sync map fragments" % (len(syncmap))
    if output_html:
        try:
            print "[INFO] Writing HTML file %s ..." % (output_file_path)
            syncmap.output_html_for_tuning(parameters["audio_file_path"], output_file_path, parameters)
            print "[INFO] Writing HTML file %s ... done" % (output_file_path)
        except IOError as exc:
            print "[ERRO] The following error occurred while writing the output HTML file:"
            print "[ERRO] %s" % str(exc)
            sys.exit(1)
    else:
        try:
            print "[INFO] Writing sync map in %s format to file %s ..." % (output_sm_format, output_file_path)
            syncmap.write(output_sm_format, output_file_path, parameters)
            print "[INFO] Writing sync map in %s format to file %s ... done" % (output_sm_format, output_file_path)
        except IOError as exc:
            print "[ERRO] The following error occurred while writing the output sync map:"
            print "[ERRO] %s" % str(exc)
            sys.exit(1)
    sys.exit(0)    

if __name__ == '__main__':
    main()



