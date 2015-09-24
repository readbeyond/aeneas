#!/usr/bin/env python
# coding=utf-8

"""
Convert a sync map from a format to another.
"""

import sys

import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf
from aeneas.tools import get_rel_path
from aeneas.syncmap import SyncMap, SyncMapFormat

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.2.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

def usage():
    """ Print usage message """
    name = "aeneas.tools.convert_syncmap"
    file_path_1 = get_rel_path("../tests/res/syncmaps/sonnet001.csv")
    file_path_2 = get_rel_path("../tests/res/syncmaps/sonnet001.zzz")
    # TODO file_path_3 = get_rel_path("../tests/res/syncmaps/sonnet001.smil")
    # TODO file_path_text = get_rel_path("../tests/res/inputtext/sonnet_plain.txt")
    print ""
    print "Usage:"
    print "  $ python -m %s input_syncmap_file output_syncmap_file [parameters]" % name
    print ""
    print "Examples:"
    print "  $ python -m %s %s /tmp/syncmap.srt" % (name, file_path_1)
    print "  $ python -m %s %s /tmp/syncmap.dat output_format=txt" % (name, file_path_1)
    print "  $ python -m %s %s /tmp/syncmap.txt input_format=csv" % (name, file_path_2)
    print "  $ python -m %s %s /tmp/syncmap.txt language=en" % (name, file_path_1)
    print "  $ python -m %s %s /tmp/syncmap.smil os_task_file_smil_audio_ref=audio/sonnet001.mp3 os_task_file_smil_page_ref=text/sonnet001.xhtml" % (name, file_path_1)
    # TODO print "  $ python -m %s %s /tmp/syncmap.srt text_file=%s" % (name, file_path_3, file_path_text)
    print ""

def main():
    """ Entry point """
    if len(sys.argv) < 3:
        usage()
        return
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    parameters = {}
    for i in range(3, len(sys.argv)):
        args = sys.argv[i].split("=")
        if len(args) == 2:
            key, value = args
            if key in [
                    gc.PPN_SYNCMAP_LANGUAGE,
                    gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF,
                    gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF,
                    "input_format",
                    "output_format",
                    "text_file"
            ]:
                parameters[key] = value

    if "input_format" in parameters:
        input_sm_format = parameters["input_format"]
    else:
        input_sm_format = gf.file_extension(input_file_path)
    if input_sm_format not in SyncMapFormat.ALLOWED_VALUES:
        print "[ERRO] Input sync map format '%s' is not allowed" % (input_sm_format)
        print "[INFO] Allowed formats: %s" % (" ".join(SyncMapFormat.ALLOWED_VALUES))
        return

    if "output_format" in parameters:
        output_sm_format = parameters["output_format"]
    else:
        output_sm_format = gf.file_extension(output_file_path)
    if output_sm_format not in SyncMapFormat.ALLOWED_VALUES:
        print "[ERRO] Output sync map format '%s' is not allowed" % (output_sm_format)
        print "[INFO] Allowed sync map formats: %s" % (" ".join(SyncMapFormat.ALLOWED_VALUES))
        return

    try:
        print "[INFO] Reading sync map in %s format from file %s ..." % (input_sm_format, input_file_path)
        syncmap = SyncMap()
        result = syncmap.read(input_sm_format, input_file_path, parameters)
        if not result:
            print "[ERRO] Error while reading sync map"
            return
        print "[INFO] Reading sync map in %s format from file %s ... done" % (input_sm_format, input_file_path)
        print "[INFO] Read %s sync map fragments" % (len(syncmap))
        print "[INFO] Writing sync map in %s format to file %s ..." % (output_sm_format, output_file_path)
        result = syncmap.write(output_sm_format, output_file_path, parameters)
        if not result:
            print "[ERRO] Error while writing sync map (forgot required arguments?)"
            return
        print "[INFO] Writing sync map in %s format to file %s ... done" % (output_sm_format, output_file_path)
    except Exception as e:
        print "[ERRO] Uncaught exception %s" % (str(e))
        return

    


if __name__ == '__main__':
    main()



