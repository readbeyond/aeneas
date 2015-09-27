#!/usr/bin/env python
# coding=utf-8

"""
Synthesize several text fragments,
read from file or from stdin,
producing a wav audio file.
"""

import sys

import aeneas.globalconstants as gc
from aeneas.synthesizer import Synthesizer
from aeneas.textfile import TextFile
from aeneas.tools import get_rel_path

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
    name = "aeneas.tools.synthesize_text"
    file_path_1 = get_rel_path("../tests/res/inputtext/sonnet_plain.txt")
    file_path_2 = get_rel_path("../tests/res/inputtext/sonnet_parsed.txt")
    file_path_3 = get_rel_path("../tests/res/inputtext/sonnet_subtitles.txt")
    file_path_4 = get_rel_path("../tests/res/inputtext/sonnet_unparsed_class_id.xhtml")
    print ""
    print "Usage:"
    print "  $ python -m %s language 'fragment 1|fragment 2|...|fragment N' list [parameters] /path/to/output_audio_file " % name
    print "  $ python -m %s language /path/to/text_file [parsed|plain|subtitles|unparsed] [parameters] /path/to/output_audio_file " % name
    print ""
    print "Examples:"
    print "  $ python -m %s en 'From fairest creatures|we desire|increase' list /tmp/output.wav" % (name)
    print "  $ python -m %s en %s plain /tmp/output.wav" % (name, file_path_1)
    print "  $ python -m %s en %s parsed /tmp/output.wav" % (name, file_path_2)
    print "  $ python -m %s en %s subtitles /tmp/output.wav" % (name, file_path_3)
    print "  $ python -m %s en %s unparsed id_regex=f[0-9]* /tmp/output.wav" % (name, file_path_4)
    print "  $ python -m %s en %s unparsed class_regex=ra /tmp/output.wav" % (name, file_path_4)
    print "  $ python -m %s en %s unparsed id_regex=f[0-9]* sort=unsorted /tmp/output.wav" % (name, file_path_4)
    print "  $ python -m %s en %s unparsed id_regex=f[0-9]* sort=numeric /tmp/output.wav" % (name, file_path_4)
    print "  $ python -m %s en %s unparsed id_regex=f[0-9]* sort=lexicographic /tmp/output.wav" % (name, file_path_4)
    print "  $ python -m %s en %s plain start=5 /tmp/output.wav" % (name, file_path_1)
    print "  $ python -m %s en %s plain end=10 /tmp/output.wav" % (name, file_path_1)
    print "  $ python -m %s en %s plain start=5 end=10 /tmp/output.wav" % (name, file_path_1)
    print "  $ python -m %s en %s plain backwards /tmp/output.wav" % (name, file_path_1)
    print "  $ python -m %s en %s plain quit_after=10 /tmp/output.wav" % (name, file_path_1)
    print ""

def main():
    """ Entry point """
    if len(sys.argv) < 5:
        usage()
        return
    language = sys.argv[1]
    text_file_path = sys.argv[2]
    text_format = sys.argv[3]
    audio_file_path = sys.argv[-1]
    backwards = False
    quit_after = None
    parameters = {}

    for i in range(4, len(sys.argv)-1):
        args = sys.argv[i].split("=")
        if len(args) == 1:
            backwards = (args[0] in ["b", "-b", "backwards", "--backwards"])
        if len(args) == 2:
            key, value = args
            if key == "id_regex":
                parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = value
            if key == "class_regex":
                parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX] = value
            if key == "sort":
                parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT] = value
            if (key == "start") or (key == "end"):
                try:
                    parameters[key] = int(value)
                except:
                    pass
            if key == "quit_after":
                quit_after = float(value)

    if text_format == "list":
        text_file = TextFile()
        text_file.read_from_list(text_file_path.split("|"))
    else:
        text_file = TextFile(text_file_path, text_format, parameters)
    text_file.set_language(language)

    start_fragment = None
    if "start" in parameters:
        start_fragment = parameters["start"]

    end_fragment = None
    if "end" in parameters:
        end_fragment = parameters["end"]

    print "[INFO] Read input text file with %d fragments" % (len(text_file))
    if start_fragment is not None:
        print "[INFO] Slicing from index %d" % (start_fragment)
    if end_fragment is not None:
        print "[INFO] Slicing to index %d" % (end_fragment)
    text_slice = text_file.get_slice(start_fragment, end_fragment)
    print "[INFO] Synthesizing %d fragments" % (len(text_slice))
    if quit_after is not None:
        print "[INFO] Stop synthesizing after reaching %.3f seconds" % (quit_after)
    if backwards:
        print "[INFO] Synthesizing backwards"
    synt = Synthesizer()
    synt.synthesize(text_slice, audio_file_path, quit_after, backwards)
    print "[INFO] Created file '%s'" % audio_file_path

if __name__ == '__main__':
    main()



