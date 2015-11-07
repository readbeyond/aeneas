#!/usr/bin/env python
# coding=utf-8

"""
Download a file from a Web source.

Currently, it downloads an audio file from a YouTube video.
"""

import sys

from aeneas.downloader import Downloader
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

NAME = "aeneas.tools.download"

OUTPUT_FILE_M4A = "output/sonnet.m4a"
OUTPUT_FILE_OGG = "output/sonnet.ogg"
URL_YOUTUBE = "https://www.youtube.com/watch?v=rU4a7AA8wM0"

def usage():
    """ Print usage message """
    print ""
    print "Usage:"
    print "  $ python -m %s -h" % NAME
    print "  $ python -m %s URL output_file [options]" % NAME
    print ""
    print "Options:"
    print "  -h, --help       : print full help"
    print "  -v               : verbose output"
    print "  --list           : list all available audio streams but do not download"
    print "  --smallest-audio : download smallest audio stream"
    print "  --best-audio     : download best audio stream (default)"
    print "  --format=FMT     : preferably download audio stream in FMT format"
    print ""
    print "Documentation:"
    print "  Please visit http://www.readbeyond.it/aeneas/docs/"
    print ""
    print "Example:"
    print "  $ python -m %s %s --list" % (NAME, URL_YOUTUBE)
    print "  $ python -m %s %s %s" % (NAME, URL_YOUTUBE, OUTPUT_FILE_M4A)
    print "  $ python -m %s %s %s --best-audio" % (NAME, URL_YOUTUBE, OUTPUT_FILE_M4A)
    print "  $ python -m %s %s %s --best-audio --format=ogg" % (NAME, URL_YOUTUBE, OUTPUT_FILE_OGG)
    print ""
    sys.exit(2)

def main():
    """ Entry point """
    if ("-h" in sys.argv) or ("--help" in sys.argv):
        # show full help
        usage()
    if len(sys.argv) < 3:
        usage()
    verbose = False
    best_audio = True
    source_url = sys.argv[1]
    output_file_path = sys.argv[2]
    preferred_format = None
    download = True
    for i in range(2, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-v":
            verbose = True
        elif arg == "--smallest-audio":
            best_audio = False
        elif arg == "--list":
            download = False
        else:
            args = arg.split("=")
            if len(args) == 2:
                key, value = args
                if key == "--format":
                    preferred_format = value

    # check we have enough arguments
    if not download:
        if len(sys.argv) < 3:
            usage()
    elif len(sys.argv) < 3:
        usage()

    logger = Logger(tee=verbose)

    try:
        print "[INFO] Downloading from '%s' ..." % source_url
        downloader = Downloader(logger=logger)
        result = downloader.audio_from_youtube(
            source_url,
            output_file_path=output_file_path,
            best_audio=best_audio,
            preferred_format=preferred_format,
            download=download
        )
        print "[INFO] Downloading from '%s' ... done" % source_url
        if download:
            print "[INFO] Downloaded file '%s'" % result
        else:
            print "[INFO] Available audio streams:"
            print "%s\t%s\t%s\t%s" % ("Index", "Format", "Bitrate", "Size")
            i = 0
            for audio in result:
                ext = audio.extension
                bitrate = audio.bitrate
                size = gf.human_readable_number(audio.get_filesize())
                print "%d\t%s\t%s\t%s" % (i, ext, bitrate, size)
                i += 1
    except ImportError:
        print "[ERRO] You need to install Pythom module pafy to download audio from YouTube. Run:"
        print "[ERRO] $ sudo pip install pafy"
        sys.exit(1)
    except Exception as exc:
        print "[ERRO] The following error occurred while downloading audio from YouTube:"
        print "[ERRO] %s" % str(exc)
        sys.exit(1)

    sys.exit(0)

if __name__ == '__main__':
    main()



