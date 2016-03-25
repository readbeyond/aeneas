#!/usr/bin/env python
# coding=utf-8

"""
Download a file from a Web source.

Currently, it downloads an audio file from a YouTube video.
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from aeneas.downloader import Downloader
from aeneas.tools.abstract_cli_program import AbstractCLIProgram
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.5.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class DownloadCLI(AbstractCLIProgram):
    """
    Download a file from a Web source.

    Currently, it downloads an audio file from a YouTube video.
    """
    OUTPUT_FILE_M4A = "output/sonnet.m4a"
    OUTPUT_FILE_OGG = "output/sonnet.ogg"
    URL_YOUTUBE = "https://www.youtube.com/watch?v=rU4a7AA8wM0"

    NAME = gf.file_name_without_extension(__file__)

    HELP = {
        "description": u"Download an audio file from a YouTube video.",
        "synopsis": [
            (u"YOUTUBE_URL [OUTPUT_FILE]", True)
        ],
        "examples": [
            u"%s --list" % (URL_YOUTUBE),
            u"%s %s" % (URL_YOUTUBE, OUTPUT_FILE_M4A),
            u"%s %s --index=0" % (URL_YOUTUBE, OUTPUT_FILE_M4A),
            u"%s %s --smallest-audio" % (URL_YOUTUBE, OUTPUT_FILE_OGG),
            u"%s %s --largest-audio --format=ogg" % (URL_YOUTUBE, OUTPUT_FILE_OGG),
        ],
        "options": [
            u"--format=FMT : preferably download audio stream in FMT format",
            u"--index=IDX : download audio stream with given index",
            u"--largest-audio : download largest audio stream (default)",
            u"--list : list all available audio streams but do not download",
            u"--smallest-audio : download smallest audio stream",
        ]
    }

    def perform_command(self):
        """
        Perform command and return the appropriate exit code.

        :rtype: int
        """
        if len(self.actual_arguments) < 2:
            return self.print_help()
        source_url = self.actual_arguments[0]
        output_file_path = self.actual_arguments[1]

        download = not self.has_option("--list")
        # largest_audio = True by default or if explicitly given
        if self.has_option("--largest-audio"):
            largest_audio = True
        else:
            largest_audio = not self.has_option("--smallest-audio")
        preferred_format = self.has_option_with_value("--format")
        preferred_index = gf.safe_int(self.has_option_with_value("--index"), None)

        try:
            if download:
                self.print_info(u"Downloading audio stream from '%s' ..." % source_url)
                downloader = Downloader(logger=self.logger)
                result = downloader.audio_from_youtube(
                    source_url,
                    download=download,
                    output_file_path=output_file_path,
                    preferred_index=preferred_index,
                    largest_audio=largest_audio,
                    preferred_format=preferred_format
                )
                self.print_info(u"Downloading audio stream from '%s' ... done" % source_url)
                self.print_success(u"Downloaded file '%s'" % result)
            else:
                self.print_info(u"Downloading stream info from '%s' ..." % source_url)
                downloader = Downloader(logger=self.logger)
                result = downloader.audio_from_youtube(
                    source_url,
                    download=False
                )
                self.print_info(u"Downloading stream info from '%s' ... done" % source_url)
                msg = []
                msg.append(u"%s\t%s\t%s\t%s" % ("Index", "Format", "Bitrate", "Size"))
                i = 0
                for audio in result:
                    ext = audio.extension
                    bitrate = audio.bitrate
                    size = gf.human_readable_number(audio.get_filesize())
                    msg.append(u"%d\t%s\t%s\t%s" % (i, ext, bitrate, size))
                    i += 1
                self.print_generic(u"Available audio streams:")
                self.print_generic(u"\n".join(msg))
            return self.NO_ERROR_EXIT_CODE
        except ImportError:
            self.print_error(u"You need to install Python module pafy to download audio from YouTube. Run:")
            self.print_error(u"$ pip install pafy")
            self.print_error(u"or, to install for all users:")
            self.print_error(u"$ sudo pip install pafy")
        except Exception as exc:
            self.print_error(u"An unexpected Exception occurred while downloading audio from YouTube:")
            self.print_error(u"%s" % exc)

        return self.ERROR_EXIT_CODE


def main():
    """
    Execute program.
    """
    DownloadCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()



