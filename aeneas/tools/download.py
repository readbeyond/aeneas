#!/usr/bin/env python
# coding=utf-8

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2017, Alberto Pettarin (www.albertopettarin.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
            u"%s %s --format=140" % (URL_YOUTUBE, OUTPUT_FILE_M4A),
            u"%s %s --smallest-audio" % (URL_YOUTUBE, OUTPUT_FILE_OGG),
            u"%s %s --largest-audio" % (URL_YOUTUBE, OUTPUT_FILE_M4A),
        ],
        "options": [
            u"--format=IDX : download audio stream with given format",
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
        download_format = self.has_option_with_value("--format")

        try:
            if download:
                self.print_info(u"Downloading audio stream from '%s' ..." % source_url)
                downloader = Downloader(logger=self.logger)
                result = downloader.audio_from_youtube(
                    source_url,
                    download=True,
                    output_file_path=output_file_path,
                    download_format=download_format,
                    largest_audio=largest_audio,
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
                msg.append(u"%s\t%s\t%s\t%s" % ("Format", "Extension", "Bitrate", "Size"))
                for r in result:
                    filesize = gf.human_readable_number(r["filesize"])
                    msg.append(u"%s\t%s\t%s\t%s" % (r["format"], r["ext"], r["abr"], filesize))
                self.print_generic(u"Available audio streams:")
                self.print_generic(u"\n".join(msg))
            return self.NO_ERROR_EXIT_CODE
        except ImportError:
            self.print_no_dependency_error()
        except Exception as exc:
            self.print_error(u"An unexpected error occurred while downloading audio from YouTube:")
            self.print_error(u"%s" % exc)

        return self.ERROR_EXIT_CODE


def main():
    """
    Execute program.
    """
    DownloadCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()
