#!/usr/bin/env python
# coding=utf-8

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2016, Alberto Pettarin (www.albertopettarin.it)
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
Read audio file properties.
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from aeneas.audiofile import AudioFile
from aeneas.audiofile import AudioFileProbeError
from aeneas.audiofile import AudioFileUnsupportedFormatError
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.tools.abstract_cli_program import AbstractCLIProgram
import aeneas.globalfunctions as gf


class ReadAudioCLI(AbstractCLIProgram):
    """
    Read audio file properties.
    """
    AUDIO_FILE = gf.relative_path("res/audio.mp3", __file__)

    NAME = gf.file_name_without_extension(__file__)

    HELP = {
        "description": u"Read audio file properties.",
        "synopsis": [
            (u"AUDIO_FILE", True)
        ],
        "options": [
        ],
        "parameters": [
        ],
        "examples": [
            u"%s" % (AUDIO_FILE)
        ]
    }

    def perform_command(self):
        """
        Perform command and return the appropriate exit code.

        :rtype: int
        """
        if len(self.actual_arguments) < 1:
            return self.print_help()
        audio_file_path = self.actual_arguments[0]

        try:
            audiofile = AudioFile(audio_file_path, rconf=self.rconf, logger=self.logger)
            audiofile.read_properties()
            self.print_generic(audiofile.__unicode__())
            return self.NO_ERROR_EXIT_CODE
        except OSError:
            self.print_error(u"Cannot read file '%s'" % (audio_file_path))
            self.print_error(u"Make sure the input file path is written/escaped correctly")
        except AudioFileProbeError:
            self.print_error(u"Unable to call the ffprobe executable '%s'" % (self.rconf[RuntimeConfiguration.FFPROBE_PATH]))
            self.print_error(u"Make sure the path to ffprobe is correct")
        except AudioFileUnsupportedFormatError:
            self.print_error(u"Cannot read properties of file '%s'" % (audio_file_path))
            self.print_error(u"Make sure the input file has a format supported by ffprobe")

        return self.ERROR_EXIT_CODE


def main():
    """
    Execute program.
    """
    ReadAudioCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()
