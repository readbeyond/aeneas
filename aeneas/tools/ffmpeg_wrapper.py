#!/usr/bin/env python
# coding=utf-8

"""
Convert audio files to mono WAV using the ``ffmpeg`` wrapper.
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from aeneas.ffmpegwrapper import FFMPEGPathError
from aeneas.ffmpegwrapper import FFMPEGWrapper
from aeneas.runtimeconfiguration import RuntimeConfiguration
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

class FFMPEGWrapperCLI(AbstractCLIProgram):
    """
    Convert audio files to mono WAV using the ``ffmpeg`` wrapper.
    """
    INPUT_FILE = gf.relative_path("res/audio.mp3", __file__)
    OUTPUT_FILE = "output/audio.wav"

    NAME = gf.file_name_without_extension(__file__)

    HELP = {
        "description": u"Convert audio files to mono WAV using the ffmpeg wrapper.",
        "synopsis": [
            (u"INPUT_FILE OUTPUT_FILE", True)
        ],
        "examples": [
            u"%s %s" % (INPUT_FILE, OUTPUT_FILE)
        ]
    }

    def perform_command(self):
        """
        Perform command and return the appropriate exit code.

        :rtype: int
        """
        if len(self.actual_arguments) < 2:
            return self.print_help()
        input_file_path = self.actual_arguments[0]
        output_file_path = self.actual_arguments[1]

        if not self.check_input_file(input_file_path):
            return self.ERROR_EXIT_CODE
        if not self.check_output_file(output_file_path):
            return self.ERROR_EXIT_CODE

        try:
            converter = FFMPEGWrapper(rconf=self.rconf, logger=self.logger)
            converter.convert(input_file_path, output_file_path)
            self.print_success(u"Converted '%s' into '%s'" % (input_file_path, output_file_path))
            return self.NO_ERROR_EXIT_CODE
        except FFMPEGPathError:
            self.print_error(u"Unable to call the ffmpeg executable '%s'" % (self.rconf[RuntimeConfiguration.FFMPEG_PATH]))
            self.print_error(u"Make sure the path to ffmpeg is correct")
        except OSError:
            self.print_error(u"Cannot convert file '%s' into '%s'" % (input_file_path, output_file_path))
            self.print_error(u"Make sure the input file has a format supported by ffmpeg")

        return self.ERROR_EXIT_CODE



def main():
    """
    Execute program.
    """
    FFMPEGWrapperCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()



