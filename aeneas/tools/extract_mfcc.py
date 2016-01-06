#!/usr/bin/env python
# coding=utf-8

"""
Extract MFCCs from a given audio file.
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy
import sys

from aeneas.audiofile import AudioFileMonoWAVE
from aeneas.audiofile import AudioFileUnsupportedFormatError
from aeneas.ffmpegwrapper import FFMPEGWrapper
from aeneas.tools.abstract_cli_program import AbstractCLIProgram
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.4.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class ExtractMFCCCLI(AbstractCLIProgram):
    """
    Extract MFCCs from a given audio file.
    """
    INPUT_FILE = gf.relative_path("res/audio.wav", __file__)
    OUTPUT_FILE = "output/audio.wav.mfcc.txt"

    NAME = gf.file_name_without_extension(__file__)

    HELP = {
        "description": u"Extract MFCCs from a given audio file.",
        "synopsis": [
            u"AUDIO_FILE OUTPUT_FILE"
        ],
        "examples": [
            u"%s %s" % (INPUT_FILE, OUTPUT_FILE)
        ],
        "options": [
            u"-p, --pure : use pure Python code, even if the C extension is available"
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
        pure = self.has_option([u"-p", u"--pure"])

        self.check_c_extensions("cmfcc")
        if not self.check_input_file(input_file_path):
            return self.ERROR_EXIT_CODE
        if not self.check_output_file(output_file_path):
            return self.ERROR_EXIT_CODE

        tmp_handler, tmp_file_path = gf.tmp_file(suffix=".wav")
        try:
            self.print_info(u"Converting audio file to mono...")
            converter = FFMPEGWrapper(logger=self.logger)
            converter.convert(input_file_path, tmp_file_path)
            self.print_info(u"Converting audio file to mono... done")
        except OSError:
            self.print_error(u"Cannot convert audio file '%s'" % input_file_path)
            self.print_error(u"Check that its format is supported by ffmpeg")
            return self.ERROR_EXIT_CODE

        try:
            audiofile = AudioFileMonoWAVE(tmp_file_path, logger=self.logger)
            audiofile.load_data()
            audiofile.extract_mfcc(force_pure_python=pure)
            audiofile.clear_data()
            gf.delete_file(tmp_handler, tmp_file_path)
            numpy.savetxt(output_file_path, audiofile.audio_mfcc)
            self.print_info(u"MFCCs saved to %s" % (output_file_path))
            return self.NO_ERROR_EXIT_CODE
        except AudioFileUnsupportedFormatError:
            self.print_error(u"Cannot read file '%s'" % (tmp_file_path))
            self.print_error(u"Check that it is a mono WAV file")
        except OSError:
            self.print_error(u"Cannot write file '%s'" % (output_file_path))

        return self.ERROR_EXIT_CODE



def main():
    """
    Execute program.
    """
    ExtractMFCCCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()



