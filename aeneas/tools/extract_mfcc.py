#!/usr/bin/env python
# coding=utf-8

"""
Extract MFCCs from a given audio file.
"""

from __future__ import absolute_import
from __future__ import print_function
import io
import sys
import numpy

from aeneas.audiofile import AudioFileConverterError
from aeneas.audiofile import AudioFileNotInitializedError
from aeneas.audiofile import AudioFileUnsupportedFormatError
from aeneas.audiofilemfcc import AudioFileMFCC
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

class ExtractMFCCCLI(AbstractCLIProgram):
    """
    Extract MFCCs from a given audio file.
    """
    INPUT_FILE = gf.relative_path("res/audio.wav", __file__)
    OUTPUT_FILE = "output/audio.wav.mfcc.txt"

    NAME = gf.file_name_without_extension(__file__)

    HELP = {
        "description": u"Extract MFCCs from a given audio file as a fat matrix.",
        "synopsis": [
            (u"AUDIO_FILE OUTPUT_FILE", True)
        ],
        "examples": [
            u"%s %s" % (INPUT_FILE, OUTPUT_FILE)
        ],
        "options": [
            u"-b, --binary : output MFCCs as a float64 binary file",
            u"-d, --delete-first : do not output the 0th MFCC coefficient",
            u"-n, --npy : output MFCCs as a NumPy .npy binary file",
            u"-t, --transpose : transpose the MFCCs matrix, returning a tall matrix",
            u"-z, --npz : output MFCCs as a NumPy compressed .npz binary file",
            u"--format=FMT : output to text file using format FMT (default: '%.18e')"
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

        output_text_format = self.has_option_with_value(u"--format")
        if output_text_format is None:
            output_text_format = u"%.18e"
        output_binary = self.has_option([u"-b", u"--binary"])
        output_npz = self.has_option([u"-z", u"--npz"])
        output_npy = self.has_option([u"-n", u"--npy"])
        delete_first = self.has_option([u"-d", u"--delete-first"])
        transpose = self.has_option([u"-t", u"--transpose"])

        self.check_c_extensions("cmfcc")
        if not self.check_input_file(input_file_path):
            return self.ERROR_EXIT_CODE
        if not self.check_output_file(output_file_path):
            return self.ERROR_EXIT_CODE

        try:
            mfccs = AudioFileMFCC(input_file_path, rconf=self.rconf, logger=self.logger).all_mfcc
            if delete_first:
                mfccs = mfccs[1:, :]
            if transpose:
                mfccs = mfccs.transpose()
            if output_binary:
                # save as a raw C float64 binary file
                mapped = numpy.memmap(output_file_path, dtype="float64", mode="w+", shape=mfccs.shape)
                mapped[:] = mfccs[:]
                mapped.flush()
                del mapped
            elif output_npz:
                # save as a .npz compressed binary file
                with io.open(output_file_path, "wb") as output_file:
                    numpy.savez(output_file, mfccs)
            elif output_npy:
                # save as a .npy binary file
                with io.open(output_file_path, "wb") as output_file:
                    numpy.save(output_file, mfccs)
            else:
                # save as a text file
                # NOTE: in Python 2, passing the fmt value a Unicode string crashes NumPy
                #       hence, converting back to bytes, which works in Python 3 too
                numpy.savetxt(output_file_path, mfccs, fmt=gf.safe_bytes(output_text_format))
            self.print_info(u"MFCCs shape: %d %d" % (mfccs.shape))
            self.print_success(u"MFCCs saved to '%s'" % (output_file_path))
            return self.NO_ERROR_EXIT_CODE
        except AudioFileConverterError:
            self.print_error(u"Unable to call the ffmpeg executable '%s'" % (self.rconf[RuntimeConfiguration.FFMPEG_PATH]))
            self.print_error(u"Make sure the path to ffmpeg is correct")
        except (AudioFileUnsupportedFormatError, AudioFileNotInitializedError):
            self.print_error(u"Cannot read file '%s'" % (input_file_path))
            self.print_error(u"Check that its format is supported by ffmpeg")
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



