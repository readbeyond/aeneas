#!/usr/bin/env python
# coding=utf-8

"""
Extract a list of speech intervals from the given audio file,
using the MFCC energy-based VAD algorithm.
"""

from __future__ import absolute_import
from __future__ import print_function
import io
import sys

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

class RunVADCLI(AbstractCLIProgram):
    """
    Extract a list of speech intervals from the given audio file,
    using the MFCC energy-based VAD algorithm.
    """
    INPUT_FILE = gf.relative_path("res/audio.mp3", __file__)
    OUTPUT_BOTH = "output/both.txt"
    OUTPUT_NONSPEECH = "output/nonspeech.txt"
    OUTPUT_SPEECH = "output/speech.txt"
    MODES = [u"both", u"nonspeech", u"speech"]

    NAME = gf.file_name_without_extension(__file__)

    HELP = {
        "description": u"Extract a list of speech intervals using the MFCC energy-based VAD.",
        "synopsis": [
            (u"AUDIO_FILE [%s] [OUTPUT_FILE]" % (u"|".join(MODES)), True)
        ],
        "examples": [
            u"%s both %s" % (INPUT_FILE, OUTPUT_BOTH),
            u"%s nonspeech %s" % (INPUT_FILE, OUTPUT_NONSPEECH),
            u"%s speech %s" % (INPUT_FILE, OUTPUT_SPEECH)
        ],
        "options": [
            u"-i, --index : output intervals as indices instead of seconds",
        ]
    }

    def perform_command(self):
        """
        Perform command and return the appropriate exit code.

        :rtype: int
        """
        if len(self.actual_arguments) < 2:
            return self.print_help()
        audio_file_path = self.actual_arguments[0]
        mode = self.actual_arguments[1]
        if mode not in [u"speech", u"nonspeech", u"both"]:
            return self.print_help()
        output_file_path = None
        if len(self.actual_arguments) >= 3:
            output_file_path = self.actual_arguments[2]
        output_time = not self.has_option([u"-i", u"--index"])

        self.check_c_extensions("cmfcc")
        if not self.check_input_file(audio_file_path):
            return self.ERROR_EXIT_CODE
        if (output_file_path is not None) and (not self.check_output_file(output_file_path)):
            return self.ERROR_EXIT_CODE

        self.print_info(u"Reading audio...")
        try:
            audio_file_mfcc = AudioFileMFCC(audio_file_path, rconf=self.rconf, logger=self.logger)
        except AudioFileConverterError:
            self.print_error(u"Unable to call the ffmpeg executable '%s'" % (self.rconf[RuntimeConfiguration.FFMPEG_PATH]))
            self.print_error(u"Make sure the path to ffmpeg is correct")
            return self.ERROR_EXIT_CODE
        except (AudioFileUnsupportedFormatError, AudioFileNotInitializedError):
            self.print_error(u"Cannot read file '%s'" % (audio_file_path))
            self.print_error(u"Check that its format is supported by ffmpeg")
            return self.ERROR_EXIT_CODE
        except Exception as exc:
            self.print_error(u"An unexpected error occurred while reading the audio file:")
            self.print_error(u"%s" % exc)
            return self.ERROR_EXIT_CODE
        self.print_info(u"Reading audio... done")

        self.print_info(u"Executing VAD...")
        audio_file_mfcc.run_vad()
        self.print_info(u"Executing VAD... done")

        speech = audio_file_mfcc.intervals(speech=True, time=output_time)
        nonspeech = audio_file_mfcc.intervals(speech=False, time=output_time)
        if mode == u"speech":
            intervals = speech
        elif mode == u"nonspeech":
            intervals = nonspeech
        elif mode == u"both":
            speech = [[x[0], x[1], u"speech"] for x in speech]
            nonspeech = [[x[0], x[1], u"nonspeech"] for x in nonspeech]
            intervals = sorted(speech + nonspeech)
        intervals = [tuple(interval) for interval in intervals]
        self.write_to_file(output_file_path, intervals, output_time)

        return self.NO_ERROR_EXIT_CODE

    def write_to_file(self, output_file_path, intervals, time):
        """
        Write intervals to file.

        :param output_file_path: path of the output file to be written;
                                 if ``None``, print to stdout
        :type  output_file_path: string (path)
        :param intervals: a list of tuples, each representing an interval
        :type  intervals: list of tuples
        """
        msg = []
        if len(intervals) > 0:
            if len(intervals[0]) == 2:
                template = u"%.3f\t%.3f" if time else u"%d\t%d"
            else:
                template = u"%.3f\t%.3f\t%s" if time else u"%d\t%d\t%s"
            msg = [template % (interval) for interval in intervals]
        if output_file_path is None:
            self.print_info(u"Intervals detected:")
            for line in msg:
                self.print_generic(line)
        else:
            with io.open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write(u"\n".join(msg))
                self.print_success(u"Created file '%s'" % output_file_path)



def main():
    """
    Execute program.
    """
    RunVADCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()



