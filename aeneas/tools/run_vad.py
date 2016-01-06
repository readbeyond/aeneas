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

from aeneas.audiofile import AudioFileMonoWAVE
from aeneas.audiofile import AudioFileUnsupportedFormatError
from aeneas.ffmpegwrapper import FFMPEGWrapper
from aeneas.tools.abstract_cli_program import AbstractCLIProgram
from aeneas.vad import VAD
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
            u"AUDIO_FILE [%s] OUTPUT_FILE" % (u"|".join(MODES))
        ],
        "examples": [
            u"%s both %s" % (INPUT_FILE, OUTPUT_BOTH),
            u"%s nonspeech %s" % (INPUT_FILE, OUTPUT_NONSPEECH),
            u"%s speech %s" % (INPUT_FILE, OUTPUT_SPEECH)
        ]
    }

    def perform_command(self):
        """
        Perform command and return the appropriate exit code.

        :rtype: int
        """
        if len(self.actual_arguments) < 3:
            return self.print_help()
        audio_file_path = self.actual_arguments[0]
        mode = self.actual_arguments[1]
        output_file_path = self.actual_arguments[2]
        if mode not in [u"speech", u"nonspeech", u"both"]:
            return self.print_help()

        self.check_c_extensions("cmfcc")
        if not self.check_input_file(audio_file_path):
            return self.ERROR_EXIT_CODE
        if not self.check_output_file(output_file_path):
            return self.ERROR_EXIT_CODE

        tmp_handler, tmp_file_path = gf.tmp_file(suffix=".wav")
        try:
            self.print_info(u"Converting audio file to mono...")
            converter = FFMPEGWrapper(logger=self.logger)
            converter.convert(audio_file_path, tmp_file_path)
            self.print_info(u"Converting audio file to mono... done")
        except OSError:
            self.print__error(u"Cannot convert audio file '%s'" % audio_file_path)
            self.print_error(u"Check that its format is supported by ffmpeg")
            return self.ERROR_EXIT_CODE

        try:
            self.print_info(u"Extracting MFCCs...")
            audiofile = AudioFileMonoWAVE(tmp_file_path)
            audiofile.extract_mfcc()
            self.print_info(u"Extracting MFCCs... done")
        except (AudioFileUnsupportedFormatError, OSError):
            self.print_error(u"Cannot read the converted WAV file '%s'" % tmp_file_path)
            return self.ERROR_EXIT_CODE

        self.print_info(u"Executing VAD...")
        vad = VAD(audiofile.audio_mfcc, audiofile.audio_length, logger=self.logger)
        vad.compute_vad()
        self.print_info(u"Executing VAD... done")

        gf.delete_file(tmp_handler, tmp_file_path)

        if mode == u"speech":
            intervals = vad.speech
        elif mode == u"nonspeech":
            intervals = vad.nonspeech
        elif mode == u"both":
            speech = [[x[0], x[1], u"speech"] for x in vad.speech]
            nonspeech = [[x[0], x[1], u"nonspeech"] for x in vad.nonspeech]
            intervals = sorted(speech + nonspeech)
        intervals = [tuple(interval) for interval in intervals]
        self.write_to_file(output_file_path, intervals)

        return self.NO_ERROR_EXIT_CODE

    def write_to_file(self, output_file_path, intervals):
        """
        Write intervals to file.

        :param output_file_path: path of the output file to be written
        :type  output_file_path: string (path)
        :param intervals: a list of tuples, each representing an interval
        :type  intervals: list of tuples
        """
        msg = []
        if len(intervals) > 0:
            if len(intervals[0]) == 2:
                template = u"%.3f\t%.3f"
            else:
                template = u"%.3f\t%.3f\t%s"
            msg = [template % (interval) for interval in intervals]
        with io.open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(u"\n".join(msg))
            self.print_info(u"Created file %s" % output_file_path)



def main():
    """
    Execute program.
    """
    RunVADCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()



