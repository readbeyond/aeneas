#!/usr/bin/env python
# coding=utf-8

"""
Detect the audio head and/or tail of the given audio file.
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from aeneas.audiofile import AudioFileConverterError
from aeneas.audiofile import AudioFileNotInitializedError
from aeneas.audiofile import AudioFileUnsupportedFormatError
from aeneas.audiofilemfcc import AudioFileMFCC
from aeneas.language import Language
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.sd import SD
from aeneas.textfile import TextFileFormat
from aeneas.tools.abstract_cli_program import AbstractCLIProgram
import aeneas.globalconstants as gc
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

class RunSDCLI(AbstractCLIProgram):
    """
    Detect the audio head and/or tail of the given audio file.
    """
    AUDIO_FILE = gf.relative_path("res/audio.mp3", __file__)
    PARAMETERS_HEAD = "--min-head=0.0 --max-head=5.0"
    PARAMETERS_TAIL = "--min-tail=1.0 --max-tail=5.0"
    TEXT_FILE = gf.relative_path("res/parsed.txt", __file__)

    NAME = gf.file_name_without_extension(__file__)

    HELP = {
        "description": u"Detect the audio head and/or tail of the given audio file.",
        "synopsis": [
            (u"list 'fragment 1|fragment 2|...|fragment N' LANGUAGE AUDIO_FILE", True),
            (u"[mplain|munparsed|parsed|plain|subtitles|unparsed] TEXT_FILE LANGUAGE AUDIO_FILE", True)
        ],
        "examples": [
            u"parsed %s eng %s" % (TEXT_FILE, AUDIO_FILE),
            u"parsed %s eng %s %s" % (TEXT_FILE, AUDIO_FILE, PARAMETERS_HEAD),
            u"parsed %s eng %s %s" % (TEXT_FILE, AUDIO_FILE, PARAMETERS_TAIL),
            u"parsed %s eng %s %s %s" % (TEXT_FILE, AUDIO_FILE, PARAMETERS_HEAD, PARAMETERS_TAIL),
        ],
        "options": [
            u"--class-regex=REGEX : extract text from elements with class attribute matching REGEX (unparsed)",
            u"--id-regex=REGEX : extract text from elements with id attribute matching REGEX (unparsed)",
            u"--l1-id-regex=REGEX : extract text from level 1 elements with id attribute matching REGEX (munparsed)",
            u"--l2-id-regex=REGEX : extract text from level 2 elements with id attribute matching REGEX (munparsed)",
            u"--l3-id-regex=REGEX : extract text from level 3 elements with id attribute matching REGEX (munparsed)",
            u"--max-head=DUR : audio head has at most DUR seconds",
            u"--max-tail=DUR : audio tail has at most DUR seconds",
            u"--min-head=DUR : audio head has at least DUR seconds",
            u"--min-tail=DUR : audio tail has at least DUR seconds",
            u"--sort=ALGORITHM : sort the matched element id attributes using ALGORITHM (lexicographic, numeric, unsorted)"
        ]
    }

    def perform_command(self):
        """
        Perform command and return the appropriate exit code.

        :rtype: int
        """
        if len(self.actual_arguments) < 4:
            return self.print_help()
        text_format = gf.safe_unicode(self.actual_arguments[0])
        if text_format == u"list":
            text = gf.safe_unicode(self.actual_arguments[1])
        elif text_format in TextFileFormat.ALLOWED_VALUES:
            text = self.actual_arguments[1]
            if not self.check_input_file(text):
                return self.ERROR_EXIT_CODE
        else:
            return self.print_help()

        l1_id_regex = self.has_option_with_value(u"--l1-id-regex")
        l2_id_regex = self.has_option_with_value(u"--l2-id-regex")
        l3_id_regex = self.has_option_with_value(u"--l3-id-regex")
        id_regex = self.has_option_with_value(u"--id-regex")
        class_regex = self.has_option_with_value(u"--class-regex")
        sort = self.has_option_with_value(u"--sort")
        parameters = {
            gc.PPN_TASK_IS_TEXT_MUNPARSED_L1_ID_REGEX : l1_id_regex,
            gc.PPN_TASK_IS_TEXT_MUNPARSED_L2_ID_REGEX : l2_id_regex,
            gc.PPN_TASK_IS_TEXT_MUNPARSED_L3_ID_REGEX : l3_id_regex,
            gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX : id_regex,
            gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX : class_regex,
            gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT : sort
        }
        if (text_format == TextFileFormat.MUNPARSED) and ((l1_id_regex is None) or (l2_id_regex is None) or (l3_id_regex is None)):
            self.print_error(u"You must specify --l1-id-regex and --l2-id-regex and --l3-id-regex for munparsed format")
            return self.ERROR_EXIT_CODE
        if (text_format == TextFileFormat.UNPARSED) and (id_regex is None) and (class_regex is None):
            self.print_error(u"You must specify --id-regex and/or --class-regex for unparsed format")
            return self.ERROR_EXIT_CODE

        language = gf.safe_unicode(self.actual_arguments[2])

        audio_file_path = self.actual_arguments[3]
        if not self.check_input_file(audio_file_path):
            return self.ERROR_EXIT_CODE

        text_file = self.get_text_file(text_format, text, parameters)
        if text_file is None:
            self.print_error(u"Unable to build a TextFile from the given parameters")
            return self.ERROR_EXIT_CODE
        elif len(text_file) == 0:
            self.print_error(u"No text fragments found")
            return self.ERROR_EXIT_CODE
        text_file.set_language(language)
        self.print_info(u"Read input text with %d fragments" % (len(text_file)))

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

        self.print_info(u"Running VAD...")
        audio_file_mfcc.run_vad()
        self.print_info(u"Running VAD... done")

        min_head = gf.safe_float(self.has_option_with_value(u"--min-head"), None)
        max_head = gf.safe_float(self.has_option_with_value(u"--max-head"), None)
        min_tail = gf.safe_float(self.has_option_with_value(u"--min-tail"), None)
        max_tail = gf.safe_float(self.has_option_with_value(u"--max-tail"), None)

        self.print_info(u"Detecting audio interval...")
        start_detector = SD(audio_file_mfcc, text_file, rconf=self.rconf, logger=self.logger)
        start, end = start_detector.detect_interval(min_head, max_head, min_tail, max_tail)
        self.print_info(u"Detecting audio interval... done")

        self.print_result(audio_file_mfcc.audio_length, start, end)
        return self.NO_ERROR_EXIT_CODE

    def print_result(self, audio_len, start, end):
        """
        Print result of SD.

        :param audio_len: the length of the entire audio file, in seconds
        :type  audio_len: float
        :param start: the start position of the spoken text
        :type  start: float
        :param end: the end position of the spoken text
        :type  end: float
        """
        msg = []
        zero = 0
        head_len = start
        text_len = end - start
        tail_len = audio_len - end
        msg.append(u"")
        msg.append(u"Head: %.3f %.3f (%.3f)" % (zero, start, head_len))
        msg.append(u"Text: %.3f %.3f (%.3f)" % (start, end, text_len))
        msg.append(u"Tail: %.3f %.3f (%.3f)" % (end, audio_len, tail_len))
        msg.append(u"")
        zero_h = gf.time_to_hhmmssmmm(0)
        start_h = gf.time_to_hhmmssmmm(start)
        end_h = gf.time_to_hhmmssmmm(end)
        audio_len_h = gf.time_to_hhmmssmmm(audio_len)
        head_len_h = gf.time_to_hhmmssmmm(head_len)
        text_len_h = gf.time_to_hhmmssmmm(text_len)
        tail_len_h = gf.time_to_hhmmssmmm(tail_len)
        msg.append("Head: %s %s (%s)" % (zero_h, start_h, head_len_h))
        msg.append("Text: %s %s (%s)" % (start_h, end_h, text_len_h))
        msg.append("Tail: %s %s (%s)" % (end_h, audio_len_h, tail_len_h))
        msg.append(u"")
        self.print_info(u"\n".join(msg))



def main():
    """
    Execute program.
    """
    RunSDCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()



