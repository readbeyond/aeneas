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
Synthesize several text fragments,
producing a WAV audio file.
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from aeneas.language import Language
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.synthesizer import Synthesizer
from aeneas.textfile import TextFileFormat
from aeneas.tools.abstract_cli_program import AbstractCLIProgram
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf


class SynthesizeTextCLI(AbstractCLIProgram):
    """
    Synthesize several text fragments,
    producing a WAV audio file.
    """
    OUTPUT_FILE = "output/synthesized.wav"
    TEXT_FILE_MPLAIN = gf.relative_path("res/mplain.txt", __file__)
    TEXT_FILE_MUNPARSED = gf.relative_path("res/munparsed2.xhtml", __file__)
    TEXT_FILE_PARSED = gf.relative_path("res/parsed.txt", __file__)
    TEXT_FILE_PLAIN = gf.relative_path("res/plain.txt", __file__)
    TEXT_FILE_SUBTITLES = gf.relative_path("res/subtitles.txt", __file__)
    TEXT_FILE_UNPARSED = gf.relative_path("res/unparsed.xhtml", __file__)

    NAME = gf.file_name_without_extension(__file__)

    HELP = {
        "description": u"Synthesize several text fragments.",
        "synopsis": [
            (u"list 'fragment 1|fragment 2|...|fragment N' LANGUAGE OUTPUT_FILE", True),
            (u"[mplain|munparsed|parsed|plain|subtitles|unparsed] TEXT_FILE LANGUAGE OUTPUT_FILE", True)
        ],
        "examples": [
            u"list 'From|fairest|creatures|we|desire|increase' eng %s" % (OUTPUT_FILE),
            u"mplain %s eng %s" % (TEXT_FILE_MPLAIN, OUTPUT_FILE),
            u"munparsed %s eng %s --l1-id-regex=p[0-9]+ --l2-id-regex=s[0-9]+ --l3-id-regex=w[0-9]+" % (TEXT_FILE_MUNPARSED, OUTPUT_FILE),
            u"parsed %s eng %s" % (TEXT_FILE_PARSED, OUTPUT_FILE),
            u"plain %s eng %s" % (TEXT_FILE_PLAIN, OUTPUT_FILE),
            u"subtitles %s eng %s" % (TEXT_FILE_SUBTITLES, OUTPUT_FILE),
            u"unparsed %s eng %s --id-regex=f[0-9]*" % (TEXT_FILE_UNPARSED, OUTPUT_FILE),
            u"unparsed %s eng %s --class-regex=ra" % (TEXT_FILE_UNPARSED, OUTPUT_FILE),
            u"unparsed %s eng %s --id-regex=f[0-9]* --sort=numeric" % (TEXT_FILE_UNPARSED, OUTPUT_FILE),
            u"plain %s eng %s --start=5" % (TEXT_FILE_PLAIN, OUTPUT_FILE),
            u"plain %s eng %s --end=10" % (TEXT_FILE_PLAIN, OUTPUT_FILE),
            u"plain %s eng %s --start=5 --end=10" % (TEXT_FILE_PLAIN, OUTPUT_FILE),
            u"plain %s eng %s --backwards" % (TEXT_FILE_PLAIN, OUTPUT_FILE),
            u"plain %s eng %s --quit-after=10.0" % (TEXT_FILE_PLAIN, OUTPUT_FILE),
        ],
        "options": [
            u"--class-regex=REGEX : extract text from elements with class attribute matching REGEX (unparsed)",
            u"--end=INDEX : slice the text file until fragment INDEX",
            u"--id-format=FMT : use FMT for generating text id attributes (subtitles, plain)",
            u"--id-regex=REGEX : extract text from elements with id attribute matching REGEX (unparsed)",
            u"--l1-id-regex=REGEX : extract text from level 1 elements with id attribute matching REGEX (munparsed)",
            u"--l2-id-regex=REGEX : extract text from level 2 elements with id attribute matching REGEX (munparsed)",
            u"--l3-id-regex=REGEX : extract text from level 3 elements with id attribute matching REGEX (munparsed)",
            u"--quit-after=DUR : synthesize fragments until DUR seconds or the end of text is reached",
            u"--sort=ALGORITHM : sort the matched element id attributes using ALGORITHM (lexicographic, numeric, unsorted)",
            u"--start=INDEX : slice the text file from fragment INDEX",
            u"-b, --backwards : synthesize from the last fragment to the first one",
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
        backwards = self.has_option([u"-b", u"--backwards"])
        quit_after = gf.safe_float(self.has_option_with_value(u"--quit-after"), None)
        start_fragment = gf.safe_int(self.has_option_with_value(u"--start"), None)
        end_fragment = gf.safe_int(self.has_option_with_value(u"--end"), None)
        parameters = {
            gc.PPN_TASK_IS_TEXT_MUNPARSED_L1_ID_REGEX: l1_id_regex,
            gc.PPN_TASK_IS_TEXT_MUNPARSED_L2_ID_REGEX: l2_id_regex,
            gc.PPN_TASK_IS_TEXT_MUNPARSED_L3_ID_REGEX: l3_id_regex,
            gc.PPN_TASK_IS_TEXT_UNPARSED_CLASS_REGEX: class_regex,
            gc.PPN_TASK_IS_TEXT_UNPARSED_ID_REGEX: id_regex,
            gc.PPN_TASK_IS_TEXT_UNPARSED_ID_SORT: sort,
        }
        if (text_format == TextFileFormat.MUNPARSED) and ((l1_id_regex is None) or (l2_id_regex is None) or (l3_id_regex is None)):
            self.print_error(u"You must specify --l1-id-regex and --l2-id-regex and --l3-id-regex for munparsed format")
            return self.ERROR_EXIT_CODE
        if (text_format == TextFileFormat.UNPARSED) and (id_regex is None) and (class_regex is None):
            self.print_error(u"You must specify --id-regex and/or --class-regex for unparsed format")
            return self.ERROR_EXIT_CODE

        language = gf.safe_unicode(self.actual_arguments[2])

        output_file_path = self.actual_arguments[3]
        if not self.check_output_file(output_file_path):
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
        if start_fragment is not None:
            self.print_info(u"Slicing from index %d" % (start_fragment))
        if end_fragment is not None:
            self.print_info(u"Slicing to index %d" % (end_fragment))
        text_slice = text_file.get_slice(start_fragment, end_fragment)
        self.print_info(u"Synthesizing %d fragments" % (len(text_slice)))

        if quit_after is not None:
            self.print_info(u"Stop synthesizing upon reaching %.3f seconds" % (quit_after))

        try:
            synt = Synthesizer(rconf=self.rconf, logger=self.logger)
            synt.synthesize(
                text_slice,
                output_file_path,
                quit_after=quit_after,
                backwards=backwards
            )
            self.print_success(u"Created file '%s'" % output_file_path)
            synt.clear_cache()
            return self.NO_ERROR_EXIT_CODE
        except ImportError as exc:
            tts = self.rconf[RuntimeConfiguration.TTS]
            if tts == Synthesizer.AWS:
                self.print_error(u"You need to install Python module boto3 to use the AWS Polly TTS API wrapper. Run:")
                self.print_error(u"$ pip install boto3")
                self.print_error(u"or, to install for all users:")
                self.print_error(u"$ sudo pip install boto3")
            elif tts == Synthesizer.NUANCE:
                self.print_error(u"You need to install Python module requests to use the Nuance TTS API wrapper. Run:")
                self.print_error(u"$ pip install requests")
                self.print_error(u"or, to install for all users:")
                self.print_error(u"$ sudo pip install requests")
            else:
                self.print_error(u"An unexpected error occurred while synthesizing text:")
                self.print_error(u"%s" % exc)
        except Exception as exc:
            self.print_error(u"An unexpected error occurred while synthesizing text:")
            self.print_error(u"%s" % exc)

        return self.ERROR_EXIT_CODE


def main():
    """
    Execute program.
    """
    SynthesizeTextCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()
