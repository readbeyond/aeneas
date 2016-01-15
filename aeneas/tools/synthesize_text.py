#!/usr/bin/env python
# coding=utf-8

"""
Synthesize several text fragments,
producing a WAV audio file.
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from aeneas.language import Language
from aeneas.synthesizer import Synthesizer
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
__version__ = "1.4.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class SynthesizeTextCLI(AbstractCLIProgram):
    """
    Synthesize several text fragments,
    producing a WAV audio file.
    """
    OUTPUT_FILE = "output/synthesized.wav"
    TEXT_FILE_PARSED = gf.relative_path("res/parsed.txt", __file__)
    TEXT_FILE_PLAIN = gf.relative_path("res/plain.txt", __file__)
    TEXT_FILE_SUBTITLES = gf.relative_path("res/subtitles.txt", __file__)
    TEXT_FILE_UNPARSED = gf.relative_path("res/unparsed.xhtml", __file__)

    NAME = gf.file_name_without_extension(__file__)

    HELP = {
        "description": u"Synthesize several text fragments.",
        "synopsis": [
            u"list 'fragment 1|fragment 2|...|fragment N' LANGUAGE OUTPUT_FILE",
            u"[parsed|plain|subtitles|unparsed] TEXT_FILE LANGUAGE OUTPUT_FILE"
        ],
        "examples": [
            u"list 'From|fairest|creatures|we|desire|increase' en %s" % (OUTPUT_FILE),
            u"parsed %s en %s" % (TEXT_FILE_PARSED, OUTPUT_FILE),
            u"plain %s en %s" % (TEXT_FILE_PLAIN, OUTPUT_FILE),
            u"subtitles %s en %s" % (TEXT_FILE_SUBTITLES, OUTPUT_FILE),
            u"unparsed %s en %s --id-regex=f[0-9]*" % (TEXT_FILE_UNPARSED, OUTPUT_FILE),
            u"unparsed %s en %s --class-regex=ra" % (TEXT_FILE_UNPARSED, OUTPUT_FILE),
            u"unparsed %s en %s --id-regex=f[0-9]* --sort=numeric" % (TEXT_FILE_UNPARSED, OUTPUT_FILE),
            u"plain %s en %s --start=5" % (TEXT_FILE_PLAIN, OUTPUT_FILE),
            u"plain %s en %s --end=10" % (TEXT_FILE_PLAIN, OUTPUT_FILE),
            u"plain %s en %s --start=5 --end=10" % (TEXT_FILE_PLAIN, OUTPUT_FILE),
            u"plain %s en %s --backwards" % (TEXT_FILE_PLAIN, OUTPUT_FILE),
            u"plain %s en %s --quit-after=10.0" % (TEXT_FILE_PLAIN, OUTPUT_FILE),
            u"plain %s en %s --backwards --quit-after=10.0" % (TEXT_FILE_PLAIN, OUTPUT_FILE),
            u"plain %s en %s --pure" % (TEXT_FILE_PLAIN, OUTPUT_FILE)
        ],
        "options": [
            u"--allow-unlisted-language : allow using a language code not officially supported",
            u"--class-regex=REGEX : extract text from elements with class attribute matching REGEX (unparsed)",
            u"--end=INDEX : slice the text file until fragment INDEX",
            u"--id-format=FMT : use FMT for generating text id attributes (subtitles, plain)",
            u"--id-regex=REGEX : extract text from elements with id attribute matching REGEX (unparsed)",
            u"--quit-after=DUR : synthesize fragments until DUR seconds or the end of text is reached",
            u"--sort=ALGORITHM : sort the matched element id attributes using ALGORITHM (lexicographic, numeric, unsorted)",
            u"--start=INDEX : slice the text file from fragment INDEX",
            u"-b, --backwards : synthesize from the last fragment to the first one",
            u"-p, --pure : use pure Python code, even if the C extension is available"
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
        elif text_format in [u"parsed", u"plain", u"subtitles", u"unparsed"]:
            text = self.actual_arguments[1]
            if not self.check_input_file(text):
                return self.ERROR_EXIT_CODE
        else:
            return self.print_help()

        id_regex = self.has_option_with_value(u"--id-regex")
        class_regex = self.has_option_with_value(u"--class-regex")
        sort = self.has_option_with_value(u"--sort")
        backwards = self.has_option([u"-b", u"--backwards"])
        quit_after = gf.safe_float(self.has_option_with_value(u"--quit-after"), None)
        start_fragment = gf.safe_int(self.has_option_with_value(u"--start"), None)
        end_fragment = gf.safe_int(self.has_option_with_value(u"--end"), None)
        force_pure_python = self.has_option([u"-p", u"--pure"])
        unlisted_language = self.has_option(u"--allow-unlisted-language")
        parameters = {
            gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX : id_regex,
            gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX : class_regex,
            gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT : sort
        }
        if (text_format == u"unparsed") and (id_regex is None) and (class_regex is None):
            self.print_error(u"You must specify --id-regex and/or --class-regex for unparsed format")
            return self.ERROR_EXIT_CODE

        language = gf.safe_unicode(self.actual_arguments[2])
        if not language in Language.ALLOWED_VALUES:
            self.print_error(u"Language '%s' is not supported" % (language))
            return self.ERROR_EXIT_CODE

        output_file_path = self.actual_arguments[3]
        if not self.check_output_file(output_file_path):
            return self.ERROR_EXIT_CODE

        text_file = self.get_text_file(text_format, text, parameters)
        if text_file is None:
            self.print_error(u"Unable to build a TextFile from the given parameters")
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
        if backwards:
            self.print_info(u"Synthesizing backwards => forcing pure Python code")
            force_pure_python = True
        elif force_pure_python:
            self.print_info(u"Forcing pure Python code")

        try:
            synt = Synthesizer(logger=self.logger)
            synt.synthesize(
                text_slice,
                output_file_path,
                quit_after=quit_after,
                backwards=backwards,
                force_pure_python=force_pure_python,
                allow_unlisted_languages=unlisted_language
            )
            self.print_info(u"Created file '%s'" % output_file_path)
            return self.NO_ERROR_EXIT_CODE
        except Exception as exc:
            self.print_error(u"An unexpected Exception occurred while synthesizing text:")
            self.print_error(u"%s" % exc)

        return self.ERROR_EXIT_CODE


def main():
    """
    Execute program.
    """
    SynthesizeTextCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()



