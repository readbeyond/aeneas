#!/usr/bin/env python
# coding=utf-8

"""
Synthesize text using the ``espeak`` wrapper.
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from aeneas.espeakwrapper import ESPEAKWrapper
from aeneas.language import Language
from aeneas.textfile import TextFile
from aeneas.tools.abstract_cli_program import AbstractCLIProgram
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.4.1"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class ESPEAKWrapperCLI(AbstractCLIProgram):
    """
    Synthesize text using the ``espeak`` wrapper.
    """
    OUTPUT_FILE = "output/sonnet.wav"
    TEXT = u"From fairest creatures we desire increase"
    TEXT_MULTI = u"From|fairest|creatures|we|desire|increase"

    NAME = gf.file_name_without_extension(__file__)

    HELP = {
        "description": u"Synthesize text using the espeak wrapper.",
        "synopsis": [
            u"TEXT LANGUAGE OUTPUT_FILE"
        ],
        "examples": [
            u"\"%s\" en %s" % (TEXT, OUTPUT_FILE),
            u"\"%s\" en %s -m" % (TEXT_MULTI, OUTPUT_FILE)
        ],
        "options": [
            u"-m, --multiple : text contains multiple fragments, separated by a '|' character",
        ]
    }

    def perform_command(self):
        """
        Perform command and return the appropriate exit code.

        :rtype: int
        """
        if len(self.actual_arguments) < 3:
            return self.print_help()
        text = gf.safe_unicode(self.actual_arguments[0])
        language = gf.safe_unicode(self.actual_arguments[1])
        output_file_path = self.actual_arguments[2]
        multiple = self.has_option([u"-m", u"--multiple"])

        if (not language in Language.ALLOWED_VALUES) and (not self.rconf["allow_unlisted_languages"]):
            self.print_error(u"Language code '%s' is not allowed." % language)
            return self.ERROR_EXIT_CODE

        if not self.check_output_file(output_file_path):
            return self.ERROR_EXIT_CODE

        try:
            synt = ESPEAKWrapper(rconf=self.rconf, logger=self.logger)
            if multiple:
                tfl = TextFile()
                tfl.read_from_list(text.split("|"))
                tfl.set_language(language)
                synt.synthesize_multiple(
                    tfl,
                    output_file_path
                )
            else:
                synt.synthesize_single(
                    text,
                    language,
                    output_file_path
                )
            self.print_info(u"Created file '%s'" % output_file_path)
            return self.NO_ERROR_EXIT_CODE
        except OSError:
            self.print_error(u"Unable to create file '%s'" % output_file_path)

        return self.ERROR_EXIT_CODE



def main():
    """
    Execute program.
    """
    ESPEAKWrapperCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()



