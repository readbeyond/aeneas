#!/usr/bin/env python
# coding=utf-8

"""
Read text fragments from file.
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

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

class ReadTextCLI(AbstractCLIProgram):
    """
    Read text fragments from file.
    """
    TEXT_FILE_MPLAIN = gf.relative_path("res/mplain.txt", __file__)
    TEXT_FILE_MUNPARSED = gf.relative_path("res/munparsed2.xhtml", __file__)
    TEXT_FILE_PARSED = gf.relative_path("res/parsed.txt", __file__)
    TEXT_FILE_PLAIN = gf.relative_path("res/plain.txt", __file__)
    TEXT_FILE_SUBTITLES = gf.relative_path("res/subtitles.txt", __file__)
    TEXT_FILE_UNPARSED = gf.relative_path("res/unparsed.xhtml", __file__)

    NAME = gf.file_name_without_extension(__file__)

    HELP = {
        "description": u"Read text fragments from file.",
        "synopsis": [
            (u"list 'fragment 1|fragment 2|...|fragment N'", True),
            (u"[mplain|munparsed|parsed|plain|subtitles|unparsed] TEXT_FILE", True)
        ],
        "options": [
            u"--class-regex=REGEX : extract text from elements with class attribute matching REGEX (unparsed)",
            u"--id-regex=REGEX : extract text from elements with id attribute matching REGEX (unparsed)",
            u"--id-format=FMT : use FMT for generating text id attributes (subtitles, plain)",
            u"--l1-id-regex=REGEX : extract text from level 1 elements with id attribute matching REGEX (munparsed)",
            u"--l2-id-regex=REGEX : extract text from level 2 elements with id attribute matching REGEX (munparsed)",
            u"--l3-id-regex=REGEX : extract text from level 3 elements with id attribute matching REGEX (munparsed)",
            u"--sort=ALGORITHM : sort the matched element id attributes using ALGORITHM (lexicographic, numeric, unsorted)",
        ],
        "examples": [
            u"list 'From|fairest|creatures|we|desire|increase'",
            u"mplain %s" % (TEXT_FILE_MPLAIN),
            u"munparsed %s --l1-id-regex=p[0-9]+ --l2-id-regex=s[0-9]+ --l3-id-regex=w[0-9]+" % (TEXT_FILE_MUNPARSED),
            u"parsed %s" % (TEXT_FILE_PARSED),
            u"plain %s" % (TEXT_FILE_PLAIN),
            u"plain %s --id-format=Word%%06d" % (TEXT_FILE_PLAIN),
            u"subtitles %s" % (TEXT_FILE_SUBTITLES),
            u"subtitles %s --id-format=Sub%%03d" % (TEXT_FILE_SUBTITLES),
            u"unparsed %s --id-regex=f[0-9]*" % (TEXT_FILE_UNPARSED),
            u"unparsed %s --class-regex=ra --sort=unsorted" % (TEXT_FILE_UNPARSED),
            u"unparsed %s --id-regex=f[0-9]* --sort=numeric" % (TEXT_FILE_UNPARSED),
            u"unparsed %s --id-regex=f[0-9]* --sort=lexicographic" % (TEXT_FILE_UNPARSED)
        ]
    }

    def perform_command(self):
        """
        Perform command and return the appropriate exit code.

        :rtype: int
        """
        if len(self.actual_arguments) < 2:
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
        id_format = self.has_option_with_value(u"--id-format")
        class_regex = self.has_option_with_value(u"--class-regex")
        sort = self.has_option_with_value(u"--sort")
        parameters = {
            gc.PPN_TASK_IS_TEXT_MUNPARSED_L1_ID_REGEX : l1_id_regex,
            gc.PPN_TASK_IS_TEXT_MUNPARSED_L2_ID_REGEX : l2_id_regex,
            gc.PPN_TASK_IS_TEXT_MUNPARSED_L3_ID_REGEX : l3_id_regex,
            gc.PPN_TASK_IS_TEXT_UNPARSED_ID_REGEX : id_regex,
            gc.PPN_TASK_IS_TEXT_UNPARSED_CLASS_REGEX : class_regex,
            gc.PPN_TASK_IS_TEXT_UNPARSED_ID_SORT : sort,
            gc.PPN_TASK_OS_FILE_ID_REGEX : id_format
        }
        if (text_format == TextFileFormat.MUNPARSED) and ((l1_id_regex is None) or (l2_id_regex is None) or (l3_id_regex is None)):
            self.print_error(u"You must specify --l1-id-regex and --l2-id-regex and --l3-id-regex for munparsed format")
            return self.ERROR_EXIT_CODE
        if (text_format == TextFileFormat.UNPARSED) and (id_regex is None) and (class_regex is None):
            self.print_error(u"You must specify --id-regex and/or --class-regex for unparsed format")
            return self.ERROR_EXIT_CODE
        if (text_format in [TextFileFormat.PLAIN, TextFileFormat.SUBTITLES]) and (id_format is not None):
            try:
                identifier = id_format % 1
            except (TypeError, ValueError):
                self.print_error(u"The given string '%s' is not a valid id format" % id_format)
                return self.ERROR_EXIT_CODE

        text_file = self.get_text_file(text_format, text, parameters)
        if text_file is None:
            self.print_error(u"Unable to build a TextFile from the given parameters")
        elif len(text_file) == 0:
            self.print_error(u"No text fragments found")
        else:
            self.print_generic(text_file.__unicode__())
            return self.NO_ERROR_EXIT_CODE
        return self.ERROR_EXIT_CODE



def main():
    """
    Execute program.
    """
    ReadTextCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()



