#!/usr/bin/env python
# coding=utf-8

"""
Convert a sync map from a format to another.
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from aeneas.syncmap import SyncMap
from aeneas.syncmap import SyncMapFormat
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

class ConvertSyncMapCLI(AbstractCLIProgram):
    """
    Convert a sync map from a format to another.
    """
    AUDIO = gf.relative_path("res/audio.mp3", __file__)
    SMIL_PARAMETERS = "--audio-ref=audio/sonnet001.mp3 --page-ref=text/sonnet001.xhtml"
    SYNC_MAP_CSV = gf.relative_path("res/sonnet.csv", __file__)
    SYNC_MAP_JSON = gf.relative_path("res/sonnet.json", __file__)
    SYNC_MAP_ZZZ = gf.relative_path("res/sonnet.zzz", __file__)
    OUTPUT_HTML = "output/sonnet.html"
    OUTPUT_MAP_DAT = "output/syncmap.dat"
    OUTPUT_MAP_JSON = "output/syncmap.json"
    OUTPUT_MAP_SMIL = "output/syncmap.smil"
    OUTPUT_MAP_SRT = "output/syncmap.srt"
    OUTPUT_MAP_TXT = "output/syncmap.txt"

    NAME = gf.file_name_without_extension(__file__)

    HELP = {
        "description": u"Convert a sync map from a format to another.",
        "synopsis": [
            (u"INPUT_SYNCMAP OUTPUT_SYNCMAP", True),
            (u"INPUT_SYNCMAP OUTPUT_HTML AUDIO_FILE --output-html", True),
        ],
        "examples": [
            u"%s %s" % (SYNC_MAP_JSON, OUTPUT_MAP_SRT),
            u"%s %s --output-format=txt" % (SYNC_MAP_JSON, OUTPUT_MAP_DAT),
            u"%s %s --input-format=csv" % (SYNC_MAP_ZZZ, OUTPUT_MAP_TXT),
            u"%s %s --language=en" % (SYNC_MAP_CSV, OUTPUT_MAP_JSON),
            u"%s %s %s" % (SYNC_MAP_JSON, OUTPUT_MAP_SMIL, SMIL_PARAMETERS),
            u"%s %s %s --output-html" % (SYNC_MAP_JSON, OUTPUT_HTML, AUDIO)
        ],
        "options": [
            u"--audio-ref=REF : use REF for the audio ref attribute (smil, smilh, smilm)",
            u"--input-format=FMT : input sync map file has format FMT",
            u"--language=CODE : set language to CODE",
            u"--output-format=FMT : output sync map file has format FMT",
            u"--output-html : output HTML file for fine tuning",
            u"--page-ref=REF : use REF for the text ref attribute (smil, smilh, smilm)"
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
        output_html = self.has_option(u"--output-html")

        if not self.check_input_file(input_file_path):
            return self.ERROR_EXIT_CODE
        input_sm_format = self.has_option_with_value(u"--input-format")
        if input_sm_format is None:
            input_sm_format = gf.file_extension(input_file_path)
        if not self.check_format(input_sm_format):
            return self.ERROR_EXIT_CODE

        if not self.check_output_file(output_file_path):
            return self.ERROR_EXIT_CODE

        if output_html:
            if len(self.actual_arguments) < 3:
                return self.print_help()
            audio_file_path = self.actual_arguments[2]
            if not self.check_input_file(audio_file_path):
                return self.ERROR_EXIT_CODE
        else:
            output_sm_format = self.has_option_with_value(u"--output-format")
            if output_sm_format is None:
                output_sm_format = gf.file_extension(output_file_path)
            if not self.check_format(output_sm_format):
                return self.ERROR_EXIT_CODE

        # TODO add a way to specify a text file for input formats like SMIL
        #      that do not carry the source text
        language = self.has_option_with_value(u"--language")
        audio_ref = self.has_option_with_value(u"--audio-ref")
        page_ref = self.has_option_with_value(u"--page-ref")
        parameters = {
            gc.PPN_SYNCMAP_LANGUAGE : language,
            gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF : audio_ref,
            gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF : page_ref
        }

        try:
            self.print_info(u"Reading sync map in '%s' format from file '%s'" % (input_sm_format, input_file_path))
            self.print_info(u"Reading sync map...")
            syncmap = SyncMap(logger=self.logger)
            syncmap.read(input_sm_format, input_file_path, parameters)
            self.print_info(u"Reading sync map... done")
            self.print_info(u"Read %d sync map fragments" % (len(syncmap)))
        except Exception as exc:
            self.print_error(u"An unexpected error occurred while reading the input sync map:")
            self.print_error(u"%s" % (exc))
            return self.ERROR_EXIT_CODE

        if output_html:
            try:
                self.print_info(u"Writing HTML file...")
                syncmap.output_html_for_tuning(audio_file_path, output_file_path, parameters)
                self.print_info(u"Writing HTML file... done")
                self.print_success(u"Created HTML file '%s'" % (output_file_path))
                return self.NO_ERROR_EXIT_CODE
            except Exception as exc:
                self.print_error(u"An unexpected error occurred while writing the output HTML file:")
                self.print_error(u"%s" % (exc))
        else:
            try:
                self.print_info(u"Writing sync map...")
                syncmap.write(output_sm_format, output_file_path, parameters)
                self.print_info(u"Writing sync map... done")
                self.print_success(u"Created '%s' sync map file '%s'" % (output_sm_format, output_file_path))
                return self.NO_ERROR_EXIT_CODE
            except Exception as exc:
                self.print_error(u"An unexpected error occurred while writing the output sync map:")
                self.print_error(u"%s" % (exc))

        return self.ERROR_EXIT_CODE

    def check_format(self, sm_format):
        """
        Return ``True`` if the given sync map format is allowed,
        and ``False`` otherwise.

        :param sm_format: the sync map format to be checked
        :type  sm_format: Unicode string
        :rtype: bool
        """
        if sm_format not in SyncMapFormat.ALLOWED_VALUES:
            self.print_error(u"Sync map format '%s' is not allowed" % (sm_format))
            self.print_info(u"Allowed formats:")
            self.print_generic(u" ".join(SyncMapFormat.ALLOWED_VALUES))
            return False
        return True



def main():
    """
    Execute program.
    """
    ConvertSyncMapCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()



