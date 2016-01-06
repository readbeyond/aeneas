#!/usr/bin/env python
# coding=utf-8

"""
Execute a Task, that is, a pair of audio/text files
and a configuration string.
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from aeneas.adjustboundaryalgorithm import AdjustBoundaryAlgorithm
from aeneas.downloader import Downloader
from aeneas.executetask import ExecuteTask
from aeneas.idsortingalgorithm import IDSortingAlgorithm
from aeneas.language import Language
from aeneas.logger import Logger
from aeneas.syncmap import SyncMapFormat
from aeneas.syncmap import SyncMapHeadTailFormat
from aeneas.task import Task
from aeneas.textfile import TextFileFormat
from aeneas.tools.abstract_cli_program import AbstractCLIProgram
from aeneas.validator import Validator
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

class ExecuteTaskCLI(AbstractCLIProgram):
    """
    Execute a Task, that is, a pair of audio/text files
    and a configuration string.
    """

    AUDIO_FILE = gf.relative_path("res/audio.mp3", __file__)

    DEMOS = {
        u"--example-json" : {
            u"description": u"input: plain text, output: JSON",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=en|is_text_type=plain|os_task_file_format=json",
            u"syncmap": "output/sonnet.json",
            u"options": u""
        },
        u"--example-srt" : {
            u"description": u"input: subtitles text, output: SRT",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/subtitles.txt", __file__),
            u"config": u"task_language=en|is_text_type=subtitles|os_task_file_format=srt",
            u"syncmap": "output/sonnet.srt",
            u"options": u""
        },
        u"--example-smil" : {
            u"description": u"input: unparsed text, output: SMIL",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/page.xhtml", __file__),
            u"config": u"task_language=en|is_text_type=unparsed|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric|os_task_file_format=smil|os_task_file_smil_audio_ref=p001.mp3|os_task_file_smil_page_ref=p001.xhtml",
            u"syncmap": "output/sonnet.smil",
            u"options": u""
        },
        u"--example-youtube" : {
            u"description": u"input: audio from YouTube, output: TXT",
            u"audio": "https://www.youtube.com/watch?v=rU4a7AA8wM0",
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=en|is_text_type=plain|os_task_file_format=txt",
            u"syncmap": "output/sonnet.txt",
            u"options": u"-y"
        }
    }

    PARAMETERS = [
        u"  task_language                           : language (*)",
        u"",
        u"  is_audio_file_detect_head_max           : detect audio head, at most this many seconds",
        u"  is_audio_file_detect_head_min           : detect audio head, at least this many seconds",
        u"  is_audio_file_detect_tail_max           : detect audio tail, at most this many seconds",
        u"  is_audio_file_detect_tail_min           : detect audio tail, at least this many seconds",
        u"  is_audio_file_head_length               : ignore this many seconds from the begin of the audio file",
        u"  is_audio_file_process_length            : process this many seconds of the audio file",
        u"  is_audio_file_tail_length               : ignore this many seconds from the end of the audio file",
        u"",
        u"  is_text_type                            : input text format (*)",
        u"  is_text_unparsed_class_regex            : regex matching class attributes (unparsed)",
        u"  is_text_unparsed_id_regex               : regex matching id attributes (unparsed)",
        u"  is_text_unparsed_id_sort                : sort matched elements by id (unparsed) (*)",
        u"  is_text_file_ignore_regex               : ignore text matched by regex for audio alignment purposes",
        u"  is_text_file_transliterate_map          : apply the given transliteration map for audio alignment purposes",
        u"",
        u"  os_task_file_format                     : output sync map format (*)",
        u"  os_task_file_id_regex                   : id regex for the output sync map (subtitles, plain)",
        u"  os_task_file_head_tail_format           : format audio head/tail (*)",
        u"  os_task_file_smil_audio_ref             : value for the audio ref (smil, smilh, smilm)",
        u"  os_task_file_smil_page_ref              : value for the text ref (smil, smilh, smilm)",
        u"",
        u"  task_adjust_boundary_algorithm          : adjust sync map fragments using algorithm (*)",
        u"  task_adjust_boundary_aftercurrent_value : offset value, in seconds (aftercurrent)",
        u"  task_adjust_boundary_beforenext_value   : offset value, in seconds (beforenext)",
        u"  task_adjust_boundary_offset_value       : offset value, in seconds (offset)",
        u"  task_adjust_boundary_percent_value      : percent value, in [0..100], (percent)",
        u"  task_adjust_boundary_rate_value         : max rate, in characters/s (rate, rateaggressive)",
    ]

    VALUES = {
        "task_language" : Language,
        "is_text_type" : TextFileFormat,
        "is_text_unparsed_id_sort" : IDSortingAlgorithm,
        "os_task_file_format" : SyncMapFormat,
        "os_task_file_head_tail_format" : SyncMapHeadTailFormat,
        "task_adjust_boundary_algorithm" : AdjustBoundaryAlgorithm,
    }

    NAME = gf.file_name_without_extension(__file__)

    HELP = {
        "description": u"Execute a Task.",
        "synopsis": [
            u"--examples",
            u"[--example-json|--example-srt|--example-smil]",
            u"--list-parameters",
            u"--list-values PARAM",
            u"AUDIO TEXT CONFIG_STRING OUTPUT_FILE",
            u"YOUTUBE_URL TEXT CONFIG_STRING OUTPUT_FILE -y",
        ],
        "examples": [
            u"--examples"
        ],
        "options": [
            u"--allow-unlisted-language : allow using a language code not officially supported",
            u"--example-json : run example with JSON output",
            u"--example-smil : run example with SMIL output",
            u"--example-srt : run example with SRT output",
            u"--example-youtube : run example with audio downloaded from YouTube",
            u"--examples : print a list of examples",
            u"--keep-audio : do not delete the audio file downloaded from YouTube (-y only)",
            u"--largest-audio : download largest audio stream (-y only)",
            u"--list-parameters : list all parameters",
            u"--list-values=PARAM : list all allowed values for parameter PARAM",
            u"--output-html : output HTML file for fine tuning",
            u"--skip-validator : do not validate the given config string",
            u"-e, --examples : print examples",
            u"-y, --youtube : download audio from YouTube video",
        ]
    }

    def perform_command(self):
        """
        Perform command and return the appropriate exit code.

        :rtype: int
        """
        if len(self.actual_arguments) < 1:
            return self.print_help()

        if self.has_option([u"-e", u"--examples"]):
            return self.print_examples()

        if self.has_option([u"--list-parameters"]):
            return self.print_parameters()

        parameter = self.has_option_with_value(u"--list-values")
        if parameter is not None:
            return self.print_values(parameter)
        elif self.has_option(u"--list-values"):
            return self.print_values(u"?")

        # TODO simplify me

        # NOTE list() is needed for Python3, where keys() is not a list!
        demo = self.has_option(list(self.DEMOS.keys()))
        download_from_youtube = self.has_option([u"-y", u"--youtube"])
        largest_audio = self.has_option(u"--largest-audio")
        keep_audio = self.has_option(u"--keep-audio")
        output_html = self.has_option(u"--output-html")
        validate = not self.has_option(u"--skip-validator")
        unlisted_language = self.has_option(u"--allow-unlisted-language")

        if demo:
            validate = False
            for key in self.DEMOS:
                if self.has_option(key):
                    demo_parameters = self.DEMOS[key]
                    audio_file_path = demo_parameters[u"audio"]
                    text_file_path = demo_parameters[u"text"]
                    config_string = demo_parameters[u"config"]
                    sync_map_file_path = demo_parameters[u"syncmap"]
                    if key == u"--example-youtube":
                        download_from_youtube = True
                    break
        else:
            if len(self.actual_arguments) < 4:
                return self.print_help()
            audio_file_path = self.actual_arguments[0]
            text_file_path = self.actual_arguments[1]
            config_string = self.actual_arguments[2]
            sync_map_file_path = self.actual_arguments[3]

        html_file_path = None
        if output_html:
            keep_audio = True
            html_file_path = sync_map_file_path + u".html"

        if download_from_youtube:
            youtube_url = audio_file_path

        if (not download_from_youtube) and (not self.check_input_file(audio_file_path)):
            return self.ERROR_EXIT_CODE
        if not self.check_input_file(text_file_path):
            return self.ERROR_EXIT_CODE
        if not self.check_output_file(sync_map_file_path):
            return self.ERROR_EXIT_CODE
        if (html_file_path is not None) and (not self.check_output_file(html_file_path)):
            return self.ERROR_EXIT_CODE

        self.check_c_extensions()

        if demo:
            msg = []
            msg.append(u"Running example task with arguments:")
            if download_from_youtube:
                msg.append(u"  YouTube URL:   %s" % youtube_url)
            else:
                msg.append(u"  Audio file:    %s" % audio_file_path)
            msg.append(u"  Text file:     %s" % text_file_path)
            msg.append(u"  Config string: %s" % config_string)
            msg.append(u"  Sync map file: %s" % sync_map_file_path)
            self.print_info(u"\n".join(msg))

        if validate:
            self.print_info(u"Validating config string (specify --skip-validator to bypass)...")
            validator = Validator(logger=self.logger)
            result = validator.check_configuration_string(config_string, is_job=False, external_name=True)
            if not result.passed:
                self.print_error(u"The given config string is not valid:")
                self.print_generic(result.pretty_print())
                return self.ERROR_EXIT_CODE
            self.print_info(u"Validating config string... done")

        if download_from_youtube:
            try:
                self.print_info(u"Downloading audio from '%s' ..." % youtube_url)
                downloader = Downloader(logger=self.logger)
                audio_file_path = downloader.audio_from_youtube(
                    youtube_url,
                    download=True,
                    output_file_path=None,
                    largest_audio=largest_audio
                )
                self.print_info(u"Downloading audio from '%s' ... done" % youtube_url)
            except ImportError:
                self.print_error(u"You need to install Pythom module pafy to download audio from YouTube. Run:")
                self.print_error(u"$ sudo pip install pafy")
                return self.ERROR_EXIT_CODE
            except Exception as exc:
                self.print_error(u"An unexpected Exception occurred while downloading audio from YouTube:")
                self.print_error(u"%s" % exc)
                return self.ERROR_EXIT_CODE

        try:
            self.print_info(u"Creating task...")
            task = Task(config_string, logger=self.logger)
            task.audio_file_path_absolute = audio_file_path
            task.text_file_path_absolute = text_file_path
            task.sync_map_file_path_absolute = sync_map_file_path
            self.print_info(u"Creating task... done")
        except Exception as exc:
            self.print_error(u"An unexpected Exception occurred while creating the task:")
            self.print_error(u"%s" % exc)
            return self.ERROR_EXIT_CODE

        try:
            self.print_info(u"Executing task...")
            executor = ExecuteTask(task=task, logger=self.logger)
            executor.execute(allow_unlisted_languages=unlisted_language)
            self.print_info(u"Executing task... done")
        except Exception as exc:
            self.print_error(u"An unexpected Exception occurred while executing the task:")
            self.print_error(u"%s" % exc)
            return self.ERROR_EXIT_CODE

        try:
            self.print_info(u"Creating output sync map file...")
            path = task.output_sync_map_file()
            self.print_info(u"Creating output sync map file... done")
            self.print_info(u"Created %s" % path)
        except Exception as exc:
            self.print_error(u"An unexpected Exception occurred while writing the sync map file:")
            self.print_error(u"%s" % exc)
            return self.ERROR_EXIT_CODE

        if output_html:
            try:
                parameters = {}
                parameters[gc.PPN_TASK_OS_FILE_FORMAT] = task.configuration.os_file_format
                parameters[gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF] = task.configuration.os_file_smil_page_ref
                parameters[gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF] = task.configuration.os_file_smil_audio_ref
                self.print_info(u"Creating output HTML file...")
                task.sync_map.output_html_for_tuning(audio_file_path, html_file_path, parameters)
                self.print_info(u"Creating output HTML file... done")
                self.print_info(u"Created %s" % html_file_path)
            except Exception as exc:
                self.print_error(u"An unexpected Exception occurred while writing the HTML file:")
                self.print_error(u"%s" % exc)
                return self.ERROR_EXIT_CODE

        if download_from_youtube:
            if keep_audio:
                self.print_info(u"Option --keep-audio set: keeping downloaded file '%s'" % audio_file_path)
            else:
                gf.delete_file(None, audio_file_path)

        return self.NO_ERROR_EXIT_CODE



    def print_examples(self):
        """
        Print the examples and exit.
        """
        msg = []
        i = 1
        for key in sorted(self.DEMOS.keys()):
            example = self.DEMOS[key]
            msg.append(u"Example %d (%s)" % (i, example[u"description"]))
            msg.append(u"  $ CONFIG_STRING=\"%s\"" % (example[u"config"]))
            msg.append(u"  $ python -m aeneas.tools.%s %s %s \"$CONFIG_STRING\" %s %s" % (
                self.NAME,
                example[u"audio"],
                example[u"text"],
                example[u"syncmap"],
                example[u"options"]
            ))
            msg.append(u"  or")
            msg.append(u"  $ python -m aeneas.tools.%s %s" % (self.NAME, key))
            msg.append(u"")
            i += 1
        self.print_generic(u"\n" + u"\n".join(msg) + u"\n")
        return self.HELP_EXIT_CODE

    def print_parameters(self):
        """
        Print the list of parameters and exit.
        """
        self.print_info(u"You can use --list-values=PARAM on parameters marked by (*)")
        self.print_info(u"Available parameters:")
        self.print_generic(u"\n" + u"\n".join(self.PARAMETERS) + u"\n")
        return self.HELP_EXIT_CODE

    def print_values(self, parameter):
        """
        Print the list of values for the given parameter and exit.

        If ``parameter`` is invalid, print the list of
        parameter names that have allowed values.

        :param parameter: the parameter name
        :type  parameter: Unicode string
        """
        if parameter in self.VALUES:
            self.print_info(u"Available values for parameter '%s':" % parameter)
            self.print_generic(u", ".join(self.VALUES[parameter].ALLOWED_VALUES))
            return self.HELP_EXIT_CODE
        if parameter not in [u"?", u""]:
            self.print_error(u"Invalid parameter name '%s'" % parameter)
        self.print_info(u"The values for the following parameters can be enumerated:")
        self.print_info(u", ".join(sorted(self.VALUES.keys())))
        return self.HELP_EXIT_CODE



def main():
    """
    Execute program.
    """
    ExecuteTaskCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()



