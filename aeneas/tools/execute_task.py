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
from aeneas.espeakwrapper import ESPEAKWrapper
from aeneas.executetask import ExecuteTask
from aeneas.festivalwrapper import FESTIVALWrapper
from aeneas.idsortingalgorithm import IDSortingAlgorithm
from aeneas.language import Language
from aeneas.nuancettsapiwrapper import NuanceTTSAPIWrapper
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.syncmap import SyncMapFormat
from aeneas.syncmap import SyncMapHeadTailFormat
from aeneas.task import Task
from aeneas.textfile import TextFileFormat
from aeneas.timevalue import Decimal
from aeneas.timevalue import TimeValue
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
__version__ = "1.5.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class ExecuteTaskCLI(AbstractCLIProgram):
    """
    Execute a Task, that is, a pair of audio/text files
    and a configuration string.
    """

    AUDIO_FILE = gf.relative_path("res/audio.mp3", __file__)
    CTW_ESPEAK = gf.relative_path("../extra/ctw_espeak.py", __file__)
    CTW_SPEECT = gf.relative_path("../extra/ctw_speect/ctw_speect.py", __file__)

    DEMOS = {
        u"--example-aftercurrent" : {
            u"description": u"input: plain text (plain), output: AUD, aba beforenext 0.200",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=plain|os_task_file_format=aud|task_adjust_boundary_algorithm=aftercurrent|task_adjust_boundary_aftercurrent_value=0.200",
            u"syncmap": "output/sonnet.aftercurrent0200.aud",
            u"options": u"",
            u"show": False
        },
        u"--example-beforenext" : {
            u"description": u"input: plain text (plain), output: AUD, aba beforenext 0.200",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=plain|os_task_file_format=aud|task_adjust_boundary_algorithm=beforenext|task_adjust_boundary_beforenext_value=0.200",
            u"syncmap": "output/sonnet.beforenext0200.aud",
            u"options": u"",
            u"show": False
        },
        u"--example-cewsubprocess" : {
            u"description": u"input: plain text, output: TSV, run via cewsubprocess",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=plain|os_task_file_format=tsv",
            u"syncmap": "output/sonnet.cewsubprocess.tsv",
            u"options": u"-r=\"cew_subprocess_enabled=True\"",
            u"show": False
        },
        u"--example-ctw-espeak" : {
            u"description": u"input: plain text, output: TSV, tts engine: ctw espeak",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=plain|os_task_file_format=tsv",
            u"syncmap": "output/sonnet.ctw_espeak.tsv",
            u"options": u"-r=\"tts=custom|tts_path=%s\"" % CTW_ESPEAK,
            u"show": False
        },
        u"--example-ctw-speect" : {
            u"description": u"input: plain text, output: TSV, tts engine: ctw speect",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=plain|os_task_file_format=tsv",
            u"syncmap": "output/sonnet.ctw_speect.tsv",
            u"options": u"-r=\"tts=custom|tts_path=%s\"" % CTW_SPEECT,
            u"show": False
        },
        u"--example-eaf" : {
            u"description": u"input: plain text, output: EAF",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=plain|os_task_file_format=eaf",
            u"syncmap": "output/sonnet.eaf",
            u"options": u"",
            u"show": True
        },
        u"--example-faster-rate" : {
            u"description": u"input: plain text (plain), output: SRT, print faster than 14.0 chars/s",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=plain|os_task_file_format=srt|task_adjust_boundary_algorithm=rate|task_adjust_boundary_rate_value=14.000",
            u"syncmap": "output/sonnet.faster.srt",
            u"options": u"--faster-rate",
            u"show": False
        },
        u"--example-festival" : {
            u"description": u"input: plain text, output: TSV, tts engine: Festival",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng-GBR|is_text_type=plain|os_task_file_format=tsv",
            u"syncmap": "output/sonnet.festival.tsv",
            u"options": u"-r=\"tts=festival|tts_path=text2wave\"",
            u"show": False
        },
        u"--example-flatten-12" : {
            u"description": u"input: mplain text (multilevel), output: JSON, levels to output: 1 and 2",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/mplain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=mplain|os_task_file_format=json|os_task_file_levels=12",
            u"syncmap": "output/sonnet.flatten12.json",
            u"options": u"",
            u"show": False
        },
        u"--example-flatten-2" : {
            u"description": u"input: mplain text (multilevel), output: JSON, levels to output: 2",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/mplain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=mplain|os_task_file_format=json|os_task_file_levels=2",
            u"syncmap": "output/sonnet.flatten2.json",
            u"options": u"",
            u"show": False
        },
        u"--example-flatten-3" : {
            u"description": u"input: mplain text (multilevel), output: JSON, levels to output: 3",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/mplain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=mplain|os_task_file_format=json|os_task_file_levels=3",
            u"syncmap": "output/sonnet.flatten3.json",
            u"options": u"",
            u"show": False
        },
        u"--example-head-tail" : {
            u"description": u"input: plain text, output: TSV, explicit head and tail",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=plain|os_task_file_format=tsv|is_audio_file_head_length=0.400|is_audio_file_tail_length=0.500|os_task_file_head_tail_format=stretch",
            u"syncmap": "output/sonnet.headtail.tsv",
            u"options": u"",
            u"show": False
        },
        u"--example-json" : {
            u"description": u"input: plain text, output: JSON",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=plain|os_task_file_format=json",
            u"syncmap": "output/sonnet.json",
            u"options": u"",
            u"show": True
        },
        u"--example-mplain-json" : {
            u"description": u"input: multilevel plain text (mplain), output: JSON",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/mplain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=mplain|os_task_file_format=json",
            u"syncmap": "output/sonnet.mplain.json",
            u"options": u"",
            u"show": False
        },
        u"--example-mplain-smil" : {
            u"description": u"input: multilevel plain text (mplain), output: SMIL",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/mplain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=mplain|os_task_file_format=smil|os_task_file_smil_audio_ref=p001.mp3|os_task_file_smil_page_ref=p001.xhtml",
            u"syncmap": "output/sonnet.mplain.smil",
            u"options": u"",
            u"show": True
        },
        u"--example-munparsed-json" : {
            u"description": u"input: multilevel unparsed text (munparsed), output: JSON",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/munparsed2.xhtml", __file__),
            u"config": u"task_language=eng|is_text_type=munparsed|is_text_munparsed_l1_id_regex=p[0-9]+|is_text_munparsed_l2_id_regex=s[0-9]+|is_text_munparsed_l3_id_regex=w[0-9]+|os_task_file_format=json",
            u"syncmap": "output/sonnet.munparsed.json",
            u"options": u"",
            u"show": False
        },
        u"--example-munparsed-smil" : {
            u"description": u"input: multilevel unparsed text (munparsed), output: SMIL",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/munparsed.xhtml", __file__),
            u"config": u"task_language=eng|is_text_type=munparsed|is_text_munparsed_l1_id_regex=p[0-9]+|is_text_munparsed_l2_id_regex=p[0-9]+s[0-9]+|is_text_munparsed_l3_id_regex=p[0-9]+s[0-9]+w[0-9]+|os_task_file_format=smil|os_task_file_smil_audio_ref=p001.mp3|os_task_file_smil_page_ref=p001.xhtml",
            u"syncmap": "output/sonnet.munparsed.smil",
            u"options": u"",
            u"show": True
        },
        u"--example-mws" : {
            u"description": u"input: plain text, output: JSON, resolution: 0.500s",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=plain|os_task_file_format=json",
            u"syncmap": "output/sonnet.mws.json",
            u"options": u"-r=\"mfcc_window_length=1.500|mfcc_window_shift=0.500\"",
            u"show": False
        },
        u"--example-no-zero" : {
            u"description": u"input: multilevel plain text (mplain), output: JSON, no zero duration",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/mplain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=mplain|os_task_file_format=json|os_task_file_no_zero=True",
            u"syncmap": "output/sonnet.nozero.json",
            u"options": u"--zero",
            u"show": False
        },
        u"--example-offset" : {
            u"description": u"input: plain text (plain), output: AUD, aba offset 0.200",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=plain|os_task_file_format=aud|task_adjust_boundary_algorithm=offset|task_adjust_boundary_offset_value=0.200",
            u"syncmap": "output/sonnet.offset0200.aud",
            u"options": u"",
            u"show": False
        },
        u"--example-percent" : {
            u"description": u"input: plain text (plain), output: AUD, aba percent 50",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=plain|os_task_file_format=aud|task_adjust_boundary_algorithm=percent|task_adjust_boundary_percent_value=50",
            u"syncmap": "output/sonnet.percent50.aud",
            u"options": u"",
            u"show": False
        },
        u"--example-py" : {
            u"description": u"input: plain text, output: JSON, pure python",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=plain|os_task_file_format=json",
            u"syncmap": "output/sonnet.cext.json",
            u"options": u"-r=\"c_extensions=False\"",
            u"show": False
        },
        u"--example-rates" : {
            u"description": u"input: plain text (plain), output: SRT, max rate 14.0 chars/s, print rates",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=plain|os_task_file_format=srt|task_adjust_boundary_algorithm=rate|task_adjust_boundary_rate_value=14.000",
            u"syncmap": "output/sonnet.rates.srt",
            u"options": u"--rates",
            u"show": False
        },
        u"--example-sd" : {
            u"description": u"input: plain text, output: TSV, head/tail detection",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=plain|os_task_file_format=tsv|is_audio_file_detect_head_max=10.000|is_audio_file_detect_tail_max=10.000",
            u"syncmap": "output/sonnet.sd.tsv",
            u"options": u"",
            u"show": False
        },
        u"--example-srt" : {
            u"description": u"input: subtitles text, output: SRT",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/subtitles.txt", __file__),
            u"config": u"task_language=eng|is_text_type=subtitles|os_task_file_format=srt",
            u"syncmap": "output/sonnet.srt",
            u"options": u"",
            u"show": True
        },
        u"--example-smil" : {
            u"description": u"input: unparsed text, output: SMIL",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/page.xhtml", __file__),
            u"config": u"task_language=eng|is_text_type=unparsed|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric|os_task_file_format=smil|os_task_file_smil_audio_ref=p001.mp3|os_task_file_smil_page_ref=p001.xhtml",
            u"syncmap": "output/sonnet.smil",
            u"options": u"",
            u"show": True
        },
        u"--example-tsv" : {
            u"description": u"input: parsed text, output: TSV",
            u"audio": AUDIO_FILE,
            u"text": gf.relative_path("res/parsed.txt", __file__),
            u"config": u"task_language=eng|is_text_type=parsed|os_task_file_format=tsv",
            u"syncmap": "output/sonnet.tsv",
            u"options": u"",
            u"show": True
        },
        u"--example-youtube" : {
            u"description": u"input: audio from YouTube, output: TXT",
            u"audio": "https://www.youtube.com/watch?v=rU4a7AA8wM0",
            u"text": gf.relative_path("res/plain.txt", __file__),
            u"config": u"task_language=eng|is_text_type=plain|os_task_file_format=txt",
            u"syncmap": "output/sonnet.txt",
            u"options": u"-y",
            u"show": True
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
        u"  is_text_mplain_word_separator           : word separator (mplain)",
        u"  is_text_munparsed_l1_id_regex           : regex matching level 1 id attributes (munparsed)",
        u"  is_text_munparsed_l2_id_regex           : regex matching level 2 id attributes (munparsed)",
        u"  is_text_munparsed_l3_id_regex           : regex matching level 3 id attributes (munparsed)",
        u"  is_text_unparsed_class_regex            : regex matching class attributes (unparsed)",
        u"  is_text_unparsed_id_regex               : regex matching id attributes (unparsed)",
        u"  is_text_unparsed_id_sort                : sort matched elements by id (unparsed) (*)",
        u"  is_text_file_ignore_regex               : ignore text matched by regex for audio alignment purposes",
        u"  is_text_file_transliterate_map          : apply the given transliteration map for audio alignment purposes",
        u"",
        u"  os_task_file_format                     : output sync map format (*)",
        u"  os_task_file_id_regex                   : id regex for the output sync map (subtitles, plain)",
        u"  os_task_file_head_tail_format           : format audio head/tail (*)",
        u"  os_task_file_levels                     : output the specified levels (mplain)",
        u"  os_task_file_no_zero                    : if True, do not allow zero-length fragments",
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
        "espeak" : sorted(ESPEAKWrapper.LANGUAGE_TO_VOICE_CODE.keys()),
        "festival" : sorted(FESTIVALWrapper.LANGUAGE_TO_VOICE_CODE.keys()),
        "nuancettsapi": sorted(NuanceTTSAPIWrapper.LANGUAGE_TO_VOICE_CODE.keys()),
        "task_language" : Language.ALLOWED_VALUES,
        "is_text_type" : TextFileFormat.ALLOWED_VALUES,
        "is_text_unparsed_id_sort" : IDSortingAlgorithm.ALLOWED_VALUES,
        "os_task_file_format" : SyncMapFormat.ALLOWED_VALUES,
        "os_task_file_head_tail_format" : SyncMapHeadTailFormat.ALLOWED_VALUES,
        "task_adjust_boundary_algorithm" : AdjustBoundaryAlgorithm.ALLOWED_VALUES,
    }

    NAME = gf.file_name_without_extension(__file__)

    HELP = {
        "description": u"Execute a Task.",
        "synopsis": [
            (u"--list-parameters", False),
            (u"--list-values[=PARAM]", False),
            (u"AUDIO_FILE  TEXT_FILE CONFIG_STRING OUTPUT_FILE", True),
            (u"YOUTUBE_URL TEXT_FILE CONFIG_STRING OUTPUT_FILE -y", True),
        ],
        "examples": [
            u"--examples",
            u"--examples-all"
        ],
        "options": [
            u"--faster-rate : print fragments with rate > task_adjust_boundary_rate_value",
            u"--keep-audio : do not delete the audio file downloaded from YouTube (-y only)",
            u"--largest-audio : download largest audio stream (-y only)",
            u"--list-parameters : list all parameters",
            u"--list-values : list all parameters for which values can be listed",
            u"--list-values=PARAM : list all allowed values for parameter PARAM",
            u"--output-html : output HTML file for fine tuning",
            u"--rates : print rate of each fragment",
            u"--skip-validator : do not validate the given config string",
            u"--zero : print fragments with zero duration",
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
            return self.print_examples(False)

        if self.has_option(u"--examples-all"):
            return self.print_examples(True)

        if self.has_option([u"--list-parameters"]):
            return self.print_parameters()

        parameter = self.has_option_with_value(u"--list-values")
        if parameter is not None:
            return self.print_values(parameter)
        elif self.has_option(u"--list-values"):
            return self.print_values(u"?")

        # NOTE list() is needed for Python3, where keys() is not a list!
        demo = self.has_option(list(self.DEMOS.keys()))
        demo_parameters = u""
        download_from_youtube = self.has_option([u"-y", u"--youtube"])
        largest_audio = self.has_option(u"--largest-audio")
        keep_audio = self.has_option(u"--keep-audio")
        output_html = self.has_option(u"--output-html")
        validate = not self.has_option(u"--skip-validator")
        print_faster_rate = self.has_option(u"--faster-rate")
        print_rates = self.has_option(u"--rates")
        print_zero = self.has_option(u"--zero")

        if demo:
            validate = False
            for key in self.DEMOS:
                if self.has_option(key):
                    demo_parameters = self.DEMOS[key]
                    audio_file_path = demo_parameters[u"audio"]
                    text_file_path = demo_parameters[u"text"]
                    config_string = demo_parameters[u"config"]
                    sync_map_file_path = demo_parameters[u"syncmap"]
                    # TODO allow injecting rconf options directly from DEMOS options field
                    if key == u"--example-cewsubprocess":
                        self.rconf[RuntimeConfiguration.CEW_SUBPROCESS_ENABLED] = True
                    elif key == u"--example-ctw-espeak":
                        self.rconf[RuntimeConfiguration.TTS] = "custom"
                        self.rconf[RuntimeConfiguration.TTS_PATH] = self.CTW_ESPEAK
                    elif key == u"--example-ctw-speect":
                        self.rconf[RuntimeConfiguration.TTS] = "custom"
                        self.rconf[RuntimeConfiguration.TTS_PATH] = self.CTW_SPEECT
                    elif key == u"--example-festival":
                        self.rconf[RuntimeConfiguration.TTS] = "festival"
                        self.rconf[RuntimeConfiguration.TTS_PATH] = "text2wave"
                    elif key == u"--example-mws":
                        self.rconf[RuntimeConfiguration.MFCC_WINDOW_LENGTH] = "1.500"
                        self.rconf[RuntimeConfiguration.MFCC_WINDOW_SHIFT] = "0.500"
                    elif key == u"--example-faster-rate":
                        print_faster_rate = True
                    elif key == u"--example-no-zero":
                        print_zero = True
                    elif key == u"--example-py":
                        self.rconf[RuntimeConfiguration.C_EXTENSIONS] = False
                    elif key == u"--example-rates":
                        print_rates = True
                    elif key == u"--example-youtube":
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
            if len(demo_parameters[u"options"]) > 0:
                msg.append(u"  Options:       %s" % demo_parameters[u"options"])
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
                self.print_no_pafy_error()
                return self.ERROR_EXIT_CODE
            except Exception as exc:
                self.print_error(u"An unexpected error occurred while downloading audio from YouTube:")
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
            self.print_error(u"An unexpected error occurred while creating the task:")
            self.print_error(u"%s" % exc)
            return self.ERROR_EXIT_CODE

        try:
            self.print_info(u"Executing task...")
            executor = ExecuteTask(task=task, rconf=self.rconf, logger=self.logger)
            executor.execute()
            self.print_info(u"Executing task... done")
        except Exception as exc:
            self.print_error(u"An unexpected error occurred while executing the task:")
            self.print_error(u"%s" % exc)
            return self.ERROR_EXIT_CODE

        try:
            self.print_info(u"Creating output sync map file...")
            path = task.output_sync_map_file()
            self.print_info(u"Creating output sync map file... done")
            self.print_success(u"Created file '%s'" % path)
        except Exception as exc:
            self.print_error(u"An unexpected error occurred while writing the sync map file:")
            self.print_error(u"%s" % exc)
            return self.ERROR_EXIT_CODE

        if output_html:
            try:
                parameters = {}
                parameters[gc.PPN_TASK_OS_FILE_FORMAT] = task.configuration["o_format"]
                parameters[gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF] = task.configuration["o_smil_audio_ref"]
                parameters[gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF] = task.configuration["o_smil_page_ref"]
                self.print_info(u"Creating output HTML file...")
                task.sync_map.output_html_for_tuning(audio_file_path, html_file_path, parameters)
                self.print_info(u"Creating output HTML file... done")
                self.print_success(u"Created file '%s'" % html_file_path)
            except Exception as exc:
                self.print_error(u"An unexpected error occurred while writing the HTML file:")
                self.print_error(u"%s" % exc)
                return self.ERROR_EXIT_CODE

        if download_from_youtube:
            if keep_audio:
                self.print_info(u"Option --keep-audio set: keeping downloaded file '%s'" % audio_file_path)
            else:
                gf.delete_file(None, audio_file_path)

        if print_zero:
            zero_duration = [l for l in task.sync_map.fragments_tree.vleaves_not_empty if l.begin == l.end]
            if len(zero_duration) > 0:
                self.print_warning(u"Fragments with zero duration:")
                for fragment in zero_duration:
                    self.print_generic(u"  %s" % fragment)

        if print_rates:
            self.print_info(u"Fragments with rates:")
            for fragment in task.sync_map.fragments_tree.vleaves_not_empty:
                self.print_generic(u"  %s (rate: %.3f chars/s)" % (fragment, fragment.rate))

        if print_faster_rate:
            max_rate = task.configuration["aba_rate_value"]
            if max_rate is not None:
                faster = [l for l in task.sync_map.fragments_tree.vleaves_not_empty if l.rate >= max_rate + Decimal("0.001")]
                if len(faster) > 0:
                    self.print_warning(u"Fragments with rate greater than %.3f:" % max_rate)
                    for fragment in faster:
                        self.print_generic(u"  %s (rate: %.3f chars/s)" % (fragment, fragment.rate))

        return self.NO_ERROR_EXIT_CODE

    def print_examples(self, full=False):
        """
        Print the examples and exit.

        :param bool full: if ``True``, print all examples; otherwise,
                          print only selected ones
        """
        msg = []
        i = 1
        for key in sorted(self.DEMOS.keys()):
            example = self.DEMOS[key]
            if full or example["show"]:
                msg.append(u"Example %d (%s)" % (i, example[u"description"]))
                # NOTE too verbose now that we have dozens of examples
                #msg.append(u"  $ CONFIG_STRING=\"%s\"" % (example[u"config"]))
                #msg.append(u"  $ python -m aeneas.tools.%s %s %s \"$CONFIG_STRING\" %s %s" % (
                #    self.NAME,
                #    example[u"audio"],
                #    example[u"text"],
                #    example[u"syncmap"],
                #    example[u"options"]
                #))
                #msg.append(u"  or")
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
            self.print_generic(u", ".join(self.VALUES[parameter]))
            return self.HELP_EXIT_CODE
        if parameter not in [u"?", u""]:
            self.print_error(u"Invalid parameter name '%s'" % parameter)
        self.print_info(u"Parameters for which values can be listed:")
        self.print_generic(u", ".join(sorted(self.VALUES.keys())))
        return self.HELP_EXIT_CODE



def main():
    """
    Execute program.
    """
    ExecuteTaskCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()



