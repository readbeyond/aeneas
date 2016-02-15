#!/usr/bin/env python
# coding=utf-8

"""
The runtime configuration object.

.. versionadded:: 1.4.1
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.configuration import Configuration
import aeneas.globalconstants as gc

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.5.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class RuntimeConfiguration(Configuration):
    """
    A structure representing a runtime configuration, that is,
    a set of parameters for the algorithms which process Jobs and Tasks.

    Allowed keys:

    * :attr:`aeneas.globalconstants.RC_ALLOW_UNLISTED_LANGUAGES`: bool, default: ``False``
    * :attr:`aeneas.globalconstants.RC_C_EXTENSIONS`: bool, default: ``True``
    * :attr:`aeneas.globalconstants.RC_DTW_ALGORITHM`: string, default: ``stripe``
    * :attr:`aeneas.globalconstants.RC_DTW_MARGIN`: float, default: ``60.0``
    * :attr:`aeneas.globalconstants.RC_ESPEAK_PATH`: string, default: ``espeak``
    * :attr:`aeneas.globalconstants.RC_FFMPEG_PATH`: string, default: ``ffmpeg``
    * :attr:`aeneas.globalconstants.RC_FFMPEG_SAMPLE_RATE`: int, default: ``16000``
    * :attr:`aeneas.globalconstants.RC_FFPROBE_PATH`: string, default: ``ffprobe``
    * :attr:`aeneas.globalconstants.RC_JOB_MAX_TASKS`: int, default: ``0``
    * :attr:`aeneas.globalconstants.RC_MFCC_FILTERS`: int, default: ``40``
    * :attr:`aeneas.globalconstants.RC_MFCC_SIZE`: int, default: ``13``
    * :attr:`aeneas.globalconstants.RC_MFCC_FFT_ORDER`: int, default: ``512``
    * :attr:`aeneas.globalconstants.RC_MFCC_LOWER_FREQUENCY`: float, default: ``133.3333``
    * :attr:`aeneas.globalconstants.RC_MFCC_UPPER_FREQUENCY`: float, default: ``6855.4976``
    * :attr:`aeneas.globalconstants.RC_MFCC_EMPHASIS_FACTOR`: float, default: ``0.970``
    * :attr:`aeneas.globalconstants.RC_MFCC_WINDOW_LENGTH`: float, default: ``0.100``
    * :attr:`aeneas.globalconstants.RC_MFCC_WINDOW_SHIFT`: float, default: ``0.040``
    * :attr:`aeneas.globalconstants.RC_TASK_MAX_AUDIO_LENGTH`: float, default: ``7200.0``
    * :attr:`aeneas.globalconstants.RC_TASK_MAX_TEXT_LENGTH`: int, default: ``0``
    * :attr:`aeneas.globalconstants.RC_TMP_PATH`: string, default: ``None``
    * :attr:`aeneas.globalconstants.RC_VAD_EXTEND_SPEECH_INTERVAL_AFTER`: float, default: ``0.0``
    * :attr:`aeneas.globalconstants.RC_VAD_EXTEND_SPEECH_INTERVAL_BEFORE`: float, default: ``0.0``
    * :attr:`aeneas.globalconstants.RC_VAD_LOG_ENERGY_THRESHOLD`: float, default: ``0.699``
    * :attr:`aeneas.globalconstants.RC_VAD_MIN_NONSPEECH_LENGTH`: float, default: ``0.200``

    :param string config_string: the configuration string

    :raises TypeError: if ``config_string`` is not ``None`` and
                       it is not a Unicode string
    :raises KeyError: if trying to access a key not listed above
    """

    TAG = u"RuntimeConfiguration"

    FIELDS = [
        (gc.RC_ALLOW_UNLISTED_LANGUAGES, (False, bool, ["allow_unlisted_languages"])),

        (gc.RC_C_EXTENSIONS, (True, bool, ["c_ext"])),

        (gc.RC_CEW_SUBPROCESS_ENABLED, (False, bool, ["cew_subprocess_enabled"])),
        #(gc.RC_CEW_SUBPROCESS_PATH, ("/usr/bin/python", None, ["cew_subprocess_path"])),
        (gc.RC_CEW_SUBPROCESS_PATH, ("python", None, ["cew_subprocess_path"])),
        
        (gc.RC_DTW_ALGORITHM, ("stripe", None, ["dtw_algorithm"])),
        (gc.RC_DTW_MARGIN, (60.0, float, ["dtw_margin"])),

        #(gc.RC_ESPEAK_PATH, ("/usr/bin/espeak", None, ["espeak_path"])),
        (gc.RC_ESPEAK_PATH, ("espeak", None, ["espeak_path"])),

        #(gc.RC_FFMPEG_PATH, ("/usr/bin/ffmpeg", None, ["ffmpeg_path"])),
        (gc.RC_FFMPEG_PATH, ("ffmpeg", None, ["ffmpeg_path"])),
        (gc.RC_FFMPEG_SAMPLE_RATE, (16000, int, ["ffmpeg_sr"])),

        #(gc.RC_FFMPEG_PATH, ("/usr/bin/ffprobe", None, ["ffprobe_path"])),
        (gc.RC_FFPROBE_PATH, ("ffprobe", None, ["ffprobe_path"])),

        (gc.RC_JOB_MAX_TASKS, (0, int, ["job_max_tasks"])),

        (gc.RC_MFCC_FILTERS, (40, int, ["mfcc_filters"])),
        (gc.RC_MFCC_SIZE, (13, int, ["mfcc_size"])),
        (gc.RC_MFCC_FFT_ORDER, (512, int, ["mfcc_order"])),
        (gc.RC_MFCC_LOWER_FREQUENCY, (133.3333, float, ["mfcc_lower_freq"])),
        (gc.RC_MFCC_UPPER_FREQUENCY, (6855.4976, float, ["mfcc_upper_freq"])),
        (gc.RC_MFCC_EMPHASIS_FACTOR, (0.970, float, ["mfcc_emph"])),
        (gc.RC_MFCC_WINDOW_LENGTH, (0.100, float, ["mfcc_win_len"])),
        (gc.RC_MFCC_WINDOW_SHIFT, (0.040, float, ["mfcc_win_shift"])),

        (gc.RC_TASK_MAX_AUDIO_LENGTH, (7200.0, float, ["task_max_a_len"])),
        (gc.RC_TASK_MAX_TEXT_LENGTH, (0, int, ["task_max_t_len"])),

        (gc.RC_TMP_PATH, (None, None, ["tmp_path"])),

        (gc.RC_VAD_EXTEND_SPEECH_INTERVAL_AFTER, (0.0, float, ["vad_extend_s_after"])),
        (gc.RC_VAD_EXTEND_SPEECH_INTERVAL_BEFORE, (0.0, float, ["vad_extend_s_before"])),
        (gc.RC_VAD_LOG_ENERGY_THRESHOLD, (0.699, float, ["vad_log_energy_thr"])),
        (gc.RC_VAD_MIN_NONSPEECH_LENGTH, (0.200, float, ["vad_min_ns_len"])),
    ]

    def __init__(self, config_string=None):
        super(RuntimeConfiguration, self).__init__(config_string)



