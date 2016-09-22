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
This module contains the following classes:

* :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`,
  representing the runtime configuration.

.. versionadded:: 1.4.1
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.configuration import Configuration
from aeneas.timevalue import TimeValue


class RuntimeConfiguration(Configuration):
    """
    A structure representing a runtime configuration, that is,
    a set of parameters for the algorithms which process jobs and tasks.

    Allowed keys are listed below as class members.

    :param string config_string: the configuration string
    :raises: TypeError: if ``config_string`` is not ``None`` and
                        it is not a Unicode string
    :raises: KeyError: if trying to access a key not listed above
    """

    ALLOW_UNLISTED_LANGUAGES = "allow_unlisted_languages"
    """
    If ``True``, allow using a language code
    not listed in the TTS supported languages list;
    otherwise, generate an error if the user attempts
    to use a language not listed.

    Default: ``False``.

    .. versionadded:: 1.4.1
    """

    C_EXTENSIONS = "c_extensions"
    """
    If ``True`` and the Python C/C++ extensions
    are available, use them.
    Otherwise, use the pure Python code.

    This option is equivalent to
    setting ``CDTW``, ``CEW``, ``CFW``,
    and ``CMFCC`` to ``True`` or ``False`` at once.

    Default: ``True``.

    .. versionadded:: 1.4.1
    """

    CDTW = "cdtw"
    """
    If ``True`` and the Python C extension ``cdtw``
    is available, use it.
    Otherwise, use the pure Python code.

    Default: ``True``.

    .. versionadded:: 1.5.1
    """

    CEW = "cew"
    """
    If ``True`` and the Python C extension ``cew``
    is available, use it.
    Otherwise, use the pure Python code.

    Default: ``True``.

    .. versionadded:: 1.5.1
    """

    CFW = "cfw"
    """
    If ``True`` and the Python C++ extension ``cfw``
    is available, use it.
    Otherwise, use the pure Python code.

    Default: ``True``.

    .. versionadded:: 1.6.0
    """

    CEW_SUBPROCESS_ENABLED = "cew_subprocess_enabled"
    """
    If ``True``, calls to ``aeneas.cew``
    will be done via ``subprocess``, using the
    :class:`~aeneas.cewsubprocess.CEWSubprocess`
    helper class.

    Default: ``False``.

    .. versionadded:: 1.5.0
    """

    CEW_SUBPROCESS_PATH = "cew_subprocess_path"
    """
    Use the given path to the python executable
    when calling ``aeneas.cew`` via ``subprocess``.

    You might need to use a full path, like ``/path/to/your/python``.

    Default: ``python``.

    .. versionadded:: 1.5.0
    """

    CMFCC = "cmfcc"
    """
    If ``True`` and the Python C extension ``cmfcc``
    is available, use it.
    Otherwise, use the pure Python code.

    Default: ``True``.

    .. versionadded:: 1.5.1
    """

    DTW_ALGORITHM = "dtw_algorithm"
    """
    DTW aligner algorithm.

    Allowed values:

    * :data:`~aeneas.dtw.DTWAlgorithm.EXACT` (``exact``)
    * :data:`~aeneas.dtw.DTWAlgorithm.STRIPE` (``stripe``, default)

    .. versionadded:: 1.4.1
    """

    DTW_MARGIN = "dtw_margin"
    """
    DTW aligner margin, in seconds, for the ``stripe`` algorithm.

    Default: ``60``, corresponding to ``60s`` ahead and behind
    (i.e., ``120s`` total margin).

    .. versionadded:: 1.4.1
    """

    FFMPEG_PATH = "ffmpeg_path"
    """
    Path to the ``ffmpeg`` executable.

    You might need to use a full path, like ``/path/to/your/ffmpeg``.

    Default: ``ffmpeg``.

    .. versionadded:: 1.4.1
    """

    FFMPEG_SAMPLE_RATE = "ffmpeg_sample_rate"
    """
    Sample rate for ``ffmpeg``, in Hertz.

    Default: ``16000``.

    .. versionadded:: 1.4.1
    """

    FFPROBE_PATH = "ffprobe_path"
    """
    Path to the ``ffprobe`` executable.

    You might use a full path, like ``/path/to/your/ffprobe``.

    Default: ``ffprobe``.

    .. versionadded:: 1.4.1
    """

    JOB_MAX_TASKS = "job_max_tasks"
    """
    Maximum number of Tasks of a Job.
    If a Job has more Tasks than this value,
    it will not be executed and an error will be raised.
    Use ``0`` for disabling this check.

    Default: ``0`` (disabled).

    .. versionadded:: 1.4.1
    """

    MFCC_FILTERS = "mfcc_filters"
    """
    Number of filters for extracting MFCCs.

    Default: ``40``.

    .. versionadded:: 1.4.1
    """

    MFCC_SIZE = "mfcc_size"
    """
    Number of MFCCs to extract, including the 0th.

    Default: ``13``.

    .. versionadded:: 1.4.1
    """

    MFCC_FFT_ORDER = "mfcc_fft_order"
    """
    Order of the RFFT for extracting MFCCs.
    It must be a power of two.

    Default: ``512``.

    .. versionadded:: 1.4.1
    """

    MFCC_LOWER_FREQUENCY = "mfcc_lower_frequency"
    """
    Lower frequency to be used for extracting MFCCs, in Hertz.

    Default: ``133.3333``.

    .. versionadded:: 1.4.1
    """

    MFCC_UPPER_FREQUENCY = "mfcc_upper_frequency"
    """
    Upper frequency to be used for extracting MFCCs, in Hertz.

    Default: ``6855.4976``.

    .. versionadded:: 1.4.1
    """

    MFCC_EMPHASIS_FACTOR = "mfcc_emphasis_factor"
    """
    Emphasis factor to be applied to MFCCs.

    Default: ``0.970``.

    .. versionadded:: 1.4.1
    """

    MFCC_WINDOW_LENGTH = "mfcc_window_length"
    """
    Length of the window for extracting MFCCs, in seconds.
    It is usual to set it between 1.5 and 4 times
    the value of ``MFCC_WINDOW_SHIFT``.

    Default: ``0.100``.

    .. versionadded:: 1.4.1
    """

    MFCC_WINDOW_SHIFT = "mfcc_window_shift"
    """
    Shift of the window for extracting MFCCs, in seconds.
    This parameter is basically the time step
    of the synchronization maps output.

    Default: ``0.040``.

    .. versionadded:: 1.4.1
    """

    MFCC_WINDOW_LENGTH_L1 = "mfcc_window_length_l1"
    """
    Length of the window, in seconds,
    for extracting MFCCs at level 1 (paragraph).
    It is usual to set it between 1.5 and 4 times
    the value of ``MFCC_WINDOW_SHIFT_L1``.

    Default: ``0.500``.

    .. versionadded:: 1.5.0
    """

    MFCC_WINDOW_SHIFT_L1 = "mfcc_window_shift_l1"
    """
    Shift of the window, in seconds,
    for extracting MFCCs at level 1 (paragraph).
    This parameter is basically the time step
    of the synchronization map output at level 1.

    Default: ``0.200``.

    .. versionadded:: 1.5.0
    """

    MFCC_WINDOW_LENGTH_L2 = "mfcc_window_length_l2"
    """
    Length of the window, in seconds,
    for extracting MFCCs at level 2 (sentence).
    It is usual to set it between 1.5 and 4 times
    the value of ``MFCC_WINDOW_SHIFT_L2``.

    Default: ``0.100``.

    .. versionadded:: 1.5.0
    """

    MFCC_WINDOW_SHIFT_L2 = "mfcc_window_shift_l2"
    """
    Shift of the window, in seconds,
    for extracting MFCCs at level 2 (sentence).
    This parameter is basically the time step
    of the synchronization map output at level 2.

    Default: ``0.040``.

    .. versionadded:: 1.5.0
    """

    MFCC_WINDOW_LENGTH_L3 = "mfcc_window_length_l3"
    """
    Length of the window, in seconds,
    for extracting MFCCs at level 3 (word).
    It is usual to set it between 1.5 and 4 times
    the value of ``MFCC_WINDOW_SHIFT_L3``.

    Default: ``0.020``.

    .. versionadded:: 1.5.0
    """

    MFCC_WINDOW_SHIFT_L3 = "mfcc_window_shift_l3"
    """
    Shift of the window, in seconds,
    for extracting MFCCs at level 3 (word).
    This parameter is basically the time step
    of the synchronization map output at level 3.

    Default: ``0.005``.

    .. versionadded:: 1.5.0
    """

    MFCC_GRANULARITY_MAP = {
        1: (MFCC_WINDOW_LENGTH_L1, MFCC_WINDOW_SHIFT_L1),
        2: (MFCC_WINDOW_LENGTH_L2, MFCC_WINDOW_SHIFT_L2),
        3: (MFCC_WINDOW_LENGTH_L3, MFCC_WINDOW_SHIFT_L3),
    }
    """
    Map level numbers to ``MFCC_WINDOW_LENGTH_*``
    and ``MFCC_WINDOW_SHIFT_*`` keys.

    .. versionadded:: 1.5.0
    """

    NUANCE_TTS_API_ID = "nuance_tts_api_id"
    """
    Your ID value to use the Nuance TTS API.

    You will be billed according to your Nuance Developers account plan.

    Important: this feature is experimental, use at your own risk.
    It is recommended not to use this TTS at word-level granularity,
    as it will create many requests, hence it will be expensive.
    If you still want to use it, you can enable
    the TTS caching mechanism by setting
    :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS_CACHE`
    to ``True``.

    .. versionadded:: 1.5.0
    """

    NUANCE_TTS_API_KEY = "nuance_tts_api_key"
    """
    Your KEY value to use the Nuance TTS API.

    You will be billed according to your Nuance Developers account plan.

    Important: this feature is experimental, use at your own risk.
    It is recommended not to use this TTS at word-level granularity,
    as it will create many requests, hence it will be expensive.
    If you still want to use it, you can enable
    the TTS caching mechanism by setting
    :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS_CACHE`
    to ``True``.

    .. versionadded:: 1.5.0
    """

    NUANCE_TTS_API_SLEEP = "nuance_tts_api_sleep"
    """
    Wait this number of seconds before the next HTTP POST request
    to the Nuance TTS API.
    This parameter can be used to throttle the HTTP usage.
    It cannot be a negative value.

    Default: ``1.000``.

    .. versionadded:: 1.5.0
    """

    NUANCE_TTS_API_RETRY_ATTEMPTS = "nuance_tts_api_retry_attempts"
    """
    Retry an HTTP POST request to the Nuance TTS API
    for this number of times before giving up.
    It must be an integer greater than zero.

    Default: ``5``.

    .. versionadded:: 1.5.0
    """

    TASK_MAX_AUDIO_LENGTH = "task_max_audio_length"
    """
    Maximum length of the audio file of a Task, in seconds.
    If a Task has an audio file longer than this value,
    it will not be executed and an error will be raised.
    Use ``0`` for disabling this check.

    Default: ``7200`` seconds.

    .. versionadded:: 1.4.1
    """

    TASK_MAX_TEXT_LENGTH = "task_max_text_length"
    """
    Maximum number of text fragments in the text file of a Task.
    If a Task has more text fragments than this value,
    it will not be executed and an error will be raised.

    Use ``0`` for disabling this check.

    Default: ``0`` (disabled).

    .. versionadded:: 1.4.1
    """

    TMP_PATH = "tmp_path"
    """
    Path to the temporary directory to be used.
    Default: ``None``, meaning that the default temporary directory
    will be set by
    :data:`~aeneas.globalconstants.TMP_PATH_DEFAULT_POSIX`
    or
    :data:`~aeneas.globalconstants.TMP_PATH_DEFAULT_NONPOSIX`
    depending on your OS.

    .. versionadded:: 1.4.1
    """

    TTS = "tts"
    """
    The TTS engine to use for synthesizing text.

    Allowed values are listed in :data:`~aeneas.synthesizer.Synthesizer.ALLOWED_VALUES`.

    The default value is
    :data:`~aeneas.synthesizer.Synthesizer.ESPEAK` (``espeak``)
    which will use the built-in eSpeak TTS wrapper.
    You might need to provide a ``/full/path/to/your/espeak`` value
    to the
    :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS_PATH`
    parameter if the command ``espeak`` is not available in
    one of the directories listed in your ``PATH`` environment variable.

    Specify the value
    :data:`~aeneas.synthesizer.Synthesizer.ESPEAKNG` (``espeak-ng``)
    to use the eSpeak-ng TTS wrapper.
    You might need to provide a ``/full/path/to/your/espeak-ng`` value
    to the
    :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS_PATH`
    parameter if the command ``espeak-ng`` is not available in
    one of the directories listed in your ``PATH`` environment variable.

    Specify the value
    :data:`~aeneas.synthesizer.Synthesizer.FESTIVAL` (``festival``)
    to use the built-in Festival TTS wrapper.
    You might need to provide a ``/full/path/to/your/text2wave`` value
    to the
    :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS_PATH`
    parameter if the command ``text2wave`` is not available in
    one of the directories listed in your ``PATH`` environment variable.

    Specify the value
    :data:`~aeneas.synthesizer.Synthesizer.NUANCE` (``nuance``)
    to use the built-in Nuance TTS API wrapper;
    you will need to provide your Nuance Developer API ID and API Key using the
    :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.NUANCE_TTS_API_ID`
    and
    :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.NUANCE_TTS_API_KEY`
    parameters.
    Please note that you will be billed according to your Nuance Developers account plan.

    Specify the value
    :data:`~aeneas.synthesizer.Synthesizer.CUSTOM` (``custom``)
    to use a custom TTS;
    you will need to provide the path to the Python source file
    containing your TTS wrapper using the
    :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS_PATH`
    parameter.

    .. versionadded:: 1.5.0
    """

    TTS_CACHE = "tts_cache"
    """
    If set to ``True``, synthesize each distinct text fragment
    only once, caching the resulting audio data as a file on disk.

    The cache files will be removed after the synthesis is compled.

    This option is useful when calling TTS engines,
    via subprocess or remote APIs,
    on text files with many identical fragments,
    for example when aligning at word-level granularity.

    Enabling this option will create the cache files in
    :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TMP_PATH`,
    so make sure that that path has enough free space.

    Default: ``False``.

    .. versionadded:: 1.6.0
    """

    TTS_PATH = "tts_path"
    """
    Path to the TTS engine executable
    or the Python CustomTTSWrapper ``.py`` source file
    (see the ``aeneas/extra`` directory for examples).

    You might need to use a full path,
    like ``/path/to/your/ttsengine`` or
    ``/path/to/your/ttswrapper.py``.

    Default: ``None``, implying to use the default path
    defined by each TTS wrapper, if it calls the TTS engine
    via ``subprocess`` (otherwise it does not matter).

    .. versionadded:: 1.5.0
    """

    TTS_VOICE_CODE = "tts_voice_code"
    """
    The code of the TTS voice to use.
    If you specify this value, it will override the default voice code
    associated with the language of your text.

    Default: ``None``.

    .. versionadded:: 1.5.0
    """

    TTS_L1 = "tts_l1"
    """
    The TTS engine to use for synthesizing text
    at level 1 (paragraph).

    See also :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS`.

    Default: ``espeak``.

    .. versionadded:: 1.6.0
    """

    TTS_PATH_L1 = "tts_path_l1"
    """
    Path to the TTS engine executable to use for synthesizing text
    at level 1 (paragraph).

    See also :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS_PATH`.

    Default: ``None``.

    .. versionadded:: 1.6.0
    """

    TTS_L2 = "tts_l2"
    """
    The TTS engine to use for synthesizing text
    at level 2 (sentence).

    See also :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS`.

    Default: ``espeak``.

    .. versionadded:: 1.6.0
    """

    TTS_PATH_L2 = "tts_path_l2"
    """
    Path to the TTS engine executable to use for synthesizing text
    at level 2 (sentence).

    See also :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS_PATH`.

    Default: ``None``.

    .. versionadded:: 1.6.0
    """

    TTS_L3 = "tts_l3"
    """
    The TTS engine to use for synthesizing text
    at level 3 (word).

    See also :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS`.

    Default: ``espeak``.

    .. versionadded:: 1.6.0
    """

    TTS_PATH_L3 = "tts_path_l3"
    """
    Path to the TTS engine executable to use for synthesizing text
    at level 3 (word).

    See also :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS_PATH`.

    Default: ``None``.

    .. versionadded:: 1.6.0
    """

    TTS_GRANULARITY_MAP = {
        1: (TTS_L1, TTS_PATH_L1),
        2: (TTS_L2, TTS_PATH_L2),
        3: (TTS_L3, TTS_PATH_L3),
    }
    """
    Map level numbers to ``TTS_*``
    and ``TTS_PATH_*`` keys.

    .. versionadded:: 1.6.0
    """

    VAD_EXTEND_SPEECH_INTERVAL_AFTER = "vad_extend_speech_after"
    """
    Extend to the right (after/future)
    a speech interval found by the VAD algorithm,
    by this many seconds.

    Default: ``0`` seconds.

    .. versionadded:: 1.4.1
    """

    VAD_EXTEND_SPEECH_INTERVAL_BEFORE = "vad_extend_speech_before"
    """
    Extend to the left (before/past)
    a speech interval found by the VAD algorithm,
    by this many seconds.

    Default: ``0`` seconds.

    .. versionadded:: 1.4.1
    """

    VAD_LOG_ENERGY_THRESHOLD = "vad_log_energy_threshold"
    """
    Threshold for the VAD algorithm to decide
    that a given frame contains speech.
    Note that this is the log10 of the energy coefficient.

    Default: ``0.699`` = ``log10(5)``, that is, a frame must have
    an energy at least 5 times higher than the minimum
    to be considered a speech frame.

    .. versionadded:: 1.4.1
    """

    VAD_MIN_NONSPEECH_LENGTH = "vad_min_nonspeech_length"
    """
    Minimum length, in seconds, of a nonspeech interval.

    Default: ``0.200`` seconds.

    .. versionadded:: 1.4.1
    """

    # NOTE not using aliases just not to become confused
    #      about external (user rconf) and internal (lib code) key names
    #      although the functionality might be useful in the future
    FIELDS = [
        (ALLOW_UNLISTED_LANGUAGES, (False, bool, [])),

        (C_EXTENSIONS, (True, bool, [])),
        (CDTW, (True, bool, [])),
        (CEW, (True, bool, [])),
        (CFW, (True, bool, [])),
        (CMFCC, (True, bool, [])),

        (CEW_SUBPROCESS_ENABLED, (False, bool, [])),
        (CEW_SUBPROCESS_PATH, ("python", None, [])),    # or a full path like "/usr/bin/python"

        (DTW_ALGORITHM, ("stripe", None, [])),
        (DTW_MARGIN, ("60.000", TimeValue, [])),

        (FFMPEG_PATH, ("ffmpeg", None, [])),            # or a full path like "/usr/bin/ffmpeg"
        (FFMPEG_SAMPLE_RATE, (16000, int, [])),

        (FFPROBE_PATH, ("ffprobe", None, [])),          # or a full path like "/usr/bin/ffprobe"

        (JOB_MAX_TASKS, (0, int, [])),

        (MFCC_FILTERS, (40, int, [])),
        (MFCC_SIZE, (13, int, [])),
        (MFCC_FFT_ORDER, (512, int, [])),
        (MFCC_LOWER_FREQUENCY, (133.3333, float, [])),
        (MFCC_UPPER_FREQUENCY, (6855.4976, float, [])),
        (MFCC_EMPHASIS_FACTOR, (0.970, float, [])),
        (MFCC_WINDOW_LENGTH, ("0.100", TimeValue, [])),
        (MFCC_WINDOW_SHIFT, ("0.040", TimeValue, [])),

        (MFCC_WINDOW_LENGTH_L1, ("0.500", TimeValue, [])),
        (MFCC_WINDOW_SHIFT_L1, ("0.200", TimeValue, [])),
        (MFCC_WINDOW_LENGTH_L2, ("0.100", TimeValue, [])),
        (MFCC_WINDOW_SHIFT_L2, ("0.040", TimeValue, [])),
        (MFCC_WINDOW_LENGTH_L3, ("0.020", TimeValue, [])),
        (MFCC_WINDOW_SHIFT_L3, ("0.005", TimeValue, [])),

        (NUANCE_TTS_API_ID, (None, None, [])),
        (NUANCE_TTS_API_KEY, (None, None, [])),
        (NUANCE_TTS_API_SLEEP, ("1.000", TimeValue, [])),
        (NUANCE_TTS_API_RETRY_ATTEMPTS, (5, int, [])),

        (TASK_MAX_AUDIO_LENGTH, ("7200.0", TimeValue, [])),
        (TASK_MAX_TEXT_LENGTH, (0, int, [])),

        (TMP_PATH, (None, None, [])),

        (TTS, ("espeak", None, [])),
        (TTS_PATH, (None, None, [])),                   # None (= default) or "espeak" or "/usr/bin/espeak"
        (TTS_VOICE_CODE, (None, None, [])),
        (TTS_CACHE, (False, bool, [])),

        (TTS_L1, ("espeak", None, [])),
        (TTS_PATH_L1, (None, None, [])),                # None (= default) or "espeak" or "/usr/bin/espeak"
        (TTS_L2, ("espeak", None, [])),
        (TTS_PATH_L2, (None, None, [])),                # None (= default) or "espeak" or "/usr/bin/espeak"
        (TTS_L3, ("espeak", None, [])),
        (TTS_PATH_L3, (None, None, [])),                # None (= default) or "espeak" or "/usr/bin/espeak"

        (VAD_EXTEND_SPEECH_INTERVAL_AFTER, ("0.000", TimeValue, [])),
        (VAD_EXTEND_SPEECH_INTERVAL_BEFORE, ("0.000", TimeValue, [])),
        (VAD_LOG_ENERGY_THRESHOLD, (0.699, float, [])),
        (VAD_MIN_NONSPEECH_LENGTH, ("0.200", TimeValue, [])),
    ]

    TAG = u"RuntimeConfiguration"

    def __init__(self, config_string=None):
        super(RuntimeConfiguration, self).__init__(config_string)

    def clone(self):
        """
        Return a new configuration object
        that contains a copy of this configuration object.

        :rtype: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
        """
        new_rconf = RuntimeConfiguration()
        new_rconf.data = dict(self.data)
        new_rconf.types = dict(self.types)
        new_rconf.aliases = dict(self.aliases)
        return new_rconf

    @property
    def sample_rate(self):
        """
        Return the value of the
        :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.FFMPEG_SAMPLE_RATE`
        key stored in this configuration object.

        :rtype: int
        """
        return self[self.FFMPEG_SAMPLE_RATE]

    @property
    def mws(self):
        """
        Return the value of the
        :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.MFCC_WINDOW_SHIFT`
        key stored in this configuration object.

        :rtype: :class:`~aeneas.timevalue.TimeValue`
        """
        return self[self.MFCC_WINDOW_SHIFT]

    @property
    def mwl(self):
        """
        Return the value of the
        :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.MFCC_WINDOW_LENGTH`
        key stored in this configuration object.

        :rtype: :class:`~aeneas.timevalue.TimeValue`
        """
        return self[self.MFCC_WINDOW_LENGTH]

    @property
    def tts(self):
        """
        Return the value of the
        :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS`
        key stored in this configuration object.

        :rtype: string
        """
        return self[self.TTS]

    @property
    def tts_path(self):
        """
        Return the value of the
        :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS_PATH`
        key stored in this configuration object.

        :rtype: string
        """
        return self[self.TTS_PATH]

    def set_granularity(self, level):
        """
        Set the values for
        :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.MFCC_WINDOW_LENGTH`
        and
        :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.MFCC_WINDOW_SHIFT`
        matching the given granularity level.

        Currently supported levels:

        * ``1`` (paragraph)
        * ``2`` (sentence)
        * ``3`` (word)

        :param int level: the desired granularity level
        """
        if level in self.MFCC_GRANULARITY_MAP.keys():
            length_key, shift_key = self.MFCC_GRANULARITY_MAP[level]
            self[self.MFCC_WINDOW_LENGTH] = self[length_key]
            self[self.MFCC_WINDOW_SHIFT] = self[shift_key]

    def set_tts(self, level):
        """
        Set the values for
        :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS`
        and
        :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.TTS_PATH`
        matching the given granularity level.

        Currently supported levels:

        * ``1`` (paragraph)
        * ``2`` (sentence)
        * ``3`` (word)

        :param int level: the desired granularity level
        """
        if level in self.TTS_GRANULARITY_MAP.keys():
            tts_key, tts_path_key = self.TTS_GRANULARITY_MAP[level]
            self[self.TTS] = self[tts_key]
            self[self.TTS_PATH] = self[tts_path_key]
