#!/usr/bin/env python
# coding=utf-8

"""
Global constants, mostly default values,
public parameter names, and executable paths.
"""

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.4.1"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"


### CONSTANTS ###

CONFIG_TXT_FILE_NAME = "config.txt"
""" File name for the TXT configuration file in containers """

CONFIG_XML_FILE_NAME = "config.xml"
""" File name for the XML configuration file in containers """

CONFIG_XML_TASKS_TAG = "tasks"
""" ``<tasks>`` tag in the XML configuration file """

CONFIG_XML_TASK_TAG = "task"
""" ``<task>`` tag in the XML configuration file """

CONFIG_RESERVED_CHARACTERS = ["~"]
""" List of reserved characters which are forbidden in configuration files """

CONFIG_STRING_SEPARATOR_SYMBOL = "|"
""" Separator of ``key=value`` pairs in config strings """

CONFIG_STRING_ASSIGNMENT_SYMBOL = "="
""" Assignment symbol in config string ``key=value`` pairs """

PARSED_TEXT_SEPARATOR = "|"
""" Separator for input text files in parsed format """

# reserved parameter names (RPN)
RPN_JOB_IDENTIFIER = "job_identifier"
"""
The identifier of a job. Reserved.

Usage: reserved
"""

RPN_TASK_IDENTIFIER = "task_identifier"
"""
The identifier of a task. Reserved.

Usage: reserved
"""

# public parameter names (PPN)
PPN_JOB_DESCRIPTION = "job_description"
"""
A human-readable description of the job.

Usage: config string, TXT config file, XML config file

Values: string

Example::

    job_description=This is a sample description

"""

PPN_JOB_LANGUAGE = "job_language"
"""
The language of the job.

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.language.Language`

Example::

    job_language=en

"""

PPN_JOB_IS_AUDIO_FILE_NAME_REGEX = "is_audio_file_name_regex"
"""
The regex to match audio files in this job.

Usage: config string, TXT config file

Values: regex

Example::

    is_audio_file_name_regex=.*\.mp3
    is_audio_file_name_regex=audio.ogg

"""

PPN_JOB_IS_AUDIO_FILE_RELATIVE_PATH = "is_audio_file_relative_path"
"""
The path, relative to the task root directory,
where the audio files should be searched in input containers.

Usage: config string, TXT config file

Values: string (path)

Example::

    is_audio_file_relative_path=../audio
    is_audio_file_relative_path=mp3
    is_audio_file_relative_path=.

"""

PPN_JOB_IS_HIERARCHY_PREFIX = "is_hierarchy_prefix"
"""
The path, relative to the position of the TXT/XML config file,
to be considered the task root directory, in input containers.

Usage: config string, TXT config file

Values: string (path)

Example::

    is_hierarchy_prefix=OEBPS/Resources
    is_hierarchy_prefix=.

"""

PPN_JOB_IS_HIERARCHY_TYPE = "is_hierarchy_type"
"""
The type of hierarchy of the input job container.

Usage: config string, TXT config file

Values: listed in :class:`aeneas.hierarchytype.HierarchyType`

Example::

    is_hierarchy_type=flat
    is_hierarchy_type=paged

"""

PPN_JOB_IS_TASK_DIRECTORY_NAME_REGEX = "is_task_dir_name_regex"
"""
The regex to match task directory names
within the base task directory in input containers.
Applies to paged hierarchies only.

Usage: config string, TXT config file

Values: regex

Example::

    is_task_dir_name_regex=[0-9]+
    is_text_dir_name_regex=page[0-9]+

"""

PPN_JOB_IS_TEXT_FILE_FORMAT = "is_text_type"
"""
The text file format of text files in input containers.

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.textfile.TextFileFormat`

Example::

    is_text_type=plain
    is_text_type=parsed
    is_text_type=unparsed

"""

PPN_JOB_IS_TEXT_FILE_NAME_REGEX = "is_text_file_name_regex"
"""
The regex for matching the text file name
of tasks in input containers.

Usage: config string, TXT config file

Values: regex

Example::

    is_text_file_name_regex=.*\.xhtml
    is_text_file_name_regex=page.xhtml

"""

PPN_JOB_IS_TEXT_FILE_RELATIVE_PATH = "is_text_file_relative_path"
"""
The path, relative to the task root directory,
where the text files should be searched in input containers.

Usage: config string, TXT config file

Values: string (path)

Example::

    is_audio_file_relative_path=../pages
    is_audio_file_relative_path=xhtml
    is_audio_file_relative_path=.

"""

PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX = "is_text_unparsed_class_regex"
"""
The regex for matching the ``class`` attribute
of XML elements containing text fragments to be extracted
from ``unparsed`` text files.

Usage: config string, TXT config file, XML config file

Values: regex

Example::

    is_text_unparsed_class_regex=ra
    is_text_unparsed_class_regex=readaloud
    is_text_unparsed_class_regex=ra[0-9]+

"""

PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX = "is_text_unparsed_id_regex"
"""
The regex for matching the ``id`` attribute
of XML elements containing text fragments to be extracted
from ``unparsed`` text files.

Usage: config string, TXT config file, XML config file

Values: regex

Example::

    is_text_unparsed_id_regex=f[0-9]+
    is_text_unparsed_id_regex=ra.*

"""

PPN_JOB_IS_TEXT_UNPARSED_ID_SORT = "is_text_unparsed_id_sort"
"""
The sorting algorithm to be used to sort the text fragments
extracted from ``unparsed`` text files, based on their ``id`` attributes.

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.idsortingalgorithm.IDSortingAlgorithm`

Example::

    is_text_unparsed_id_sort=lexicographic
    is_text_unparsed_id_sort=numeric
    is_text_unparsed_id_sort=unsorted

"""

PPN_JOB_OS_CONTAINER_FORMAT = "os_job_file_container"
"""
The format of the output container.

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.container.ContainerFormat`

Example::

    os_job_file_container=zip

"""

PPN_JOB_OS_FILE_NAME = "os_job_file_name"
"""
The file name of the output container.

Usage: config string, TXT config file, XML config file

Values: string

Example::

    os_job_file_name=output_sync_maps.zip

"""

PPN_JOB_OS_HIERARCHY_PREFIX = "os_job_file_hierarchy_prefix"
"""
The path of the root directory of the output container,
under which the task directories will be created.

Usage: config string, TXT config file, XML config file

Values: string (path)

Example::

    os_job_file_hierarchy_prefix=OEBPS/Resources

"""

PPN_JOB_OS_HIERARCHY_TYPE = "os_job_file_hierarchy_type"
"""
The type of output container structure.

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.hierarchytype.HierarchyType`

Example::

    os_job_file_hierarchy_type=flat
    os_job_file_hierarchy_type=paged

"""

PPN_SYNCMAP_LANGUAGE = "language"
"""
Key for specifying the syncmap language

Values: listed in :class:`aeneas.language.Language`

Example::

    language=en
    language=it

.. versionadded:: 1.2.0
"""

PPN_TASK_CUSTOM_ID = "task_custom_id"
"""
The custom, human-readable identifier of a task.

Usage: config string, XML config file

Values: string

Example::

    task_custom_id=sonnet001

"""

PPN_TASK_DESCRIPTION = "task_description"
"""
The description of a task.

Usage: config string, XML config file

Values: string

Example::

    task_description=This is a sample description

"""

PPN_TASK_LANGUAGE = "task_language"
"""
The language of a task.

Usage: config string, XML config file

Values: listed in :class:`aeneas.language.Language`

Example::

    task_language=en

"""

PPN_TASK_ADJUST_BOUNDARY_ALGORITHM = "task_adjust_boundary_algorithm"
"""
The algorithm to be run to adjust the fragment boundaries.
If ``None`` or ``auto``, keep the current boundaries.

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.adjustboundaryalgorithm.AdjustBoundaryAlgorithm`

Example::

    task_adjust_boundary_algorithm=aftercurrent
    task_adjust_boundary_algorithm=auto
    task_adjust_boundary_algorithm=beforenext
    task_adjust_boundary_algorithm=offset
    task_adjust_boundary_algorithm=percent
    task_adjust_boundary_algorithm=rate
    task_adjust_boundary_algorithm=rateaggressive

.. versionadded:: 1.0.4
"""

PPN_TASK_ADJUST_BOUNDARY_AFTERCURRENT_VALUE = "task_adjust_boundary_aftercurrent_value"
"""
The new boundary between two consecutive fragments
will be set at ``value`` seconds
after the end of the first fragment.

Requires ``task_adjust_boundary_algorithm=aftercurrent``.

Usage: config string, TXT config file, XML config file

Values: float

Example::

    task_adjust_boundary_aftercurrent_value=0.150

.. versionadded:: 1.0.4
"""

PPN_TASK_ADJUST_BOUNDARY_BEFORENEXT_VALUE = "task_adjust_boundary_beforenext_value"
"""
The new boundary between two consecutive fragments
will be set at ``value`` seconds
before the beginning of the second fragment.

Requires ``task_adjust_boundary_algorithm=beforenext``.

Usage: config string, TXT config file, XML config file

Values: float

Example::

    task_adjust_boundary_beforenext_value=0.200

.. versionadded:: 1.0.4
"""

PPN_TASK_ADJUST_BOUNDARY_OFFSET_VALUE = "task_adjust_boundary_offset_value"
"""
The new boundary between two consecutive fragments
will be set at ``value`` seconds from the current value.
A negative ``value`` will move the boundary back,
a positive ``value`` will move the boundary forward.

Requires ``task_adjust_boundary_algorithm=offset``.

Usage: config string, TXT config file, XML config file

Values: float

Example::

    task_adjust_boundary_offset_value=-0.200
    task_adjust_boundary_offset_value=0.150

.. versionadded:: 1.1.0
"""

PPN_TASK_ADJUST_BOUNDARY_PERCENT_VALUE = "task_adjust_boundary_percent_value"
"""
The new boundary between two consecutive fragments
will be set at this ``value`` percent
of the nonspeech interval between the two fragments.
The value must be between ``0`` and ``100``.

Requires ``task_adjust_boundary_algorithm=percent``.

Usage: config string, TXT config file, XML config file

Values: int

Example::

    task_adjust_boundary_percent_value=0
    task_adjust_boundary_percent_value=50
    task_adjust_boundary_percent_value=75
    task_adjust_boundary_percent_value=100

.. versionadded:: 1.0.4
"""

PPN_TASK_ADJUST_BOUNDARY_RATE_VALUE = "task_adjust_boundary_rate_value"
"""
The new boundary will be set trying to keep the rate
of all the fragments below this ``value`` characters/second.
The value must be greater than ``0``.

Requires ``task_adjust_boundary_algorithm=rate`` or
``task_adjust_boundary_algorithm=rateaggressive``.

Usage: config string, TXT config file, XML config file

Values: float

Example::

    task_adjust_boundary_rate_value=21.0

.. versionadded:: 1.0.4
"""

PPN_TASK_IS_AUDIO_FILE_DETECT_HEAD_MAX = "is_audio_file_detect_head_max"
"""
When synchronizing, auto detect the head of the audio file,
using the provided value as an upper bound, and disregard
these many seconds from the beginning of the audio file.

If the ``is_audio_file_head_length`` parameter is also provided,
the auto detection will not take place.

NOTE: This is an experimental feature, use with caution.

Usage: config string, XML config file

Values: float

Example::

    is_audio_file_detect_head_max=10.0

.. versionadded:: 1.2.0
"""

PPN_TASK_IS_AUDIO_FILE_DETECT_HEAD_MIN = "is_audio_file_detect_head_min"
"""
When synchronizing, auto detect the head of the audio file,
using the provided value as a lower bound, and disregard
these many seconds from the beginning of the audio file.

If the ``is_audio_file_head_length`` parameter is also provided,
the auto detection will not take place.

NOTE: This is an experimental feature, use with caution.

Usage: config string, XML config file

Values: float

Example::

    is_audio_file_detect_head_min=3.0

.. versionadded:: 1.2.0
"""

PPN_TASK_IS_AUDIO_FILE_DETECT_TAIL_MAX = "is_audio_file_detect_tail_max"
"""
When synchronizing, auto detect the tail of the audio file,
using the provided value as an upper bound, and disregard
these many seconds from the end of the audio file.

If the ``is_audio_file_process_length`` parameter or
the ``is_audio_file_tail_length`` parameter
are also provided,
the auto detection will not take place.

NOTE: This is an experimental feature, use with caution.

Usage: config string, XML config file

Values: float

Example::

    is_audio_file_detect_tail_max=10.0

.. versionadded:: 1.2.0
"""

PPN_TASK_IS_AUDIO_FILE_DETECT_TAIL_MIN = "is_audio_file_detect_tail_min"
"""
When synchronizing, auto detect the tail of the audio file,
using the provided value as a lower bound, and disregard
these many seconds from the end of the audio file.

If the ``is_audio_file_process_length`` parameter or
the ``is_audio_file_tail_length`` parameter
are also provided,
the auto detection will not take place.

NOTE: This is an experimental feature, use with caution.

Usage: config string, XML config file

Values: float

Example::

    is_audio_file_detect_tail_min=0.0

.. versionadded:: 1.2.0
"""

PPN_TASK_IS_AUDIO_FILE_HEAD_LENGTH = "is_audio_file_head_length"
"""
When synchronizing, disregard
these many seconds from the beginning of the audio file.

Usage: config string, XML config file

Values: float

Example::

    is_audio_file_head_length=12.345

"""

PPN_TASK_IS_AUDIO_FILE_PROCESS_LENGTH = "is_audio_file_process_length"
"""
When synchronizing, process only these many seconds
from the audio file, starting at the beginning of the file
or at the end of the head.

Usage: config string, XML config file

Values: float

Example::

    is_audio_file_process_length=987.654

"""

PPN_TASK_IS_AUDIO_FILE_TAIL_LENGTH = "is_audio_file_tail_length"
"""
When synchronizing, disregard
these many seconds from the end of the audio file.

Note that if both ``is_audio_file_process_length``
and ``is_audio_file_tail_length`` are provided,
only the former will be taken into account,
and ``is_audio_file_tail_length`` will be ignored.

Usage: config string, XML config file

Values: float

Example::

    is_audio_file_tail_length=12.345

"""

PPN_TASK_IS_TEXT_FILE_FORMAT = "is_text_type"
"""
The format of the input text file.

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.textfile.TextFileFormat`

Example::

    is_text_type=plain
    is_text_type=parsed
    is_text_type=unparsed

"""

PPN_TASK_IS_TEXT_FILE_IGNORE_REGEX = "is_text_file_ignore_regex"
"""
The regex to match text to be ignored for alignment purposes.
The output sync map file will contain the original text.

Usage: config string, TXT config file, XML config file

Values: regex

Example::

    is_text_file_ignore_regex=\\[.*?\\]

"""

PPN_TASK_IS_TEXT_FILE_TRANSLITERATE_MAP = "is_text_file_transliterate_map"
"""
The path to the transliteration map file to be used to delete/replace
characters in the input text file for alignment purposes.
The output sync map file will contain the original text.

Usage: config string, TXT config file, XML config file

Values: string (path)

Example::

    is_text_file_transliterate_map=trans.map

"""

PPN_TASK_IS_TEXT_UNPARSED_CLASS_REGEX = "is_text_unparsed_class_regex"
"""
The regex to match ``class`` attributes for text fragments.
It applies to ``unparsed`` text files only.

Usage: config string, TXT config file, XML config file

Values: regex

Example::

    is_text_unparsed_class_regex=ra
    is_text_unparsed_class_regex=readaloud
    is_text_unparsed_class_regex=ra[0-9]+

"""

PPN_TASK_IS_TEXT_UNPARSED_ID_REGEX = "is_text_unparsed_id_regex"
"""
The regex to match ``id`` attributes for text fragments.
It applies to ``unparsed`` text files only.

Usage: config string, TXT config file, XML config file

Values: regex

Example::

    is_text_unparsed_id_regex=f[0-9]+
    is_text_unparsed_id_regex=ra.*

"""

PPN_TASK_IS_TEXT_UNPARSED_ID_SORT = "is_text_unparsed_id_sort"
"""
The algorithm to sort text fragments by their ``id`` attributes.
It applies to unparsed text files only.

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.idsortingalgorithm.IDSortingAlgorithm`

Example::

    is_text_unparsed_id_sort=lexicographic
    is_text_unparsed_id_sort=numeric
    is_text_unparsed_id_sort=unsorted

"""

PPN_TASK_OS_FILE_FORMAT = "os_task_file_format"
"""
The format of the sync map output for the task.

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.syncmap.SyncMapFormat`

Example::

    os_task_file_format=smil
    os_task_file_format=txt
    os_task_file_format=srt

"""

PPN_TASK_OS_FILE_ID_REGEX = "os_task_file_id_regex"
"""
The regex to be used for the fragment identifiers
of the sync map output file.

This parameter will be used only
when the input text file has `plain` or `subtitles` format;
for `parsed` and `unparsed` input text files, the identifiers
contained in the input text file will be used instead.

When specified, the value must contain an interger placeholder,
for example ``%d`` or ``%06d``.

Usage: config string, TXT config file, XML config file

Values: string

Example::

    os_task_file_id_regex=f%06d
    os_task_file_id_regex=Word%03d

.. versionadded:: 1.3.1
"""

PPN_TASK_OS_FILE_NAME = "os_task_file_name"
"""
The name of the sync map file output for the task.

If processing a Job,
the value might contain the ``PPV_OS_TASK_PREFIX`` placeholder,
that will be replaced by a suitable path string.

Usage: config string, TXT config file, XML config file

Values: string

Example::

    os_task_file_name=map.smil

"""

PPN_TASK_OS_FILE_SMIL_AUDIO_REF = "os_task_file_smil_audio_ref"
"""
The value of the ``src`` attribute for the ``<audio>`` element
in the output sync map.
It applies to ``SMIL`` sync maps only.

Usage: config string, TXT config file, XML config file

Values: string

Example::

    os_task_file_smil_audio_ref=../audio/p001.mp3
    os_task_file_smil_audio_ref=audio.mp3

"""

PPN_TASK_OS_FILE_SMIL_PAGE_REF = "os_task_file_smil_page_ref"
"""
The value of the ``src`` attribute for the ``<text>`` element
in the output sync map.
It applies to ``SMIL`` sync maps only.

Usage: config string, TXT config file, XML config file

Values: string

Example::

    os_task_file_smil_page_ref=../xhtml/page.xhtml
    os_task_file_smil_page_ref=p001.xhtml

"""

PPN_TASK_OS_FILE_HEAD_TAIL_FORMAT = "os_task_file_head_tail_format"
"""
The format of the head and tail of the sync map output for the task.

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.syncmap.SyncMapHeadTailFormat`

Example::

    os_task_file_head_tail_format=add
    os_task_file_head_tail_format=hidden
    os_task_file_head_tail_format=stretch

.. versionadded:: 1.2.0
"""

PPN_TASK_IS_TEXT_FILE_XML = "is_text_file"
"""
Key for the path, relative to the XML config file,
of the text file of the current task

Usage: XML config file

Values: string (path)

Example::

    <is_text_file>OEBPS/Resources/sonnet001.txt</is_text_file>

"""

PPN_TASK_IS_AUDIO_FILE_XML = "is_audio_file"
"""
Key for the path, relative to the XML config file,
of the audio file of the current task

Usage: XML config file

Values: string (path)

Example::

    <is_audio_file>OEBPS/Resources/sonnet001.mp3</is_audio_file>

"""

# public parameter values (PPV)
PPV_OS_TASK_PREFIX = "$PREFIX"
"""
Placeholder for the actual task directory or task_custom_id value.

Usage: TXT config file

Example::

    os_task_file_name=$PREFIX.smil

"""



### RUNTIMECONFIGURATION ###

RC_ALLOW_UNLISTED_LANGUAGES = "allow_unlisted_languages"
""" If ``True``, allow using a language code not listed in ``languages.py``;
otherwise, raise an error if the user attempts to use a language not listed.
Default: ``True``. """

RC_C_EXTENSIONS = "c_extensions"
""" If ``True`` and Python C extensions are available, use them.
Otherwise, use pure Python code.
Default: ``True``. """

RC_DTW_ALGORITM = "dtw_algorithm"
""" DTW aligner algorithm.
Default: ``stripe``. """

RC_DTW_MARGIN = "dtw_margin"
""" DTW aligner margin, in seconds, for the ``stripe`` algorithm.
Default: ``60``, corresponding to ``60s`` ahead and behind
(i.e., ``120s`` total margin). """

RC_ESPEAK_PATH = "espeak_path"
""" Path to the ``espeak`` executable.
Default: ``espeak``. """

RC_FFMPEG_PATH = "ffmpeg_path"
""" Path to the ``ffmpeg`` executable.
Default: ``ffmpeg``. """

RC_FFMPEG_SAMPLE_RATE = "ffmpeg_sample_rate"
""" Sample rate for ``ffmpeg``, in Hertz.
Default: ``16000``. """

RC_FFPROBE_PATH = "ffprobe_path"
""" Path to the ``ffprobe`` executable.
Default: ``ffprobe``. """

RC_JOB_MAX_TASKS = "job_max_tasks"
""" Maximum number of Tasks of a Job.
If a Job has more Tasks than this value,
it will not be executed and an error will be raised.
Use ``0`` for disabling this check.
Default: ``0`` (disabled). """

RC_MFCC_FILTERS = "mfcc_filters"
""" Number of filters for extracting MFCCs.
Default: ``40``. """

RC_MFCC_SIZE = "mfcc_size"
""" Number of MFCCs to extract, including the 0th.
Default: ``13``. """

RC_MFCC_FFT_ORDER = "mfcc_fft_order"
""" Order of the RFFT for extracting MFCCs.
It must be a power of two.
Default: ``512``. """

RC_MFCC_LOWER_FREQUENCY = "mfcc_lower_frequency"
""" Lower frequency to be used for extracting MFCCs, in Hertz.
Default: ``133.3333``. """

RC_MFCC_UPPER_FREQUENCY = "mfcc_upper_frequency"
""" Upper frequency to be used for extracting MFCCs, in Hertz.
Default: ``6855.4976``. """

RC_MFCC_EMPHASIS_FACTOR = "mfcc_emphasis_factor"
""" Emphasis factor to be applied to MFCCs.
Default: ``0.970``. """

RC_MFCC_WINDOW_LENGTH = "mfcc_window_length"
""" Length of the window for extracting MFCCs, in seconds.
It is usual to set it between 1.5 and 4 times
the value of ``RC_MFCC_WINDOW_SHIFT``.
Default: ``0.100``. """

RC_MFCC_WINDOW_SHIFT = "mfcc_window_shift"
""" Shift of the window for extracting MFCCs, in seconds.
This parameter is basically the time step
of the synchronization maps output.
Default: ``0.040``. """

RC_TASK_MAX_AUDIO_LENGTH = "task_max_audio_length"
""" Maximum length of the audio file of a Task, in seconds.
If a Task has an audio file longer than this value,
it will not be executed and an error will be raised.
Use ``0`` for disabling this check.
Default: ``7200`` seconds. """

RC_TASK_MAX_TEXT_LENGTH = "task_max_text_length"
""" Maximum number of text fragments in the text file of a Task.
If a Task has more text fragments than this value,
it will not be executed and an error will be raised.
Use ``0`` for disabling this check.
Default: ``0`` (disabled). """

RC_TMP_PATH = "tmp_path"
""" Path to the temporary directory to be used.
Default: ``None``, meaning that the default temporary directory
will be set by ``RC_TMP_PATH_DEFAULT_POSIX``
or ``RC_TMP_PATH_DEFAULT_NONPOSIX``. """

RC_VAD_EXTEND_SPEECH_INTERVAL_AFTER = "vad_extend_speech_after"
"""
Extend to the right (after/future)
a speech interval found by the VAD algorithm,
by this many seconds.
Default: ``0`` seconds.

.. versionadded:: 1.0.4
"""

RC_VAD_EXTEND_SPEECH_INTERVAL_BEFORE = "vad_extend_speech_before"
"""
Extend to the left (before/past)
a speech interval found by the VAD algorithm,
by this many seconds.
Default: ``0`` seconds.

.. versionadded:: 1.0.4
"""

RC_VAD_LOG_ENERGY_THRESHOLD = "vad_log_energy_threshold"
"""
Threshold for the VAD algorithm to decide
that a given frame contains speech.
Note that this is the log10 of the energy coefficient.
Default: ``0.699`` = ``log10(5)``, that is, a frame must have
an energy at least 5 times higher than the minimum
to be considered a speech frame.

.. versionadded:: 1.0.4
"""

RC_VAD_MIN_NONSPEECH_LENGTH = "vad_min_nonspeech_length"
"""
Minimum length, in seconds, of a nonspeech interval.
Default: ``0.200`` seconds.

.. versionadded:: 1.0.4
"""



### DEFAULT VALUES ###

RC_TMP_PATH_DEFAULT_POSIX = "/tmp/"
""" Default temporary directory path for POSIX OSes. """

RC_TMP_PATH_DEFAULT_NONPOSIX = None
""" Default temporary directory path for non-POSIX OSes.
Set to ``None`` so that ``tempfile`` will select
the most approriate temporary directory root path. """



