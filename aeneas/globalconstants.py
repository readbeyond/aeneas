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
Global constants, mostly default values,
public parameter names, and executable paths.
"""


# CONSTANTS

CONFIG_RESERVED_CHARACTERS = ["~"]
""" List of reserved characters which are forbidden in configuration files """

CONFIG_STRING_ASSIGNMENT_SYMBOL = "="
""" Assignment symbol in config string ``key=value`` pairs """

CONFIG_STRING_SEPARATOR_SYMBOL = "|"
""" Separator of ``key=value`` pairs in config strings """

PARSED_TEXT_SEPARATOR = "|"
""" Separator for input text files in parsed format """

CONFIG_TXT_FILE_NAME = "config.txt"
""" File name for the TXT configuration file in containers """

CONFIG_XML_FILE_NAME = "config.xml"
""" File name for the XML configuration file in containers """

CONFIG_XML_TASK_TAG = "task"
""" ``<task>`` tag in the XML configuration file """

CONFIG_XML_TASKS_TAG = "tasks"
""" ``<tasks>`` tag in the XML configuration file """

MIMETYPE_MAP = {
    "aac": "audio/aac",
    "aiff": "audio/x-aiff",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "mp4": "audio/mp4",
    "oga": "audio/x-vorbis+ogg",
    "ogg": "audio/x-vorbis+ogg",
    "wav": "audio/x-wav",
    "webm": "video/webm"
}
""" Map from audio file extension to mimetype """

TMP_PATH_DEFAULT_NONPOSIX = None
"""
Default temporary directory path for non-POSIX OSes.
Set to ``None`` so that ``tempfile`` will select
the most approriate temporary directory root path.

.. versionadded:: 1.4.1
"""

TMP_PATH_DEFAULT_POSIX = "/tmp/"
"""
Default temporary directory path for POSIX OSes.

.. versionadded:: 1.4.1
"""


# PARAMETER NAMES

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

Values: listed in :class:`~aeneas.language.Language`

Example::

    job_language=eng-GBR
    job_language=eng-USA
    job_language=ita-ITA

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

Values: string

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

Values: string

Example::

    is_hierarchy_prefix=OEBPS/Resources
    is_hierarchy_prefix=.

"""

PPN_JOB_IS_HIERARCHY_TYPE = "is_hierarchy_type"
"""
The type of hierarchy of the input job container.

Usage: config string, TXT config file

Values: listed in :class:`~aeneas.hierarchytype.HierarchyType`

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

Values: string

Example::

    is_audio_file_relative_path=../pages
    is_audio_file_relative_path=xhtml
    is_audio_file_relative_path=.

"""

PPN_JOB_OS_CONTAINER_FORMAT = "os_job_file_container"
"""
The format of the output container.

Usage: config string, TXT config file, XML config file

Values: listed in :class:`~aeneas.container.ContainerFormat`

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

Values: string

Example::

    os_job_file_hierarchy_prefix=OEBPS/Resources

"""

PPN_JOB_OS_HIERARCHY_TYPE = "os_job_file_hierarchy_type"
"""
The type of output container structure.

Usage: config string, TXT config file, XML config file

Values: listed in :class:`~aeneas.hierarchytype.HierarchyType`

Example::

    os_job_file_hierarchy_type=flat
    os_job_file_hierarchy_type=paged

"""

PPN_SYNCMAP_LANGUAGE = "language"
"""
Key for specifying the syncmap language

Values: listed in :class:`~aeneas.language.Language`

Example::

    language=eng-GBR
    language=eng-USA
    language=ita-ITA

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

Values: listed in :class:`~aeneas.language.Language`

Example::

    task_language=eng
    task_language=eng-GBR
    task_language=eng-USA
    task_language=ita

"""

PPN_TASK_ADJUST_BOUNDARY_ALGORITHM = "task_adjust_boundary_algorithm"
"""
The algorithm to be run to adjust the fragment boundaries.
If ``None`` or ``auto``, keep the current boundaries.

Usage: config string, TXT config file, XML config file

Values: listed in :class:`~aeneas.adjustboundaryalgorithm.AdjustBoundaryAlgorithm`

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

PPN_TASK_ADJUST_BOUNDARY_NO_ZERO = "task_adjust_boundary_no_zero"
"""
If specified, do not allow fragments with zero duration.

Note: before version 1.7.0 this parameter
was called ``os_task_file_no_zero``.

Usage: config string, TXT config file, XML config file

Values: string

Example::

    task_adjust_boundary_no_zero=True

.. versionadded:: 1.5.0
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

PPN_TASK_ADJUST_BOUNDARY_NONSPEECH_MIN = "task_adjust_boundary_nonspeech_min"
"""
If greater than zero, create a new sync map fragment
for each nonspeech interval
with duration greater than or equal to this value.

The text to be associated with these nonspeech intervals
can be specified with ``task_adjust_boundary_nonspeech_string``.

Usage: config string, TXT config file, XML config file

Values: float

Example::

    task_adjust_boundary_nonspeech_min=0.500

.. versionadded:: 1.7.0
"""

PPN_TASK_ADJUST_BOUNDARY_NONSPEECH_STRING = "task_adjust_boundary_nonspeech_string"
"""
Specify the text to be associated with nonspeech intervals
of length greater than or equal to
the value provided in ``task_adjust_boundary_nonspeech_min``.

Use the string ``PPV_TASK_ADJUST_BOUNDARY_NONSPEECH_REMOVE``
to remove these intervals from the output sync map.

Usage: config string, TXT config file, XML config file

Values: string

Example::

    task_adjust_boundary_nonspeech_string=REMOVE
    task_adjust_boundary_nonspeech_string=(sil)
    task_adjust_boundary_nonspeech_string=<sil>

.. versionadded:: 1.7.0
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

Values: listed in :class:`~aeneas.textfile.TextFileFormat`

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

Values: string

Example::

    is_text_file_transliterate_map=trans.map

"""

PPN_TASK_IS_TEXT_MPLAIN_WORD_SEPARATOR = "is_text_mplain_word_separator"
"""
The word separator to be used when splitting words
in ``mplain`` input text files.

You can use the following special strings:

* ``equal`` for a ``=`` character (ASCII ``0x20``),
* ``pipe`` for a ``|`` character (ASCII ``0x7C``),
* ``space`` for a space character (ASCII ``0x20``),
* ``tab`` for a tab character (ASCII ``0x09``).

Any other string will be used as the word separator.
If not specified, the ``space`` will be used.

Usage: config string, TXT config file, XML config file

Values: string

Example::

    is_text_mplain_word_separator=space
    is_text_mplain_word_separator=tab
    is_text_mplain_word_separator=,

"""

PPN_TASK_IS_TEXT_MUNPARSED_L1_ID_REGEX = "is_text_munparsed_l1_id_regex"
"""
The regex to match ``id`` attributes for level 1 (paragraph) text fragments.
It applies to ``munparsed`` text files only.

Usage: config string, TXT config file, XML config file

Values: regex

Example::

    is_text_munparsed_l1_id_regex=p[0-9]+

.. versionadded:: 1.5.0
"""

PPN_TASK_IS_TEXT_MUNPARSED_L2_ID_REGEX = "is_text_munparsed_l2_id_regex"
"""
The regex to match ``id`` attributes for level 2 (sentence) text fragments.
It applies to ``munparsed`` text files only.

Usage: config string, TXT config file, XML config file

Values: regex

Example::

    is_text_munparsed_l2_id_regex=s[0-9]+
    is_text_munparsed_l2_id_regex=p[0-9]+s[0-9]+

.. versionadded:: 1.5.0

"""

PPN_TASK_IS_TEXT_MUNPARSED_L3_ID_REGEX = "is_text_munparsed_l3_id_regex"
"""
The regex to match ``id`` attributes for level 3 (word) text fragments.
It applies to ``munparsed`` text files only.

Usage: config string, TXT config file, XML config file

Values: regex

Example::

    is_text_munparsed_l3_id_regex=w[0-9]+
    is_text_munparsed_l3_id_regex=p[0-9]+s[0-9]+w[0-9]+

.. versionadded:: 1.5.0

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
It applies to ``unparsed`` text files only.

Usage: config string, TXT config file, XML config file

Values: listed in :class:`~aeneas.idsortingalgorithm.IDSortingAlgorithm`

Example::

    is_text_unparsed_id_sort=lexicographic
    is_text_unparsed_id_sort=numeric
    is_text_unparsed_id_sort=unsorted

"""

PPN_TASK_OS_FILE_FORMAT = "os_task_file_format"
"""
The format of the sync map output for the task.

Usage: config string, TXT config file, XML config file

Values: listed in :class:`~aeneas.syncmap.SyncMapFormat`

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

PPN_TASK_OS_FILE_LEVELS = "os_task_file_levels"
"""
If the input text file is multilevel,
only outputs the specified levels.

This parameter has no effect for single-level
input text files or output sync map formats.

Usage: config string, TXT config file, XML config file

Values: string

Example::

    os_task_file_levels=123
    os_task_file_levels=3

.. versionadded:: 1.5.0
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

PPN_TASK_OS_FILE_EAF_AUDIO_REF = "os_task_file_eaf_audio_ref"
"""
The value of the ``<MEDIA_URL>`` element in the output sync map,
complete with the ``file://`` prefix.
It applies to ``EAF`` sync maps only.

Usage: config string, TXT config file, XML config file

Values: string

Example::

    os_task_file_eaf_audio_ref=file:///audio/p001.mp3

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

Values: listed in :class:`~aeneas.syncmap.SyncMapHeadTailFormat`

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

Values: string

Example::

    <is_text_file>OEBPS/Resources/sonnet001.txt</is_text_file>

"""

PPN_TASK_IS_AUDIO_FILE_XML = "is_audio_file"
"""
Key for the path, relative to the XML config file,
of the audio file of the current task

Usage: XML config file

Values: string

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

PPV_TASK_ADJUST_BOUNDARY_NONSPEECH_REMOVE = "REMOVE"
"""
Use this string as the value of
the ``PPN_TASK_ADJUST_BOUNDARY_NONSPEECH_STRING`` parameter
to remove long nonspeech intervals from the output sync map.

.. versionadded:: 1.7.0
"""
