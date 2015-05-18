#!/usr/bin/env python
# coding=utf-8

"""
Global constants, mostly default values,
public parameter names, and executable paths.
"""

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl (www.readbeyond.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.0.1"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"


### PATHS ###

#ESPEAK_PATH = "/usr/bin/espeak"
ESPEAK_PATH = "espeak"
""" Path to the ``espeak`` executable """

#FFMPEG_PATH = "/usr/bin/ffmpeg"
FFMPEG_PATH = "ffmpeg"
""" Path to the ``ffmpeg`` executable """

#FFPROBE_PATH = "/usr/bin/ffprobe"
FFPROBE_PATH = "ffprobe"
""" Path to the ``ffprobe`` executable """

TMP_PATH = "/tmp/"
""" Path to the temporary directory """


### CONSTANTS ###

ALIGNER_FRAME_RATE = 25
""" Aligner MFCC frame rate, in steps per second.
Default: ``25``, corresponding to steps of ``40ms`` length. """

ALIGNER_MARGIN = 60
""" Aligner margin, in seconds, for striped algorithms.
Default: ``60``, corresponding to ``60s`` ahead and behind
(i.e., ``120s`` total margin). """

ALIGNER_USE_EXACT_ALGO_WHEN_MARGIN_TOO_LARGE = True
""" Use the exact DTW algorithm, instead of a striped algorithm,
if the aligner margin is larger than the synthesized audio file.
Default ``True``. """

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
Key for the identifier of a job

Usage: reserved
"""

RPN_TASK_IDENTIFIER = "task_identifier"
"""
Key for the identifier of a task

Usage: reserved
"""

# public parameter names (PPN)
PPN_JOB_DESCRIPTION = "job_description"
"""
Key for the description of a job

Usage: config string, TXT config file, XML config file

Values: string

Example::

    job_description=This is a sample description

"""

PPN_JOB_LANGUAGE = "job_language"
"""
Key for the language of a job

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.language.Language`

Example::

    job_language=en

"""

PPN_JOB_IS_AUDIO_FILE_NAME_REGEX = "is_audio_file_name_regex"
"""
Key for the regex for matching the audio file name
of tasks in input containers

Usage: config string, TXT config file

Values: regex

Example::

    is_audio_file_name_regex=.*\.mp3
    is_audio_file_name_regex=audio.ogg

"""

PPN_JOB_IS_AUDIO_FILE_RELATIVE_PATH = "is_audio_file_relative_path"
"""
Key for the path, relative to the task root directory,
where the audio files should be searched in input containers

Usage: config string, TXT config file

Values: string (path)

Example::

    is_audio_file_relative_path=../audio
    is_audio_file_relative_path=mp3
    is_audio_file_relative_path=.

"""

PPN_JOB_IS_HIERARCHY_PREFIX = "is_hierarchy_prefix"
"""
Key for the path, relative to the position of the TXT/XML config file,
to be considered the task root directory, in input containers

Usage: config string, TXT config file

Values: string (path)

Example::

    is_hierarchy_prefix=OEBPS/Resources
    is_hierarchy_prefix=.

"""

PPN_JOB_IS_HIERARCHY_TYPE = "is_hierarchy_type"
"""
Key for the type of input container structure

Usage: config string, TXT config file

Values: listed in :class:`aeneas.hierarchytype.HierarchyType`

Example::

    is_hierarchy_type=flat
    is_hierarchy_type=paged

"""

PPN_JOB_IS_TASK_DIRECTORY_NAME_REGEX = "is_task_dir_name_regex"
"""
Key for the regex for matching the task directory names
in input containers with paged hierarchy

Usage: config string, TXT config file

Values: regex

Example::

    is_task_dir_name_regex=[0-9]+
    is_text_dir_name_regex=page[0-9]+

"""

PPN_JOB_IS_TEXT_FILE_FORMAT = "is_text_type"
"""
Key for the format of text files in input containers

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.textfile.TextFileFormat`

Example::

    is_text_type=plain
    is_text_type=parsed
    is_text_type=unparsed

"""

PPN_JOB_IS_TEXT_FILE_NAME_REGEX = "is_text_file_name_regex"
"""
Key for the regex for matching the text file name
of tasks in input containers

Usage: config string, TXT config file

Values: regex

Example::

    is_text_file_name_regex=.*\.xhtml
    is_text_file_name_regex=page.xhtml

"""

PPN_JOB_IS_TEXT_FILE_RELATIVE_PATH = "is_text_file_relative_path"
"""
Key for the path, relative to the task root directory,
where the text files should be searched in input containers

Usage: config string, TXT config file

Values: string (path)

Example::

    is_audio_file_relative_path=../pages
    is_audio_file_relative_path=xhtml
    is_audio_file_relative_path=.

"""

PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX = "is_text_unparsed_class_regex"
"""
Key for the regex for matching the ``class`` attribute
of XML elements containing text fragments to be extracted
from ``unparsed`` text files

Usage: config string, TXT config file, XML config file

Values: regex

Example::

    is_text_unparsed_class_regex=ra
    is_text_unparsed_class_regex=readaloud
    is_text_unparsed_class_regex=ra[0-9]+

"""

PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX = "is_text_unparsed_id_regex"
"""
Key for the regex for matching the ``id`` attribute
of XML elements containing text fragments to be extracted
from ``unparsed`` text files

Usage: config string, TXT config file, XML config file

Values: regex

Example::

    is_text_unparsed_id_regex=f[0-9]+
    is_text_unparsed_id_regex=ra.*

"""

PPN_JOB_IS_TEXT_UNPARSED_ID_SORT = "is_text_unparsed_id_sort"
"""
Key for the sorting algorithm to be used to sort the text fragments
extracted from ``unparsed`` text files, based on their ``id`` attributes

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.idsortingalgorithm.IDSortingAlgorithm`

Example::

    is_text_unparsed_id_sort=lexicographic
    is_text_unparsed_id_sort=numeric
    is_text_unparsed_id_sort=unsorted

"""

PPN_JOB_OS_CONTAINER_FORMAT = "os_job_file_container"
"""
Key for the format of the output container

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.container.ContainerFormat`

Example::

    os_job_file_container=zip

"""

PPN_JOB_OS_FILE_NAME = "os_job_file_name"
"""
Key for the file name of the output container 

Usage: config string, TXT config file, XML config file

Values: string

Example::

    os_job_file_name=output_sync_maps.zip

"""

PPN_JOB_OS_HIERARCHY_PREFIX = "os_job_file_hierarchy_prefix"
"""
Key for the path of the root directory of the output container,
under which the task directories will be created

Usage: config string, TXT config file, XML config file

Values: string (path)

Example::

    os_job_file_hierarchy_prefix=OEBPS/Resources

"""

PPN_JOB_OS_HIERARCHY_TYPE = "os_job_file_hierarchy_type"
"""
Key for the type of output container structure

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.hierarchytype.HierarchyType`

Example::

    os_job_file_hierarchy_type=flat
    os_job_file_hierarchy_type=paged

"""

PPN_TASK_CUSTOM_ID = "task_custom_id"
"""
Key for the custom identifier of a task

Usage: config string, XML config file

Values: string

Example::

    task_custom_id=sonnet001

"""

PPN_TASK_DESCRIPTION = "task_description"
"""
Key for the description of a task

Usage: config string, XML config file

Values: string

Example::

    task_description=This is a sample description

"""

PPN_TASK_LANGUAGE = "task_language"
"""
Key for the language of a task

Usage: config string, XML config file

Values: listed in :class:`aeneas.language.Language`

Example::

    task_language=en

"""

PPN_TASK_IS_AUDIO_FILE_HEAD_LENGTH = "is_audio_file_head_length"
"""
Key for the number of seconds, from the beginning of the audio file
of the task, to be ignored

Usage: config string, XML config file

Values: float

Example::

    is_audio_file_head_length=12.345

"""

PPN_TASK_IS_AUDIO_FILE_PROCESS_LENGTH = "is_audio_file_process_length"
"""
Key for the number of seconds of the audio file of the task to process

Usage: config string, XML config file

Values: float

Example::

    is_audio_file_process_length=987.654

"""

PPN_TASK_IS_TEXT_FILE_FORMAT = "is_text_type"
"""
Key for the format of the text file of the task

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.textfile.TextFileFormat`

Example::

    is_text_type=plain
    is_text_type=parsed
    is_text_type=unparsed

"""

PPN_TASK_IS_TEXT_UNPARSED_CLASS_REGEX = "is_text_unparsed_class_regex"
"""
Key for the regex for matching the ``class`` attribute
of XML elements containing text fragments to be extracted
from the ``unparsed`` text file of the task

Usage: config string, TXT config file, XML config file

Values: regex

Example::

    is_text_unparsed_class_regex=ra
    is_text_unparsed_class_regex=readaloud
    is_text_unparsed_class_regex=ra[0-9]+

"""

PPN_TASK_IS_TEXT_UNPARSED_ID_REGEX = "is_text_unparsed_id_regex"
"""
Key for the regex for matching the ``id`` attribute
of XML elements containing text fragments to be extracted
from the ``unparsed`` text file of the task

Usage: config string, TXT config file, XML config file

Values: regex

Example::

    is_text_unparsed_id_regex=f[0-9]+
    is_text_unparsed_id_regex=ra.*

"""

PPN_TASK_IS_TEXT_UNPARSED_ID_SORT = "is_text_unparsed_id_sort"
"""
Key for the sorting algorithm to be used to sort the text fragments
extracted from ``unparsed`` text files, based on their ``id`` attributes

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.idsortingalgorithm.IDSortingAlgorithm`

Example::

    is_text_unparsed_id_sort=lexicographic
    is_text_unparsed_id_sort=numeric
    is_text_unparsed_id_sort=unsorted

"""

PPN_TASK_OS_FILE_FORMAT = "os_task_file_format"
"""
Key for the format of the sync map output file

Usage: config string, TXT config file, XML config file

Values: listed in :class:`aeneas.syncmap.SyncMapFormat`

Example::

    os_task_file_format=smil
    os_task_file_format=txt
    os_task_file_format=srt

"""

PPN_TASK_OS_FILE_NAME = "os_task_file_name"
"""
Key for the file name of the sync map output file

Usage: config string, TXT config file, XML config file

Values: string

Example::

    os_task_file_name=map.smil

"""

PPN_TASK_OS_FILE_SMIL_AUDIO_REF = "os_task_file_smil_audio_ref"
"""
Key for the ``src`` attribute of ``<audio>`` elements
in the output sync map file in SMIL format

Usage: config string, TXT config file, XML config file

Values: string

Example::

    os_task_file_smil_audio_ref=../audio/p001.mp3
    os_task_file_smil_audio_ref=audio.mp3

"""

PPN_TASK_OS_FILE_SMIL_PAGE_REF = "os_task_file_smil_page_ref"
"""
Key for the ``src`` attribute of ``<text>`` elements
in the output sync map file in SMIL format

Usage: config string, TXT config file, XML config file

Values: string

Example::

    os_task_file_smil_page_ref=../xhtml/page.xhtml
    os_task_file_smil_page_ref=p001.xhtml

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



