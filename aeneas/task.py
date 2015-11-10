#!/usr/bin/env python
# coding=utf-8

"""
A structure representing a task, that is,
an audio file and a list of text fragments
to be synchronized.
"""

import os
import uuid

from aeneas.audiofile import AudioFile
from aeneas.logger import Logger
from aeneas.textfile import TextFile
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.3.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class Task(object):
    """
    A structure representing a task, that is,
    an audio file and a list of text fragments
    to be synchronized.

    :param config_string: the task configuration string
    :type  config_string: string

    :raises TypeError: if ``config_string`` is not ``None`` and not an instance of ``str`` or ``unicode``
    """

    TAG = "Task"

    def __init__(self, config_string=None, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()
        self.identifier = str(uuid.uuid4()).lower()
        self.configuration = None
        self.audio_file_path = None # relative to input container root
        self.audio_file_path_absolute = None # concrete path, file will be read from this!
        self.audio_file = None
        self.text_file_path = None # relative to input container root
        self.text_file_path_absolute = None # concrete path, file will be read from this!
        self.text_file = None
        self.sync_map_file_path = None # relative to output container root
        self.sync_map_file_path_absolute = None # concrete path, file will be written to this!
        self.sync_map = None
        if config_string is not None:
            self.configuration = TaskConfiguration(config_string)

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def __str__(self):
        accumulator = ""
        accumulator += "%s: '%s'\n" % (gc.RPN_TASK_IDENTIFIER, self.identifier)
        accumulator += "Configuration:\n%s\n" % str(self.configuration)
        accumulator += "Audio file path: %s\n" % self.audio_file_path
        accumulator += "Audio file path (absolute): %s\n" % self.audio_file_path_absolute
        accumulator += "Text file path: %s\n" % self.text_file_path
        accumulator += "Text file path (absolute): %s\n" % self.text_file_path_absolute
        accumulator += "Sync map file path: %s\n" % self.sync_map_file_path
        accumulator += "Sync map file path (absolute): %s\n" % self.sync_map_file_path_absolute
        return accumulator

    @property
    def identifier(self):
        """
        The identifier of the task.

        :rtype: string
        """
        return self.__identifier
    @identifier.setter
    def identifier(self, value):
        self.__identifier = value

    @property
    def audio_file_path_absolute(self):
        """
        The absolute path of the audio file.

        :rtype: string (path)
        """
        return self.__audio_file_path_absolute
    @audio_file_path_absolute.setter
    def audio_file_path_absolute(self, audio_file_path_absolute):
        self.__audio_file_path_absolute = audio_file_path_absolute
        self._populate_audio_file()

    @property
    def text_file_path_absolute(self):
        """
        The absolute path of the text file.

        :rtype: string (path)
        """
        return self.__text_file_path_absolute
    @text_file_path_absolute.setter
    def text_file_path_absolute(self, text_file_path_absolute):
        self.__text_file_path_absolute = text_file_path_absolute
        self._populate_text_file()

    @property
    def sync_map_file_path_absolute(self):
        """
        The absolute path of the sync map file.

        :rtype: string (path)
        """
        return self.__sync_map_file_path_absolute
    @sync_map_file_path_absolute.setter
    def sync_map_file_path_absolute(self, sync_map_file_path_absolute):
        self.__sync_map_file_path_absolute = sync_map_file_path_absolute

    def output_sync_map_file(self, container_root_path=None):
        """
        Output the sync map file for this task.

        If ``container_root_path`` is specified,
        the output sync map file will be created
        at the path obtained by joining
        the ``container_root_path`` and the relative path
        of the sync map inside the container.

        Otherwise, the sync map file will be created at the path
        ``sync_map_file_path_absolute``.

        Return the the path of the sync map file created,
        or ``None`` if an error occurred.

        :param container_root_path: the path to the root directory
                                    for the output container
        :type  container_root_path: string (path)
        :rtype: string (path)
        """
        if self.sync_map is None:
            self._log("sync_map is None", Logger.CRITICAL)
            raise TypeError("sync_map object has not been set")

        if (
                (container_root_path is not None) and
                (self.sync_map_file_path is None)
            ):
            self._log("The (internal) path of the sync map has been set", Logger.CRITICAL)
            raise TypeError("The (internal) path of the sync map has been set")

        self._log(["container_root_path is %s", container_root_path])
        self._log(["self.sync_map_file_path is %s", self.sync_map_file_path])
        self._log(["self.sync_map_file_path_absolute is %s", self.sync_map_file_path_absolute])

        if (container_root_path is not None) and (self.sync_map_file_path is not None):
            path = os.path.join(container_root_path, self.sync_map_file_path)
        elif self.sync_map_file_path_absolute:
            path = self.sync_map_file_path_absolute
        gf.ensure_parent_directory(path)
        self._log(["Output sync map to %s", path])

        sync_map_format = self.configuration.os_file_format
        page_ref = self.configuration.os_file_smil_page_ref
        audio_ref = self.configuration.os_file_smil_audio_ref

        self._log(["sync_map_format is %s", sync_map_format])
        self._log(["page_ref is %s", page_ref])
        self._log(["audio_ref is %s", audio_ref])

        self._log("Calling sync_map.write...")
        parameters = dict()
        parameters[gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF] = page_ref
        parameters[gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF] = audio_ref
        self.sync_map.write(sync_map_format, path, parameters)
        self._log("Calling sync_map.write... done")
        return path

    def _populate_audio_file(self):
        """
        Create the ``self.audio_file`` object by reading
        the audio file at ``self.audio_file_path_absolute``.
        """
        self._log("Populate audio file...")
        if self.audio_file_path_absolute is not None:
            self._log(["audio_file_path_absolute is '%s'", self.audio_file_path_absolute])
            self.audio_file = AudioFile(
                file_path=self.audio_file_path_absolute,
                logger=self.logger
            )
            self.audio_file.read_properties()
        else:
            self._log("audio_file_path_absolute is None")
        self._log("Populate audio file... done")

    def _populate_text_file(self):
        """
        Create the ``self.text_file`` object by reading
        the text file at ``self.text_file_path_absolute``.
        """
        self._log("Populate text file...")
        if (
                (self.text_file_path_absolute is not None) and
                (self.configuration.language is not None)
            ):
            parameters = dict()
            # the following might be None
            parameters[gc.PPN_TASK_IS_TEXT_FILE_IGNORE_REGEX] = self.configuration.is_text_file_ignore_regex
            parameters[gc.PPN_TASK_IS_TEXT_FILE_TRANSLITERATE_MAP] = self.configuration.is_text_file_transliterate_map
            parameters[gc.PPN_TASK_IS_TEXT_UNPARSED_CLASS_REGEX] = self.configuration.is_text_unparsed_class_regex
            parameters[gc.PPN_TASK_IS_TEXT_UNPARSED_ID_REGEX] = self.configuration.is_text_unparsed_id_regex
            parameters[gc.PPN_TASK_IS_TEXT_UNPARSED_ID_SORT] = self.configuration.is_text_unparsed_id_sort
            parameters[gc.PPN_TASK_OS_FILE_ID_REGEX] = self.configuration.os_file_id_regex
            self.text_file = TextFile(
                file_path=self.text_file_path_absolute,
                file_format=self.configuration.is_text_file_format,
                parameters=parameters,
                logger=self.logger
            )
            self.text_file.set_language(self.configuration.language)
        else:
            self._log("text_file_path_absolute and/or language is None")
        self._log("Populate text file... done")



class TaskConfiguration(object):
    """
    A structure representing a configuration for a task, that is,
    a series of directives for I/O and processing the Task.

    :param config_string: the task configuration string
    :type  config_string: string

    :raises TypeError: if ``config_string`` is not ``None`` and not an instance of ``str`` or ``unicode``
    """

    TAG = "TaskConfiguration"

    def __init__(self, config_string=None):
        if (
                (config_string is not None) and
                (not isinstance(config_string, str)) and
                (not isinstance(config_string, unicode))
        ):
            raise TypeError("config_string is not an instance of str or unicode")
        # task fields
        self.field_names = [
            gc.PPN_TASK_DESCRIPTION,
            gc.PPN_TASK_LANGUAGE,
            gc.PPN_TASK_CUSTOM_ID,

            gc.PPN_TASK_ADJUST_BOUNDARY_ALGORITHM,
            gc.PPN_TASK_ADJUST_BOUNDARY_AFTERCURRENT_VALUE,
            gc.PPN_TASK_ADJUST_BOUNDARY_BEFORENEXT_VALUE,
            gc.PPN_TASK_ADJUST_BOUNDARY_OFFSET_VALUE,
            gc.PPN_TASK_ADJUST_BOUNDARY_PERCENT_VALUE,
            gc.PPN_TASK_ADJUST_BOUNDARY_RATE_VALUE,

            gc.PPN_TASK_IS_AUDIO_FILE_DETECT_HEAD_MIN,
            gc.PPN_TASK_IS_AUDIO_FILE_DETECT_HEAD_MAX,
            gc.PPN_TASK_IS_AUDIO_FILE_DETECT_TAIL_MIN,
            gc.PPN_TASK_IS_AUDIO_FILE_DETECT_TAIL_MAX,
            gc.PPN_TASK_IS_AUDIO_FILE_HEAD_LENGTH,
            gc.PPN_TASK_IS_AUDIO_FILE_PROCESS_LENGTH,
            gc.PPN_TASK_IS_AUDIO_FILE_TAIL_LENGTH,
            gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
            gc.PPN_TASK_IS_TEXT_FILE_IGNORE_REGEX,
            gc.PPN_TASK_IS_TEXT_FILE_TRANSLITERATE_MAP,
            gc.PPN_TASK_IS_TEXT_UNPARSED_CLASS_REGEX,
            gc.PPN_TASK_IS_TEXT_UNPARSED_ID_REGEX,
            gc.PPN_TASK_IS_TEXT_UNPARSED_ID_SORT,

            gc.PPN_TASK_OS_FILE_FORMAT,
            gc.PPN_TASK_OS_FILE_ID_REGEX,
            gc.PPN_TASK_OS_FILE_NAME,
            gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF,
            gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF,
            gc.PPN_TASK_OS_FILE_HEAD_TAIL_FORMAT
        ]
        self.fields = dict()
        for key in self.field_names:
            self.fields[key] = None

        # populate values from config_string
        if config_string is not None:
            properties = gf.config_string_to_dict(config_string)
            for key in properties:
                if key in self.field_names:
                    self.fields[key] = properties[key]

    def __str__(self):
        return "\n".join(["%s: '%s'" % (fn, self.fields[fn]) for fn in self.field_names])

    def config_string(self):
        """
        Build the storable string corresponding to this TaskConfiguration.

        :rtype: string
        """
        return (gc.CONFIG_STRING_SEPARATOR_SYMBOL).join(["%s%s%s" % (fn, gc.CONFIG_STRING_ASSIGNMENT_SYMBOL, self.fields[fn]) for fn in self.field_names if self.fields[fn] is not None])

    @property
    def description(self):
        """
        A human-readable description of the task.

        :rtype: string
        """
        return self.fields[gc.PPN_TASK_DESCRIPTION]
    @description.setter
    def description(self, value):
        self.fields[gc.PPN_TASK_DESCRIPTION] = value

    @property
    def language(self):
        """
        The language of the task.

        :rtype: string (from the :class:`aeneas.language.Language` enumeration)
        """
        return self.fields[gc.PPN_TASK_LANGUAGE]
    @language.setter
    def language(self, value):
        self.fields[gc.PPN_TASK_LANGUAGE] = value

    @property
    def custom_id(self):
        """
        A human-readable id for the task.

        :rtype: string
        """
        return self.fields[gc.PPN_TASK_CUSTOM_ID]
    @custom_id.setter
    def custom_id(self, value):
        self.fields[gc.PPN_TASK_CUSTOM_ID] = value

    @property
    def adjust_boundary_algorithm(self):
        """
        The algorithm to be run to adjust the fragment boundaries.
        If ``None``, keep the current boundaries.

        .. versionadded:: 1.0.4

        :rtype: string (from the :class:`aeneas.adjustboundaryalgorithm.AdjustBoundaryAlgorithm` enumeration)
        """
        return self.fields[gc.PPN_TASK_ADJUST_BOUNDARY_ALGORITHM]
    @adjust_boundary_algorithm.setter
    def adjust_boundary_algorithm(self, value):
        self.fields[gc.PPN_TASK_ADJUST_BOUNDARY_ALGORITHM] = value

    @property
    def adjust_boundary_aftercurrent_value(self):
        """
        The new boundary between two consecutive fragments
        will be set at ``value`` seconds
        after the end of the first fragment.

        .. versionadded:: 1.0.4

        :rtype: float
        """
        return self.fields[gc.PPN_TASK_ADJUST_BOUNDARY_AFTERCURRENT_VALUE]
    @adjust_boundary_aftercurrent_value.setter
    def adjust_boundary_aftercurrent_value(self, value):
        self.fields[gc.PPN_TASK_ADJUST_BOUNDARY_AFTERCURRENT_VALUE] = value

    @property
    def adjust_boundary_beforenext_value(self):
        """
        The new boundary between two consecutive fragments
        will be set at ``value`` seconds
        before the beginning of the second fragment.

        .. versionadded:: 1.0.4

        :rtype: float
        """
        return self.fields[gc.PPN_TASK_ADJUST_BOUNDARY_BEFORENEXT_VALUE]
    @adjust_boundary_beforenext_value.setter
    def adjust_boundary_beforenext_value(self, value):
        self.fields[gc.PPN_TASK_ADJUST_BOUNDARY_BEFORENEXT_VALUE] = value

    @property
    def adjust_boundary_offset_value(self):
        """
        The new boundary between two consecutive fragments
        will be set at ``value`` seconds from the current value.
        A negative ``value`` will move the boundary back,
        a positive ``value`` will move the boundary forward.

        .. versionadded:: 1.1.0

        :rtype: float
        """
        return self.fields[gc.PPN_TASK_ADJUST_BOUNDARY_OFFSET_VALUE]
    @adjust_boundary_offset_value.setter
    def adjust_boundary_offset_value(self, value):
        self.fields[gc.PPN_TASK_ADJUST_BOUNDARY_OFFSET_VALUE] = value

    @property
    def adjust_boundary_percent_value(self):
        """
        The new boundary between two consecutive fragments
        will be set at this ``value`` percent
        of the nonspeech interval between the two fragments.
        The value must be between ``0`` and ``100``.

        .. versionadded:: 1.0.4

        :rtype: int
        """
        return self.fields[gc.PPN_TASK_ADJUST_BOUNDARY_PERCENT_VALUE]
    @adjust_boundary_percent_value.setter
    def adjust_boundary_percent_value(self, value):
        self.fields[gc.PPN_TASK_ADJUST_BOUNDARY_PERCENT_VALUE] = value

    @property
    def adjust_boundary_rate_value(self):
        """
        The new boundary will be set trying to keep the rate
        of all the fragments below this ``value`` characters/second.
        The value must be greater than ``0``.

        .. versionadded:: 1.0.4

        :rtype: float
        """
        return self.fields[gc.PPN_TASK_ADJUST_BOUNDARY_RATE_VALUE]
    @adjust_boundary_rate_value.setter
    def adjust_boundary_rate_value(self, value):
        self.fields[gc.PPN_TASK_ADJUST_BOUNDARY_RATE_VALUE] = value

    @property
    def is_text_file_format(self):
        """
        The format of the input text file.

        :rtype: string (from the :class:`aeneas.textfile.TextFileFormat`)
        """
        return self.fields[gc.PPN_TASK_IS_TEXT_FILE_FORMAT]
    @is_text_file_format.setter
    def is_text_file_format(self, value):
        self.fields[gc.PPN_TASK_IS_TEXT_FILE_FORMAT] = value

    @property
    def is_text_file_ignore_regex(self):
        """
        The regex to match text to be ignored for alignment purposes.

        :rtype: regex
        """
        return self.fields[gc.PPN_TASK_IS_TEXT_FILE_IGNORE_REGEX]
    @is_text_file_ignore_regex.setter
    def is_text_file_ignore_regex(self, value):
        self.fields[gc.PPN_TASK_IS_TEXT_FILE_IGNORE_REGEX] = value

    @property
    def is_text_file_transliterate_map(self):
        """
        The path to the transliteration map file to be used to delete/replace
        characters in the input text file for alignment purposes.

        :rtype: string (path)
        """
        return self.fields[gc.PPN_TASK_IS_TEXT_FILE_TRANSLITERATE_MAP]
    @is_text_file_transliterate_map.setter
    def is_text_file_transliterate_map(self, value):
        self.fields[gc.PPN_TASK_IS_TEXT_FILE_TRANSLITERATE_MAP] = value

    @property
    def is_text_unparsed_class_regex(self):
        """
        The regex to match ``class`` attributes for text fragments.
        It applies to unparsed text files only.

        :rtype: string
        """
        return self.fields[gc.PPN_TASK_IS_TEXT_UNPARSED_CLASS_REGEX]
    @is_text_unparsed_class_regex.setter
    def is_text_unparsed_class_regex(self, value):
        self.fields[gc.PPN_TASK_IS_TEXT_UNPARSED_CLASS_REGEX] = value

    @property
    def is_text_unparsed_id_regex(self):
        """
        The regex to match ``id`` attributes for text fragments.
        It applies to unparsed text files only.

        :rtype: string
        """
        return self.fields[gc.PPN_TASK_IS_TEXT_UNPARSED_ID_REGEX]
    @is_text_unparsed_id_regex.setter
    def is_text_unparsed_id_regex(self, value):
        self.fields[gc.PPN_TASK_IS_TEXT_UNPARSED_ID_REGEX] = value

    @property
    def is_text_unparsed_id_sort(self):
        """
        The algorithm to sort text fragments by their ``id`` attributes.
        It applies to unparsed text files only.

        :rtype: string (from the :class:`aeneas.idsortingalgorithm.IDSortingAlgorithm` enumeration)
        """
        return self.fields[gc.PPN_TASK_IS_TEXT_UNPARSED_ID_SORT]
    @is_text_unparsed_id_sort.setter
    def is_text_unparsed_id_sort(self, value):
        self.fields[gc.PPN_TASK_IS_TEXT_UNPARSED_ID_SORT] = value

    @property
    def is_audio_file_detect_head_min(self):
        """
        When synchronizing, auto detect the head of the audio file,
        using the provided value as a lower bound, and disregard
        these many seconds from the beginning of the audio file.

        If the ``is_audio_file_head_length`` parameter is also provided,
        the auto detection will not take place.

        NOTE: This is an experimental feature, use with caution.

        :rtype: float

        .. versionadded:: 1.2.0
        """
        return self.fields[gc.PPN_TASK_IS_AUDIO_FILE_DETECT_HEAD_MIN]
    @is_audio_file_detect_head_min.setter
    def is_audio_file_detect_head_min(self, value):
        self.fields[gc.PPN_TASK_IS_AUDIO_FILE_DETECT_HEAD_MIN] = value

    @property
    def is_audio_file_detect_head_max(self):
        """
        When synchronizing, auto detect the head of the audio file,
        using the provided value as an upper bound, and disregard
        these many seconds from the beginning of the audio file.

        If the ``is_audio_file_head_length`` parameter is also provided,
        the auto detection will not take place.

        NOTE: This is an experimental feature, use with caution.

        :rtype: float

        .. versionadded:: 1.2.0
        """
        return self.fields[gc.PPN_TASK_IS_AUDIO_FILE_DETECT_HEAD_MAX]
    @is_audio_file_detect_head_max.setter
    def is_audio_file_detect_head_max(self, value):
        self.fields[gc.PPN_TASK_IS_AUDIO_FILE_DETECT_HEAD_MAX] = value

    @property
    def is_audio_file_detect_tail_min(self):
        """
        When synchronizing, auto detect the tail of the audio file,
        using the provided value as a lower bound, and disregard
        these many seconds from the end of the audio file.

        If the ``is_audio_file_process_length`` parameter or
        the ``is_audio_file_tail_length`` parameter
        are also provided,
        the auto detection will not take place.

        NOTE: This is an experimental feature, use with caution.

        :rtype: float

        .. versionadded:: 1.2.0
        """
        return self.fields[gc.PPN_TASK_IS_AUDIO_FILE_DETECT_TAIL_MIN]
    @is_audio_file_detect_tail_min.setter
    def is_audio_file_detect_tail_min(self, value):
        self.fields[gc.PPN_TASK_IS_AUDIO_FILE_DETECT_TAIL_MIN] = value

    @property
    def is_audio_file_detect_tail_max(self):
        """
        When synchronizing, auto detect the tail of the audio file,
        using the provided value as an upper bound, and disregard
        these many seconds from the end of the audio file.

        If the ``is_audio_file_process_length`` parameter or
        the ``is_audio_file_tail_length`` parameter
        are also provided,
        the auto detection will not take place.

        NOTE: This is an experimental feature, use with caution.

        :rtype: float

        .. versionadded:: 1.2.0
        """
        return self.fields[gc.PPN_TASK_IS_AUDIO_FILE_DETECT_TAIL_MAX]
    @is_audio_file_detect_tail_max.setter
    def is_audio_file_detect_tail_max(self, value):
        self.fields[gc.PPN_TASK_IS_AUDIO_FILE_DETECT_TAIL_MAX] = value

    @property
    def is_audio_file_head_length(self):
        """
        When synchronizing, disregard
        these many seconds from the beginning of the audio file.

        :rtype: float
        """
        return self.fields[gc.PPN_TASK_IS_AUDIO_FILE_HEAD_LENGTH]
    @is_audio_file_head_length.setter
    def is_audio_file_head_length(self, value):
        self.fields[gc.PPN_TASK_IS_AUDIO_FILE_HEAD_LENGTH] = value

    @property
    def is_audio_file_process_length(self):
        """
        When synchronizing, process only these many seconds
        from the audio file.

        :rtype: float
        """
        return self.fields[gc.PPN_TASK_IS_AUDIO_FILE_PROCESS_LENGTH]
    @is_audio_file_process_length.setter
    def is_audio_file_process_length(self, value):
        self.fields[gc.PPN_TASK_IS_AUDIO_FILE_PROCESS_LENGTH] = value

    @property
    def is_audio_file_tail_length(self):
        """
        When synchronizing, disregard
        these many seconds from the end of the audio file.

        Note that if the ``is_audio_file_process_length`` parameter
        is also provided, only the latter will be taken into account,
        and ``is_audio_file_tail_length`` will be ignored.

        :rtype: float

        .. versionadded:: 1.2.1
        """
        return self.fields[gc.PPN_TASK_IS_AUDIO_FILE_TAIL_LENGTH]
    @is_audio_file_tail_length.setter
    def is_audio_file_tail_length(self, value):
        self.fields[gc.PPN_TASK_IS_AUDIO_FILE_TAIL_LENGTH] = value

    @property
    def os_file_format(self):
        """
        The format of the sync map output for the task.

        :rtype: string (from :class:`aeneas.syncmap.SyncMapFormat`)
        """
        return self.fields[gc.PPN_TASK_OS_FILE_FORMAT]
    @os_file_format.setter
    def os_file_format(self, value):
        self.fields[gc.PPN_TASK_OS_FILE_FORMAT] = value

    @property
    def os_file_id_regex(self):
        """
        The regex to be used for the fragment identifiers
        of the sync map output file

        :rtype: string
        """
        return self.fields[gc.PPN_TASK_OS_FILE_ID_REGEX]
    @os_file_id_regex.setter
    def os_file_id_regex(self, value):
        self.fields[gc.PPN_TASK_OS_FILE_ID_REGEX] = value

    @property
    def os_file_name(self):
        """
        The name of the sync map file output for the task.

        :rtype: string
        """
        return self.fields[gc.PPN_TASK_OS_FILE_NAME]
    @os_file_name.setter
    def os_file_name(self, value):
        self.fields[gc.PPN_TASK_OS_FILE_NAME] = value

    @property
    def os_file_smil_audio_ref(self):
        """
        The value of the ``src`` attribute for the ``<audio>`` element
        in the output sync map.
        It applies to ``SMIL`` sync maps only.

        :rtype: string
        """
        return self.fields[gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF]
    @os_file_smil_audio_ref.setter
    def os_file_smil_audio_ref(self, value):
        self.fields[gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF] = value

    @property
    def os_file_smil_page_ref(self):
        """
        The value of the ``src`` attribute for the ``<text>`` element
        in the output sync map.
        It applies to ``SMIL`` sync maps only.

        :rtype: string
        """
        return self.fields[gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF]
    @os_file_smil_page_ref.setter
    def os_file_smil_page_ref(self, value):
        self.fields[gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF] = value

    @property
    def os_file_head_tail_format(self):
        """
        The format of the head and tail of the sync map output for the task.

        :rtype: string (from :class:`aeneas.syncmap.SyncMapHeadTailFormat`)
        """
        return self.fields[gc.PPN_TASK_OS_FILE_HEAD_TAIL_FORMAT]
    @os_file_head_tail_format.setter
    def os_file_head_tail_format(self, value):
        self.fields[gc.PPN_TASK_OS_FILE_HEAD_TAIL_FORMAT] = value

    #@property
    #def xxx(self):
    #    """
    #    TBW
    #
    #    :rtype: string
    #    """
    #    return self.fields[gc.KEY]
    #@xxx.setter
    #def xxx(self, value):
    #    self.fields[gc.KEY] = value



