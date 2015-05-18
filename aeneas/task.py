#!/usr/bin/env python
# coding=utf-8

"""
A structure representing a task, that is,
an audio file and a list of text fragments
to be synchronized.
"""

import os
import uuid

import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf
from aeneas.audiofile import AudioFile
from aeneas.textfile import TextFile

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl (www.readbeyond.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.0.1"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class Task(object):
    """
    A structure representing a task, that is,
    an audio file and a list of text fragments
    to be synchronized.

    :param config_string: the task configuration string
    :type  config_string: string
    """

    TAG = "Task"

    def __init__(self, config_string=None):
        # task properties
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
        if config_string != None:
            self.configuration = TaskConfiguration(config_string)

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

    def _populate_audio_file(self):
        """
        Create the ``self.audio_file`` object by reading
        the audio file at ``self.audio_file_path_absolute``.
        """
        if self.audio_file_path_absolute != None:
            self.audio_file = AudioFile(
                file_path=self.audio_file_path_absolute,
                logger=None
            )

    def _populate_text_file(self):
        """
        Create the ``self.text_file`` object by reading
        the text file at ``self.text_file_path_absolute``.
        """
        if ((self.text_file_path_absolute != None) and
                (self.configuration.language != None)):
            parameters = dict()
            parameters[gc.PPN_TASK_IS_TEXT_UNPARSED_CLASS_REGEX] = self.configuration.is_text_unparsed_class_regex
            parameters[gc.PPN_TASK_IS_TEXT_UNPARSED_ID_REGEX] = self.configuration.is_text_unparsed_id_regex
            parameters[gc.PPN_TASK_IS_TEXT_UNPARSED_ID_SORT] = self.configuration.is_text_unparsed_id_sort
            self.text_file = TextFile(
                file_path=self.text_file_path_absolute,
                file_format=self.configuration.is_text_file_format,
                parameters=parameters,
                logger=None
            )
            self.text_file.set_language(self.configuration.language)

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
        :rtype: return the path of the sync map file created
        """
        if self.sync_map == None:
            return None

        if (container_root_path != None) and (self.sync_map_file_path == None):
            return None

        if (container_root_path != None) and (self.sync_map_file_path != None):
            path = os.path.join(container_root_path, self.sync_map_file_path)
        elif self.sync_map_file_path_absolute:
            path = self.sync_map_file_path_absolute

        sync_map_format = self.configuration.os_file_format
        parameters = dict()
        parameters[gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF] = self.configuration.os_file_smil_page_ref
        parameters[gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF] = self.configuration.os_file_smil_audio_ref
        result = self.sync_map.output(sync_map_format, path, parameters)
        if not result:
            return None
        return path




class TaskConfiguration(object):
    """
    A structure representing a configuration for a task, that is,
    a series of directives for I/O and processing the Task.

    :param config_string: the task configuration string
    :type  config_string: string
    """

    TAG = "TaskConfiguration"

    def __init__(self, config_string=None):
        # task fields
        self.field_names = [
            gc.PPN_TASK_DESCRIPTION,
            gc.PPN_TASK_LANGUAGE,
            gc.PPN_TASK_CUSTOM_ID,

            gc.PPN_TASK_IS_AUDIO_FILE_HEAD_LENGTH,
            gc.PPN_TASK_IS_AUDIO_FILE_PROCESS_LENGTH,
            gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
            gc.PPN_TASK_IS_TEXT_UNPARSED_CLASS_REGEX,
            gc.PPN_TASK_IS_TEXT_UNPARSED_ID_REGEX,
            gc.PPN_TASK_IS_TEXT_UNPARSED_ID_SORT,

            gc.PPN_TASK_OS_FILE_FORMAT,
            gc.PPN_TASK_OS_FILE_NAME,
            gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF,
            gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF
        ]
        self.fields = dict()
        for key in self.field_names:
            self.fields[key] = None

        # populate values from config_string
        if config_string != None:
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
        return (gc.CONFIG_STRING_SEPARATOR_SYMBOL).join(["%s%s%s" % (fn, gc.CONFIG_STRING_ASSIGNMENT_SYMBOL, self.fields[fn]) for fn in self.field_names if self.fields[fn] != None])

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
    def is_audio_file_head_length(self):
        """
        When synchronizing, disregard
        these many seconds from the beginning of the audio file.

        NOTE: At the moment, no sanity check is performed on this value.
        Use with caution.

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

        NOTE: At the moment, no sanity check is performed on this value.
        Use with caution.

        :rtype: float
        """
        return self.fields[gc.PPN_TASK_IS_AUDIO_FILE_PROCESS_LENGTH]
    @is_audio_file_process_length.setter
    def is_audio_file_process_length(self, value):
        self.fields[gc.PPN_TASK_IS_AUDIO_FILE_PROCESS_LENGTH] = value

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

    #@property
    #def xxx(self):
    #    return self.fields[gc.KEY]
    #@xxx.setter
    #def xxx(self, value):
    #    self.fields[gc.KEY] = value



