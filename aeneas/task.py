#!/usr/bin/env python
# coding=utf-8

"""
A structure representing a task, that is,
an audio file and a list of text fragments
to be synchronized.
"""

from __future__ import absolute_import
from __future__ import print_function
import os

from aeneas.audiofile import AudioFile
from aeneas.configurationobject import ConfigurationObject
from aeneas.logger import Logger
from aeneas.textfile import TextFile
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf

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

class Task(object):
    """
    A structure representing a task, that is,
    an audio file and a list of text fragments
    to be synchronized.

    :param config_string: the task configuration string
    :type  config_string: string
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`

    :raises TypeError: if ``config_string`` is not ``None`` and
                       it is not a Unicode string
    """

    TAG = u"Task"

    def __init__(self, config_string=None, logger=None):
        self.logger = logger or Logger()
        self.identifier = gf.uuid_string()
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

    def __unicode__(self):
        msg = [
            u"%s: '%s'" % (gc.RPN_TASK_IDENTIFIER, self.identifier),
            u"Configuration:\n%s" % self.configuration.__unicode__(),
            u"Audio file path: %s" % self.audio_file_path,
            u"Audio file path (absolute): %s" % self.audio_file_path_absolute,
            u"Text file path: %s" % self.text_file_path,
            u"Text file path (absolute): %s" % self.text_file_path_absolute,
            u"Sync map file path: %s" % self.sync_map_file_path,
            u"Sync map file path (absolute): %s" % self.sync_map_file_path_absolute
        ]
        return u"\n".join(msg)

    def __str__(self):
        return gf.safe_str(self.__unicode__())

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
            self._log(u"sync_map is None", Logger.CRITICAL)
            raise TypeError("sync_map object has not been set")

        if (container_root_path is not None) and (self.sync_map_file_path is None):
            self._log(u"The (internal) path of the sync map has been set", Logger.CRITICAL)
            raise TypeError("The (internal) path of the sync map has been set")

        self._log([u"container_root_path is %s", container_root_path])
        self._log([u"self.sync_map_file_path is %s", self.sync_map_file_path])
        self._log([u"self.sync_map_file_path_absolute is %s", self.sync_map_file_path_absolute])

        if (container_root_path is not None) and (self.sync_map_file_path is not None):
            path = os.path.join(container_root_path, self.sync_map_file_path)
        elif self.sync_map_file_path_absolute:
            path = self.sync_map_file_path_absolute
        gf.ensure_parent_directory(path)
        self._log([u"Output sync map to %s", path])

        sync_map_format = self.configuration["o_format"]
        audio_ref = self.configuration["o_smil_audio_ref"]
        page_ref = self.configuration["o_smil_page_ref"]

        self._log([u"sync_map_format is %s", sync_map_format])
        self._log([u"page_ref is %s", page_ref])
        self._log([u"audio_ref is %s", audio_ref])

        self._log(u"Calling sync_map.write...")
        # TODO just pass self.configuration?
        parameters = {
            gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF : page_ref,
            gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF : audio_ref
        }
        self.sync_map.write(sync_map_format, path, parameters)
        self._log(u"Calling sync_map.write... done")
        return path

    def _populate_audio_file(self):
        """
        Create the ``self.audio_file`` object by reading
        the audio file at ``self.audio_file_path_absolute``.
        """
        self._log(u"Populate audio file...")
        if self.audio_file_path_absolute is not None:
            self._log([u"audio_file_path_absolute is '%s'", self.audio_file_path_absolute])
            self.audio_file = AudioFile(
                file_path=self.audio_file_path_absolute,
                logger=self.logger
            )
            self.audio_file.read_properties()
        else:
            self._log(u"audio_file_path_absolute is None")
        self._log(u"Populate audio file... done")

    def _populate_text_file(self):
        """
        Create the ``self.text_file`` object by reading
        the text file at ``self.text_file_path_absolute``.
        """
        self._log(u"Populate text file...")
        if (
                (self.text_file_path_absolute is not None) and
                (self.configuration["language"] is not None)
            ):
            # TODO just pass self.configuration?
            # the following values might be None
            parameters = {
                gc.PPN_TASK_IS_TEXT_FILE_IGNORE_REGEX : self.configuration["i_t_ignore_regex"],
                gc.PPN_TASK_IS_TEXT_FILE_TRANSLITERATE_MAP : self.configuration["i_t_transliterate_map"],
                gc.PPN_TASK_IS_TEXT_UNPARSED_CLASS_REGEX :self.configuration["i_t_unparsed_class_regex"],
                gc.PPN_TASK_IS_TEXT_UNPARSED_ID_REGEX : self.configuration["i_t_unparsed_id_regex"],
                gc.PPN_TASK_IS_TEXT_UNPARSED_ID_SORT : self.configuration["i_t_unparsed_id_sort"],
                gc.PPN_TASK_OS_FILE_ID_REGEX : self.configuration["o_id_regex"]
            }
            self.text_file = TextFile(
                file_path=self.text_file_path_absolute,
                file_format=self.configuration["i_t_format"],
                parameters=parameters,
                logger=self.logger
            )
            self.text_file.set_language(self.configuration["language"])
        else:
            self._log(u"text_file_path_absolute and/or language is None")
        self._log(u"Populate text file... done")



class TaskConfiguration(ConfigurationObject):
    """
    A structure representing a configuration for a task, that is,
    a series of directives for I/O and processing the task.

    Allowed keys:

    * ``PPN_TASK_CUSTOM_ID``                          or ``custom_id``
    * ``PPN_TASK_DESCRIPTION``                        or ``description``
    * ``PPN_TASK_LANGUAGE``                           or ``language``
    * ``PPN_TASK_ADJUST_BOUNDARY_AFTERCURRENT_VALUE`` or ``aba_aftercurrent_value``
    * ``PPN_TASK_ADJUST_BOUNDARY_ALGORITHM``          or ``aba_algorithm``
    * ``PPN_TASK_ADJUST_BOUNDARY_BEFORENEXT_VALUE``   or ``aba_beforenext_value``
    * ``PPN_TASK_ADJUST_BOUNDARY_OFFSET_VALUE``       or ``aba_offset_value``
    * ``PPN_TASK_ADJUST_BOUNDARY_PERCENT_VALUE``      or ``aba_percent_value``
    * ``PPN_TASK_ADJUST_BOUNDARY_RATE_VALUE``         or ``aba_rate_value``
    * ``PPN_TASK_IS_AUDIO_FILE_DETECT_HEAD_MAX``      or ``i_a_head_max``
    * ``PPN_TASK_IS_AUDIO_FILE_DETECT_HEAD_MIN``      or ``i_a_head_min``
    * ``PPN_TASK_IS_AUDIO_FILE_DETECT_TAIL_MAX``      or ``i_a_tail_max``
    * ``PPN_TASK_IS_AUDIO_FILE_DETECT_TAIL_MIN``      or ``i_a_tail_min``
    * ``PPN_TASK_IS_AUDIO_FILE_HEAD_LENGTH``          or ``i_a_head``
    * ``PPN_TASK_IS_AUDIO_FILE_PROCESS_LENGTH``       or ``i_a_process``
    * ``PPN_TASK_IS_AUDIO_FILE_TAIL_LENGTH``          or ``i_a_tail``
    * ``PPN_TASK_IS_TEXT_FILE_FORMAT``                or ``i_t_format``
    * ``PPN_TASK_IS_TEXT_FILE_IGNORE_REGEX``          or ``i_t_ignore_regex``
    * ``PPN_TASK_IS_TEXT_FILE_TRANSLITERATE_MAP``     or ``i_t_transliterate_map``
    * ``PPN_TASK_IS_TEXT_UNPARSED_CLASS_REGEX``       or ``i_t_unparsed_class_regex``
    * ``PPN_TASK_IS_TEXT_UNPARSED_ID_REGEX``          or ``i_t_unparsed_id_regex``
    * ``PPN_TASK_IS_TEXT_UNPARSED_ID_SORT``           or ``i_t_unparsed_id_sort``
    * ``PPN_TASK_OS_FILE_FORMAT``                     or ``o_format``
    * ``PPN_TASK_OS_FILE_HEAD_TAIL_FORMAT``           or ``o_h_t_format``
    * ``PPN_TASK_OS_FILE_ID_REGEX``                   or ``o_id_regex``
    * ``PPN_TASK_OS_FILE_NAME``                       or ``o_name``
    * ``PPN_TASK_OS_FILE_SMIL_AUDIO_REF``             or ``o_smil_audio_ref``
    * ``PPN_TASK_OS_FILE_SMIL_PAGE_REF``              or ``o_smil_page_ref``

    :param config_string: the job configuration string
    :type  config_string: Unicode string

    :raises TypeError: if ``config_string`` is not ``None`` and
                       it is not a Unicode string
    :raises KeyError: if trying to access a key not listed above
    """

    TAG = u"TaskConfiguration"

    FIELDS = [
        (gc.PPN_TASK_CUSTOM_ID, (None, None, ["custom_id"])),
        (gc.PPN_TASK_DESCRIPTION, (None, None, ["description"])),
        (gc.PPN_TASK_LANGUAGE, (None, None, ["language"])),
        (gc.PPN_TASK_ADJUST_BOUNDARY_AFTERCURRENT_VALUE, (None, float, ["aba_aftercurrent_value"])),
        (gc.PPN_TASK_ADJUST_BOUNDARY_ALGORITHM, (None, None, ["aba_algorithm"])),
        (gc.PPN_TASK_ADJUST_BOUNDARY_BEFORENEXT_VALUE, (None, float, ["aba_beforenext_value"])),
        (gc.PPN_TASK_ADJUST_BOUNDARY_OFFSET_VALUE, (None, float, ["aba_offset_value"])),
        (gc.PPN_TASK_ADJUST_BOUNDARY_PERCENT_VALUE, (None, int, ["aba_percent_value"])),
        (gc.PPN_TASK_ADJUST_BOUNDARY_RATE_VALUE, (None, float, ["aba_rate_value"])),
        (gc.PPN_TASK_IS_AUDIO_FILE_DETECT_HEAD_MAX, (None, float, ["i_a_head_max"])),
        (gc.PPN_TASK_IS_AUDIO_FILE_DETECT_HEAD_MIN, (None, float, ["i_a_head_min"])),
        (gc.PPN_TASK_IS_AUDIO_FILE_DETECT_TAIL_MAX, (None, float, ["i_a_tail_max"])),
        (gc.PPN_TASK_IS_AUDIO_FILE_DETECT_TAIL_MIN, (None, float, ["i_a_tail_min"])),
        (gc.PPN_TASK_IS_AUDIO_FILE_HEAD_LENGTH, (None, float, ["i_a_head"])),
        (gc.PPN_TASK_IS_AUDIO_FILE_PROCESS_LENGTH, (None, float, ["i_a_process"])),
        (gc.PPN_TASK_IS_AUDIO_FILE_TAIL_LENGTH, (None, float, ["i_a_tail"])),
        (gc.PPN_TASK_IS_TEXT_FILE_FORMAT, (None, None, ["i_t_format"])),
        (gc.PPN_TASK_IS_TEXT_FILE_IGNORE_REGEX, (None, None, ["i_t_ignore_regex"])),
        (gc.PPN_TASK_IS_TEXT_FILE_TRANSLITERATE_MAP, (None, None, ["i_t_transliterate_map"])),
        (gc.PPN_TASK_IS_TEXT_UNPARSED_CLASS_REGEX, (None, None, ["i_t_unparsed_class_regex"])),
        (gc.PPN_TASK_IS_TEXT_UNPARSED_ID_REGEX, (None, None, ["i_t_unparsed_id_regex"])),
        (gc.PPN_TASK_IS_TEXT_UNPARSED_ID_SORT, (None, None, ["i_t_unparsed_id_sort"])),
        (gc.PPN_TASK_OS_FILE_FORMAT, (None, None, ["o_format"])),
        (gc.PPN_TASK_OS_FILE_HEAD_TAIL_FORMAT, (None, None, ["o_h_t_format"])),
        (gc.PPN_TASK_OS_FILE_ID_REGEX, (None, None, ["o_id_regex"])),
        (gc.PPN_TASK_OS_FILE_NAME, (None, None, ["o_name"])),
        (gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF, (None, None, ["o_smil_audio_ref"])),
        (gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF, (None, None, ["o_smil_page_ref"])),
    ]

    def __init__(self, config_string=None):
        super(TaskConfiguration, self).__init__(config_string)



