#!/usr/bin/env python
# coding=utf-8

"""
A structure representing a job, that is,
a collection of related Tasks.
"""

import uuid

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

class Job(object):
    """
    A structure representing a job, that is,
    a collection of related Tasks.

    :param config_string: the job configuration string
    :type  config_string: string

    :raises TypeError: if ``config_string`` is not ``None`` and not an instance of ``str`` or ``unicode``
    """

    TAG = "Job"

    def __init__(self, config_string=None):
        self.tasks = []
        self.identifier = str(uuid.uuid4()).lower()
        self.configuration = None
        if config_string is not None:
            self.configuration = JobConfiguration(config_string)

    def append_task(self, task):
        """
        Append a task to this job.

        :param task: the task to be appended
        :type  task: :class:`aeneas.task.Task`
        """
        self.tasks.append(task)

    def clear_tasks(self):
        """
        Delete all the tasks of this job.
        """
        self.tasks = []

    def __len__(self):
        return len(self.tasks)

    def __str__(self):
        accumulator = ""
        accumulator += "%s: '%s'\n" % (gc.RPN_JOB_IDENTIFIER, self.identifier)
        accumulator += "Configuration:\n%s\n" % str(self.configuration)
        i = 0
        accumulator += "Tasks:\n"
        for task in self.tasks:
            accumulator += "Task %d %s\n" % (i, task.identifier)
            i += 1
        return accumulator

    @property
    def identifier(self):
        """
        The identifier of the job.

        :rtype: string
        """
        return self.__identifier
    @identifier.setter
    def identifier(self, value):
        self.__identifier = value


class JobConfiguration(object):
    """
    A structure representing a configuration for a job, that is,
    a series of directives for I/O and processing the job.

    :param config_string: the job configuration string
    :type  config_string: string

    :raises TypeError: if ``config_string`` is not ``None`` and not an instance of ``str`` or ``unicode``
    """

    TAG = "JobConfiguration"

    def __init__(self, config_string=None):
        if (
                (config_string is not None) and
                (not isinstance(config_string, str)) and
                (not isinstance(config_string, unicode))
        ):
            raise TypeError("config_string is not an instance of str or unicode")
        # job fields
        self.field_names = [
            gc.PPN_JOB_DESCRIPTION,
            gc.PPN_JOB_LANGUAGE,

            gc.PPN_JOB_IS_AUDIO_FILE_NAME_REGEX,
            gc.PPN_JOB_IS_AUDIO_FILE_RELATIVE_PATH,
            gc.PPN_JOB_IS_HIERARCHY_PREFIX,
            gc.PPN_JOB_IS_HIERARCHY_TYPE,
            gc.PPN_JOB_IS_TASK_DIRECTORY_NAME_REGEX,
            gc.PPN_JOB_IS_TEXT_FILE_FORMAT,
            gc.PPN_JOB_IS_TEXT_FILE_NAME_REGEX,
            gc.PPN_JOB_IS_TEXT_FILE_RELATIVE_PATH,
            gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX,
            gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX,
            gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT,

            gc.PPN_JOB_OS_FILE_NAME,
            gc.PPN_JOB_OS_CONTAINER_FORMAT,
            gc.PPN_JOB_OS_HIERARCHY_TYPE,
            gc.PPN_JOB_OS_HIERARCHY_PREFIX,
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
        Build the storable string corresponding
        to this job configuration object.

        :rtype: string
        """
        return (gc.CONFIG_STRING_SEPARATOR_SYMBOL).join(["%s%s%s" % (fn, gc.CONFIG_STRING_ASSIGNMENT_SYMBOL, self.fields[fn]) for fn in self.field_names if self.fields[fn] is not None])

    @property
    def description(self):
        """
        A human-readable description of the job.

        :rtype: string
        """
        return self.fields[gc.PPN_JOB_DESCRIPTION]
    @description.setter
    def description(self, value):
        self.fields[gc.PPN_JOB_DESCRIPTION] = value

    @property
    def language(self):
        """
        The language of the job.

        :rtype: string (from the :class:`aeneas.language.Language` enumeration)
        """
        return self.fields[gc.PPN_JOB_LANGUAGE]
    @language.setter
    def language(self, value):
        self.fields[gc.PPN_JOB_LANGUAGE] = value

    @property
    def is_audio_file_name_regex(self):
        """
        The regex to match audio files in this job.

        :rtype: string
        """
        return self.fields[gc.PPN_JOB_IS_AUDIO_FILE_NAME_REGEX]
    @is_audio_file_name_regex.setter
    def is_audio_file_name_regex(self, value):
        self.fields[gc.PPN_JOB_IS_AUDIO_FILE_NAME_REGEX] = value

    @property
    def is_audio_file_relative_path(self):
        """
        The relative path of each audio file
        with respect to the base task directory.

        :rtype: string (path)
        """
        return self.fields[gc.PPN_JOB_IS_AUDIO_FILE_RELATIVE_PATH]
    @is_audio_file_relative_path.setter
    def is_audio_file_relative_path(self, value):
        self.fields[gc.PPN_JOB_IS_AUDIO_FILE_RELATIVE_PATH] = value

    @property
    def is_hierarchy_prefix(self):
        """
        The path, inside the input job container,
        of the base task directory.

        :rtype: string (path)
        """
        return self.fields[gc.PPN_JOB_IS_HIERARCHY_PREFIX]
    @is_hierarchy_prefix.setter
    def is_hierarchy_prefix(self, value):
        self.fields[gc.PPN_JOB_IS_HIERARCHY_PREFIX] = value

    @property
    def is_hierarchy_type(self):
        """
        The type of hierarchy of the input job container.

        :rtype: string (from :class:`aeneas.hierarchytype.HierarchyType` enumeration)
        """
        return self.fields[gc.PPN_JOB_IS_HIERARCHY_TYPE]
    @is_hierarchy_type.setter
    def is_hierarchy_type(self, value):
        self.fields[gc.PPN_JOB_IS_HIERARCHY_TYPE] = value

    @property
    def is_task_directory_name_regex(self):
        """
        The regex to match task directory names
        within the base task directory.
        Applies to paged hierarchies only.

        :rtype: string
        """
        return self.fields[gc.PPN_JOB_IS_TASK_DIRECTORY_NAME_REGEX]
    @is_task_directory_name_regex.setter
    def is_task_directory_name_regex(self, value):
        self.fields[gc.PPN_JOB_IS_TASK_DIRECTORY_NAME_REGEX] = value

    @property
    def is_text_file_format(self):
        """
        The text file format of the input file texts.

        :rtype: string (from the :class:`aeneas.textfile.TextFileFormat`)
        """
        return self.fields[gc.PPN_JOB_IS_TEXT_FILE_FORMAT]
    @is_text_file_format.setter
    def is_text_file_format(self, value):
        self.fields[gc.PPN_JOB_IS_TEXT_FILE_FORMAT] = value

    @property
    def is_text_file_name_regex(self):
        """
        The regex to match text files in this job.

        :rtype: string
        """
        return self.fields[gc.PPN_JOB_IS_TEXT_FILE_NAME_REGEX]
    @is_text_file_name_regex.setter
    def is_text_file_name_regex(self, value):
        self.fields[gc.PPN_JOB_IS_TEXT_FILE_NAME_REGEX] = value

    @property
    def is_text_file_relative_path(self):
        """
        The relative path of each text file
        with respect to the base task directory.

        :rtype: string
        """
        return self.fields[gc.PPN_JOB_IS_TEXT_FILE_RELATIVE_PATH]
    @is_text_file_relative_path.setter
    def is_text_file_relative_path(self, value):
        self.fields[gc.PPN_JOB_IS_TEXT_FILE_RELATIVE_PATH] = value

    @property
    def is_text_unparsed_class_regex(self):
        """
        The regex to match ``class`` attributes for text fragments.
        It applies to unparsed text files only.

        :rtype: string
        """
        return self.fields[gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX]
    @is_text_unparsed_class_regex.setter
    def is_text_unparsed_class_regex(self, value):
        self.fields[gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX] = value

    @property
    def is_text_unparsed_id_regex(self):
        """
        The regex to match ``id`` attributes for text fragments.
        It applies to unparsed text files only.

        :rtype: string
        """
        return self.fields[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX]
    @is_text_unparsed_id_regex.setter
    def is_text_unparsed_id_regex(self, value):
        self.fields[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX] = value

    @property
    def is_text_unparsed_id_sort(self):
        """
        The algorithm to sort text fragments by their ``id`` attributes.
        It applies to unparsed text files only.

        :rtype: string (from the :class:`aeneas.idsortingalgorithm.IDSortingAlgorithm` enumeration)
        """
        return self.fields[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT]
    @is_text_unparsed_id_sort.setter
    def is_text_unparsed_id_sort(self, value):
        self.fields[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT] = value

    @property
    def os_file_name(self):
        """
        The file name for the output container for the job.

        :rtype: string
        """
        return self.fields[gc.PPN_JOB_OS_FILE_NAME]
    @os_file_name.setter
    def os_file_name(self, value):
        self.fields[gc.PPN_JOB_OS_FILE_NAME] = value

    @property
    def os_container_format(self):
        """
        The format for the output container for the job.

        :rtype: string (from the :class:`aeneas.container.ContainerFormat` enumeration)
        """
        return self.fields[gc.PPN_JOB_OS_CONTAINER_FORMAT]
    @os_container_format.setter
    def os_container_format(self, value):
        self.fields[gc.PPN_JOB_OS_CONTAINER_FORMAT] = value

    @property
    def os_hierarchy_type(self):
        """
        The type of hierarchy of the output job container.

        :rtype: string (from :class:`aeneas.hierarchytype.HierarchyType` enumeration)
        """
        return self.fields[gc.PPN_JOB_OS_HIERARCHY_TYPE]
    @os_hierarchy_type.setter
    def os_hierarchy_type(self, value):
        self.fields[gc.PPN_JOB_OS_HIERARCHY_TYPE] = value

    @property
    def os_hierarchy_prefix(self):
        """
        The path, inside the input job container,
        of the base task directory.

        :rtype: string (path)
        """
        return self.fields[gc.PPN_JOB_OS_HIERARCHY_PREFIX]
    @os_hierarchy_prefix.setter
    def os_hierarchy_prefix(self, value):
        self.fields[gc.PPN_JOB_OS_HIERARCHY_PREFIX] = value

    #@property
    #def xxx(self):
    #    return self.fields[gc.KEY]
    #@xxx.setter
    #def xxx(self, value):
    #    self.fields[gc.KEY] = value



