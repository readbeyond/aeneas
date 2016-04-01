#!/usr/bin/env python
# coding=utf-8

"""
This module contains the following classes:

* :class:`~aeneas.job.Job`, representing a job;
* :class:`~aeneas.job.JobConfiguration`, representing a job configuration.
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.configuration import Configuration
from aeneas.logger import Loggable
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf

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

class Job(Loggable):
    """
    A structure representing a job, that is,
    a collection of related Tasks.

    :param string config_string: the job configuration string
    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    :raises: TypeError: if ``config_string`` is not ``None`` and
                        not a Unicode string
    """

    TAG = u"Job"

    def __init__(self, config_string=None, rconf=None, logger=None):
        super(Job, self).__init__(rconf=rconf, logger=logger)
        self.tasks = []
        self.identifier = gf.uuid_string()
        self.configuration = None if config_string is None else JobConfiguration(config_string)

    def __len__(self):
        return len(self.tasks)

    def __unicode__(self):
        i = 0
        msg = []
        msg.append(u"%s: '%s'" % (gc.RPN_JOB_IDENTIFIER, self.identifier))
        msg.append(u"Configuration:\n%s" % self.configuration.__unicode__())
        msg.append(u"Tasks:")
        for task in self.tasks:
            msg.append(u"Task %d %s" % (i, task.identifier))
            i += 1
        return u"\n".join(msg)

    def __str__(self):
        return gf.safe_str(self.__unicode__())

    def add_task(self, task):
        """
        Add a task to this job.

        :param task: the task to be added
        :type  task: :class:`~aeneas.task.Task`
        """
        self.tasks.append(task)

    def clear_tasks(self):
        """
        Delete all the tasks of this job.
        """
        self.tasks = []

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



class JobConfiguration(Configuration):
    """
    A structure representing a configuration for a job, that is,
    a series of directives for I/O and processing the job.

    Allowed keys:

    * :data:`~aeneas.globalconstants.PPN_JOB_DESCRIPTION`                   or ``description``
    * :data:`~aeneas.globalconstants.PPN_JOB_LANGUAGE`                      or ``language``
    * :data:`~aeneas.globalconstants.PPN_JOB_IS_AUDIO_FILE_NAME_REGEX`      or ``i_a_name_regex``
    * :data:`~aeneas.globalconstants.PPN_JOB_IS_AUDIO_FILE_RELATIVE_PATH`   or ``i_a_relative_path``
    * :data:`~aeneas.globalconstants.PPN_JOB_IS_HIERARCHY_PREFIX`           or ``i_hierarchy_prefix``
    * :data:`~aeneas.globalconstants.PPN_JOB_IS_HIERARCHY_TYPE`             or ``i_hierarchy_type``
    * :data:`~aeneas.globalconstants.PPN_JOB_IS_TASK_DIRECTORY_NAME_REGEX`  or ``i_task_directory_name_regex``
    * :data:`~aeneas.globalconstants.PPN_JOB_IS_TEXT_FILE_NAME_REGEX`       or ``i_t_name_regex``
    * :data:`~aeneas.globalconstants.PPN_JOB_IS_TEXT_FILE_RELATIVE_PATH`    or ``i_t_relative_path``
    * :data:`~aeneas.globalconstants.PPN_JOB_OS_CONTAINER_FORMAT`           or ``o_container_format``
    * :data:`~aeneas.globalconstants.PPN_JOB_OS_FILE_NAME`                  or ``o_name``
    * :data:`~aeneas.globalconstants.PPN_JOB_OS_HIERARCHY_PREFIX`           or ``o_hierarchy_prefix``
    * :data:`~aeneas.globalconstants.PPN_JOB_OS_HIERARCHY_TYPE`             or ``o_hierarchy_type``
    * :data:`~aeneas.globalconstants.PPN_JOB_IS_TEXT_FILE_FORMAT`           or ``i_t_format``
    * :data:`~aeneas.globalconstants.PPN_JOB_IS_TEXT_MUNPARSED_L1_ID_REGEX` or ``i_t_munparsed_l1_id_regex``
    * :data:`~aeneas.globalconstants.PPN_JOB_IS_TEXT_MUNPARSED_L2_ID_REGEX` or ``i_t_munparsed_l2_id_regex``
    * :data:`~aeneas.globalconstants.PPN_JOB_IS_TEXT_MUNPARSED_L3_ID_REGEX` or ``i_t_munparsed_l3_id_regex``
    * :data:`~aeneas.globalconstants.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX`  or ``i_t_unparsed_class_regex``
    * :data:`~aeneas.globalconstants.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX`     or ``i_t_unparsed_id_regex``
    * :data:`~aeneas.globalconstants.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT`      or ``i_t_unparsed_id_sort``

    :param string config_string: the job configuration string
    :raises: TypeError: if ``config_string`` is not ``None`` and
                        it is not a Unicode string
    :raises: KeyError: if trying to access a key not listed above
    """

    FIELDS = [
        (gc.PPN_JOB_DESCRIPTION, (None, None, ["description"])),
        (gc.PPN_JOB_LANGUAGE, (None, None, ["language"])),
        (gc.PPN_JOB_IS_AUDIO_FILE_NAME_REGEX, (None, None, ["i_a_name_regex"])),
        (gc.PPN_JOB_IS_AUDIO_FILE_RELATIVE_PATH, (None, None, ["i_a_relative_path"])),
        (gc.PPN_JOB_IS_HIERARCHY_PREFIX, (None, None, ["i_hierarchy_prefix"])),
        (gc.PPN_JOB_IS_HIERARCHY_TYPE, (None, None, ["i_hierarchy_type"])),
        (gc.PPN_JOB_IS_TASK_DIRECTORY_NAME_REGEX, (None, None, ["i_task_directory_name_regex"])),
        (gc.PPN_JOB_IS_TEXT_FILE_NAME_REGEX, (None, None, ["i_t_name_regex"])),
        (gc.PPN_JOB_IS_TEXT_FILE_RELATIVE_PATH, (None, None, ["i_t_relative_path"])),
        (gc.PPN_JOB_OS_CONTAINER_FORMAT, (None, None, ["o_container_format"])),
        (gc.PPN_JOB_OS_FILE_NAME, (None, None, ["o_name"])),
        (gc.PPN_JOB_OS_HIERARCHY_PREFIX, (None, None, ["o_hierarchy_prefix"])),
        (gc.PPN_JOB_OS_HIERARCHY_TYPE, (None, None, ["o_hierarchy_type"])),
        # TODO the following ones should probably go
        (gc.PPN_JOB_IS_TEXT_FILE_FORMAT, (None, None, ["i_t_format"])),
        (gc.PPN_JOB_IS_TEXT_MUNPARSED_L1_ID_REGEX, (None, None, ["i_t_munparsed_l1_id_regex"])),
        (gc.PPN_JOB_IS_TEXT_MUNPARSED_L2_ID_REGEX, (None, None, ["i_t_munparsed_l2_id_regex"])),
        (gc.PPN_JOB_IS_TEXT_MUNPARSED_L3_ID_REGEX, (None, None, ["i_t_munparsed_l3_id_regex"])),
        (gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX, (None, None, ["i_t_unparsed_class_regex"])),
        (gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX, (None, None, ["i_t_unparsed_id_regex"])),
        (gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT, (None, None, ["i_t_unparsed_id_sort"])),
    ]

    TAG = u"JobConfiguration"

    def __init__(self, config_string=None):
        super(JobConfiguration, self).__init__(config_string)



