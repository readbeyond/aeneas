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

* :class:`~aeneas.job.Job`, representing a job;
* :class:`~aeneas.job.JobConfiguration`, representing a job configuration.
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.configuration import Configuration
from aeneas.logger import Loggable
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf


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

    :param string config_string: the job configuration string
    :raises: TypeError: if ``config_string`` is not ``None`` and
                        it is not a Unicode string
    :raises: KeyError: if trying to access a key not listed above
    """

    FIELDS = [
        (gc.PPN_JOB_DESCRIPTION, (None, None, ["description"], u"description")),
        (gc.PPN_JOB_LANGUAGE, (None, None, ["language"], u"language")),
        (gc.PPN_JOB_IS_AUDIO_FILE_NAME_REGEX, (None, None, ["i_a_name_regex"], u"regex to find audio files")),
        (gc.PPN_JOB_IS_AUDIO_FILE_RELATIVE_PATH, (None, None, ["i_a_relative_path"], u"relative path of audio files")),
        (gc.PPN_JOB_IS_HIERARCHY_PREFIX, (None, None, ["i_hierarchy_prefix"], u"prefix inside the input container")),
        (gc.PPN_JOB_IS_HIERARCHY_TYPE, (None, None, ["i_hierarchy_type"], u"type of input container hierarchy")),
        (gc.PPN_JOB_IS_TASK_DIRECTORY_NAME_REGEX, (None, None, ["i_task_directory_name_regex"], u"regex to find task dirs")),
        (gc.PPN_JOB_IS_TEXT_FILE_NAME_REGEX, (None, None, ["i_t_name_regex"], u"regex to find text files")),
        (gc.PPN_JOB_IS_TEXT_FILE_RELATIVE_PATH, (None, None, ["i_t_relative_path"], u"relative path of text files")),
        (gc.PPN_JOB_OS_CONTAINER_FORMAT, (None, None, ["o_container_format"], u"format of input container")),
        (gc.PPN_JOB_OS_FILE_NAME, (None, None, ["o_name"], u"name of output container")),
        (gc.PPN_JOB_OS_HIERARCHY_PREFIX, (None, None, ["o_hierarchy_prefix"], u"prefix inside the output container")),
        (gc.PPN_JOB_OS_HIERARCHY_TYPE, (None, None, ["o_hierarchy_type"], u"type of output container hierarchy")),
    ]

    TAG = u"JobConfiguration"

    def __init__(self, config_string=None):
        super(JobConfiguration, self).__init__(config_string)
