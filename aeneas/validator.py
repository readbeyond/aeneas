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

* :class:`~aeneas.validator.Validator`, assessing whether user input is well-formed;
* :class:`~aeneas.validator.ValidatorResult`, a record holding validation result and possibly messages.
"""

from __future__ import absolute_import
from __future__ import print_function
import io

from aeneas.analyzecontainer import AnalyzeContainer
from aeneas.container import Container
from aeneas.container import ContainerFormat
from aeneas.executetask import AdjustBoundaryAlgorithm
from aeneas.hierarchytype import HierarchyType
from aeneas.idsortingalgorithm import IDSortingAlgorithm
from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.syncmap import SyncMapFormat
from aeneas.syncmap import SyncMapHeadTailFormat
from aeneas.textfile import TextFileFormat
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf


class Validator(Loggable):
    """
    A validator to assess whether user input is well-formed.

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    ALLOWED_VALUES = [
        #
        # NOTE disabling the check on language since now we support multiple TTS
        # COMMENTED (
        # COMMENTED    gc.PPN_JOB_LANGUAGE,
        # COMMENTED    Language.ALLOWED_VALUES
        # COMMENTED ),
        # COMMENTED (
        # COMMENTED     gc.PPN_TASK_LANGUAGE,
        # COMMENTED     Language.ALLOWED_VALUES
        # COMMENTED ),
        #
        (
            gc.PPN_JOB_IS_HIERARCHY_TYPE,
            HierarchyType.ALLOWED_VALUES
        ),
        (
            gc.PPN_JOB_OS_CONTAINER_FORMAT,
            ContainerFormat.ALLOWED_VALUES
        ),
        (
            gc.PPN_JOB_OS_HIERARCHY_TYPE,
            HierarchyType.ALLOWED_VALUES
        ),
        (
            gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
            TextFileFormat.ALLOWED_VALUES
        ),
        (
            gc.PPN_TASK_OS_FILE_FORMAT,
            SyncMapFormat.ALLOWED_VALUES
        ),
        (
            gc.PPN_TASK_IS_TEXT_UNPARSED_ID_SORT,
            IDSortingAlgorithm.ALLOWED_VALUES
        ),
        (
            gc.PPN_TASK_ADJUST_BOUNDARY_ALGORITHM,
            AdjustBoundaryAlgorithm.ALLOWED_VALUES
        ),
        (
            gc.PPN_TASK_OS_FILE_HEAD_TAIL_FORMAT,
            SyncMapHeadTailFormat.ALLOWED_VALUES
        )
    ]

    IMPLIED_PARAMETERS = [
        (
            # is_hierarchy_type=paged => is_task_dir_name_regex
            gc.PPN_JOB_IS_HIERARCHY_TYPE,
            [HierarchyType.PAGED],
            [gc.PPN_JOB_IS_TASK_DIRECTORY_NAME_REGEX]
        ),
        (
            # is_text_type=unparsed => is_text_unparsed_id_sort
            gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
            [TextFileFormat.UNPARSED],
            [gc.PPN_TASK_IS_TEXT_UNPARSED_ID_SORT]
        ),
        (
            # is_text_type=munparsed => is_text_munparsed_l1_id_regex
            gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
            [TextFileFormat.MUNPARSED],
            [gc.PPN_TASK_IS_TEXT_MUNPARSED_L1_ID_REGEX]
        ),
        (
            # is_text_type=munparsed => is_text_munparsed_l2_id_regex
            gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
            [TextFileFormat.MUNPARSED],
            [gc.PPN_TASK_IS_TEXT_MUNPARSED_L2_ID_REGEX]
        ),
        (
            # is_text_type=munparsed => is_text_munparsed_l3_id_regex
            gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
            [TextFileFormat.MUNPARSED],
            [gc.PPN_TASK_IS_TEXT_MUNPARSED_L3_ID_REGEX]
        ),
        (
            # is_text_type=unparsed => is_text_unparsed_class_regex or
            #                          is_text_unparsed_id_regex
            gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
            [TextFileFormat.UNPARSED],
            [
                gc.PPN_TASK_IS_TEXT_UNPARSED_CLASS_REGEX,
                gc.PPN_TASK_IS_TEXT_UNPARSED_ID_REGEX
            ]
        ),
        (
            # os_task_file_format=smil  => os_task_file_smil_audio_ref
            # os_task_file_format=smilh => os_task_file_smil_audio_ref
            # os_task_file_format=smilm => os_task_file_smil_audio_ref
            gc.PPN_TASK_OS_FILE_FORMAT,
            [SyncMapFormat.SMIL, SyncMapFormat.SMILH, SyncMapFormat.SMILM],
            [gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF]
        ),
        (
            # os_task_file_format=smil  => os_task_file_smil_page_ref
            # os_task_file_format=smilh => os_task_file_smil_page_ref
            # os_task_file_format=smilm => os_task_file_smil_page_ref
            gc.PPN_TASK_OS_FILE_FORMAT,
            [SyncMapFormat.SMIL, SyncMapFormat.SMILH, SyncMapFormat.SMILM],
            [gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF]
        ),
        (
            # task_adjust_boundary_algorithm=percent => task_adjust_boundary_percent_value
            gc.PPN_TASK_ADJUST_BOUNDARY_ALGORITHM,
            [AdjustBoundaryAlgorithm.PERCENT],
            [gc.PPN_TASK_ADJUST_BOUNDARY_PERCENT_VALUE]
        ),
        (
            # task_adjust_boundary_algorithm=rate => task_adjust_boundary_rate_value
            gc.PPN_TASK_ADJUST_BOUNDARY_ALGORITHM,
            [AdjustBoundaryAlgorithm.RATE],
            [gc.PPN_TASK_ADJUST_BOUNDARY_RATE_VALUE]
        ),
        (
            # task_adjust_boundary_algorithm=rate_aggressive => task_adjust_boundary_rate_value
            gc.PPN_TASK_ADJUST_BOUNDARY_ALGORITHM,
            [AdjustBoundaryAlgorithm.RATEAGGRESSIVE],
            [gc.PPN_TASK_ADJUST_BOUNDARY_RATE_VALUE]
        ),
        (
            # task_adjust_boundary_algorithm=currentend => task_adjust_boundary_currentend_value
            gc.PPN_TASK_ADJUST_BOUNDARY_ALGORITHM,
            [AdjustBoundaryAlgorithm.AFTERCURRENT],
            [gc.PPN_TASK_ADJUST_BOUNDARY_AFTERCURRENT_VALUE]
        ),
        (
            # task_adjust_boundary_algorithm=rate => task_adjust_boundary_nextstart_value
            gc.PPN_TASK_ADJUST_BOUNDARY_ALGORITHM,
            [AdjustBoundaryAlgorithm.BEFORENEXT],
            [gc.PPN_TASK_ADJUST_BOUNDARY_BEFORENEXT_VALUE]
        ),
        (
            # task_adjust_boundary_algorithm=offset => task_adjust_boundary_offset_value
            gc.PPN_TASK_ADJUST_BOUNDARY_ALGORITHM,
            [AdjustBoundaryAlgorithm.OFFSET],
            [gc.PPN_TASK_ADJUST_BOUNDARY_OFFSET_VALUE]
        )
    ]

    JOB_REQUIRED_PARAMETERS = [
        gc.PPN_JOB_LANGUAGE,
        gc.PPN_JOB_OS_CONTAINER_FORMAT,
        gc.PPN_JOB_OS_FILE_NAME,
    ]

    TASK_REQUIRED_PARAMETERS = [
        gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
        gc.PPN_TASK_LANGUAGE,
        gc.PPN_TASK_OS_FILE_FORMAT,
        gc.PPN_TASK_OS_FILE_NAME,
    ]

    TASK_REQUIRED_PARAMETERS_EXTERNAL_NAME = [
        gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
        gc.PPN_TASK_LANGUAGE,
        gc.PPN_TASK_OS_FILE_FORMAT,
    ]

    TXT_REQUIRED_PARAMETERS = [
        gc.PPN_JOB_IS_AUDIO_FILE_NAME_REGEX,
        gc.PPN_JOB_IS_AUDIO_FILE_RELATIVE_PATH,
        gc.PPN_JOB_IS_HIERARCHY_PREFIX,
        gc.PPN_JOB_IS_HIERARCHY_TYPE,
        gc.PPN_JOB_IS_TEXT_FILE_NAME_REGEX,
        gc.PPN_JOB_IS_TEXT_FILE_RELATIVE_PATH,
        gc.PPN_JOB_LANGUAGE,
        gc.PPN_JOB_OS_CONTAINER_FORMAT,
        gc.PPN_JOB_OS_FILE_NAME,
        gc.PPN_JOB_OS_HIERARCHY_PREFIX,
        gc.PPN_JOB_OS_HIERARCHY_TYPE,
        gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
        gc.PPN_TASK_OS_FILE_FORMAT,
        gc.PPN_TASK_OS_FILE_NAME,
    ]

    XML_JOB_REQUIRED_PARAMETERS = [
        gc.PPN_JOB_OS_CONTAINER_FORMAT,
        gc.PPN_JOB_OS_FILE_NAME,
        gc.PPN_JOB_OS_HIERARCHY_PREFIX,
        gc.PPN_JOB_OS_HIERARCHY_TYPE,
    ]

    XML_TASK_REQUIRED_PARAMETERS = [
        gc.PPN_TASK_IS_AUDIO_FILE_XML,
        gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
        gc.PPN_TASK_IS_TEXT_FILE_XML,
        gc.PPN_TASK_LANGUAGE,
        gc.PPN_TASK_OS_FILE_FORMAT,
        gc.PPN_TASK_OS_FILE_NAME,
    ]

    TAG = u"Validator"

    def __init__(self, rconf=None, logger=None):
        super(Validator, self).__init__(rconf=rconf, logger=logger)
        self.result = None

    def check_file_encoding(self, input_file_path):
        """
        Check whether the given file is UTF-8 encoded.

        :param string input_file_path: the path of the file to be checked
        :rtype: :class:`~aeneas.validator.ValidatorResult`
        """
        self.log([u"Checking encoding of file '%s'", input_file_path])
        self.result = ValidatorResult()
        if self._are_safety_checks_disabled(u"check_file_encoding"):
            return self.result
        if not gf.file_can_be_read(input_file_path):
            self._failed(u"File '%s' cannot be read." % (input_file_path))
            return self.result
        with io.open(input_file_path, "rb") as file_object:
            bstring = file_object.read()
            self._check_utf8_encoding(bstring)
        return self.result

    def check_raw_string(self, string, is_bstring=True):
        """
        Check whether the given string
        is properly UTF-8 encoded (if ``is_bytes`` is ``True``),
        it is not empty, and
        it does not contain reserved characters.

        :param string string: the byte string or Unicode string to be checked
        :param bool is_bstring: if True, string is a byte string
        :rtype: :class:`~aeneas.validator.ValidatorResult`
        """
        self.log(u"Checking the given byte string")
        self.result = ValidatorResult()
        if self._are_safety_checks_disabled(u"check_raw_string"):
            return self.result
        if is_bstring:
            self._check_utf8_encoding(string)
            if not self.result.passed:
                return self.result
            string = gf.safe_unicode(string)
        self._check_not_empty(string)
        if not self.result.passed:
            return self.result
        self._check_reserved_characters(string)
        return self.result

    def check_configuration_string(
            self,
            config_string,
            is_job=True,
            external_name=False
    ):
        """
        Check whether the given job or task configuration string
        is well-formed (if ``is_bstring`` is ``True``)
        and it has all the required parameters.

        :param string config_string: the byte string or Unicode string to be checked
        :param bool is_job: if ``True``, ``config_string`` is a job config string
        :param bool external_name: if ``True``, the task name is provided externally,
                                   and it is not required to appear
                                   in the config string
        :rtype: :class:`~aeneas.validator.ValidatorResult`
        """
        if is_job:
            self.log(u"Checking job configuration string")
        else:
            self.log(u"Checking task configuration string")
        self.result = ValidatorResult()
        if self._are_safety_checks_disabled(u"check_configuration_string"):
            return self.result
        if is_job:
            required_parameters = self.JOB_REQUIRED_PARAMETERS
        elif external_name:
            required_parameters = self.TASK_REQUIRED_PARAMETERS_EXTERNAL_NAME
        else:
            required_parameters = self.TASK_REQUIRED_PARAMETERS
        is_bstring = gf.is_bytes(config_string)
        if is_bstring:
            self.log(u"Checking that config_string is well formed")
            self.check_raw_string(config_string, is_bstring=True)
            if not self.result.passed:
                return self.result
            config_string = gf.safe_unicode(config_string)
        self.log(u"Checking required parameters")
        parameters = gf.config_string_to_dict(config_string, self.result)
        self._check_required_parameters(required_parameters, parameters)
        self.log([u"Checking config_string: returning %s", self.result.passed])
        return self.result

    def check_config_txt(self, contents, is_config_string=False):
        """
        Check whether the given TXT config file contents
        (if ``is_config_string`` is ``False``) or
        TXT config string (if ``is_config_string`` is ``True``)
        is well-formed and it has all the required parameters.

        :param string contents: the TXT config file contents or TXT config string
        :param bool is_config_string: if ``True``, contents is a config string
        :rtype: :class:`~aeneas.validator.ValidatorResult`
        """
        self.log(u"Checking contents TXT config file")
        self.result = ValidatorResult()
        if self._are_safety_checks_disabled(u"check_config_txt"):
            return self.result
        is_bstring = gf.is_bytes(contents)
        if is_bstring:
            self.log(u"Checking that contents is well formed")
            self.check_raw_string(contents, is_bstring=True)
            if not self.result.passed:
                return self.result
            contents = gf.safe_unicode(contents)
        if not is_config_string:
            self.log(u"Converting file contents to config string")
            contents = gf.config_txt_to_string(contents)
        self.log(u"Checking required parameters")
        required_parameters = self.TXT_REQUIRED_PARAMETERS
        parameters = gf.config_string_to_dict(contents, self.result)
        self._check_required_parameters(required_parameters, parameters)
        self.log([u"Checking contents: returning %s", self.result.passed])
        return self.result

    def check_config_xml(self, contents):
        """
        Check whether the given XML config file contents
        is well-formed and it has all the required parameters.

        :param string contents: the XML config file contents or XML config string
        :param bool is_config_string: if ``True``, contents is a config string
        :rtype: :class:`~aeneas.validator.ValidatorResult`
        """
        self.log(u"Checking contents XML config file")
        self.result = ValidatorResult()
        if self._are_safety_checks_disabled(u"check_config_xml"):
            return self.result
        contents = gf.safe_bytes(contents)
        self.log(u"Checking that contents is well formed")
        self.check_raw_string(contents, is_bstring=True)
        if not self.result.passed:
            return self.result
        self.log(u"Checking required parameters for job")
        job_parameters = gf.config_xml_to_dict(contents, self.result, parse_job=True)
        self._check_required_parameters(self.XML_JOB_REQUIRED_PARAMETERS, job_parameters)
        if not self.result.passed:
            return self.result
        self.log(u"Checking required parameters for task")
        tasks_parameters = gf.config_xml_to_dict(contents, self.result, parse_job=False)
        for parameters in tasks_parameters:
            self.log([u"Checking required parameters for task: '%s'", parameters])
            self._check_required_parameters(self.XML_TASK_REQUIRED_PARAMETERS, parameters)
            if not self.result.passed:
                return self.result
        return self.result

    def check_container(self, container_path, container_format=None, config_string=None):
        """
        Check whether the given container is well-formed.

        :param string container_path: the path of the container to be checked
        :param container_format: the format of the container
        :type  container_format: :class:`~aeneas.container.ContainerFormat`
        :param string config_string: the configuration string generated by the wizard
        :rtype: :class:`~aeneas.validator.ValidatorResult`
        """
        self.log([u"Checking container '%s'", container_path])
        self.result = ValidatorResult()

        if self._are_safety_checks_disabled(u"check_container"):
            return self.result

        if not (gf.file_exists(container_path) or gf.directory_exists(container_path)):
            self._failed(u"Container '%s' not found." % container_path)
            return self.result

        container = Container(container_path, container_format)
        try:
            self.log(u"Checking container has config file")
            if config_string is not None:
                self.log(u"Container with config string from wizard")
                self.check_config_txt(config_string, is_config_string=True)
            elif container.has_config_xml:
                self.log(u"Container has XML config file")
                contents = container.read_entry(container.entry_config_xml)
                if contents is None:
                    self._failed(u"Unable to read the contents of XML config file.")
                    return self.result
                self.check_config_xml(contents)
            elif container.has_config_txt:
                self.log(u"Container has TXT config file")
                contents = container.read_entry(container.entry_config_txt)
                if contents is None:
                    self._failed(u"Unable to read the contents of TXT config file.")
                    return self.result
                self.check_config_txt(contents, is_config_string=False)
            else:
                self._failed(u"Container does not have a TXT or XML configuration file.")

            self.log(u"Checking we have a valid job in the container")
            if not self.result.passed:
                return self.result
            self.log(u"Analyze the contents of the container")
            analyzer = AnalyzeContainer(container)
            if config_string is not None:
                job = analyzer.analyze(config_string=config_string)
            else:
                job = analyzer.analyze()
            self._check_analyzed_job(job, container)

        except OSError:
            self._failed(u"Unable to read the contents of the container.")
        return self.result

    def _are_safety_checks_disabled(self, caller=u"unknown_function"):
        """
        Return ``True`` if safety checks are disabled.

        :param string caller: the name of the caller function
        :rtype: bool
        """
        if self.rconf.safety_checks:
            return False
        self.log_warn([u"Safety checks disabled => %s passed", caller])
        return True

    def _failed(self, msg):
        """
        Log a validation failure.

        :param string msg: the error message
        """
        self.log(msg)
        self.result.passed = False
        self.result.add_error(msg)
        self.log(u"Failed")

    def _check_utf8_encoding(self, bstring):
        """
        Check whether the given sequence of bytes
        is properly encoded in UTF-8.

        :param bytes bstring: the byte string to be checked
        """
        if not gf.is_bytes(bstring):
            self._failed(u"The given string is not a sequence of bytes")
            return
        if not gf.is_utf8_encoded(bstring):
            self._failed(u"The given string is not encoded in UTF-8.")

    def _check_not_empty(self, string):
        """
        Check whether the given string has zero length.

        :param string string: the byte string or Unicode string to be checked
        """
        if len(string) == 0:
            self._failed(u"The given string has zero length")

    def _check_reserved_characters(self, ustring):
        """
        Check whether the given Unicode string contains reserved characters.

        :param string ustring: the string to be checked
        """
        forbidden = [c for c in gc.CONFIG_RESERVED_CHARACTERS if c in ustring]
        if len(forbidden) > 0:
            self._failed(u"The given string contains the reserved characters '%s'." % u" ".join(forbidden))

    def _check_allowed_values(self, parameters):
        """
        Check whether the given parameter value is allowed.
        Log messages into ``self.result``.

        :param dict parameters: the given parameters
        """
        for key, allowed_values in self.ALLOWED_VALUES:
            self.log([u"Checking allowed values for parameter '%s'", key])
            if key in parameters:
                value = parameters[key]
                if value not in allowed_values:
                    self._failed(u"Parameter '%s' has value '%s' which is not allowed." % (key, value))
                    return
        self.log(u"Passed")

    def _check_implied_parameters(self, parameters):
        """
        Check whether at least one of the keys in implied_keys
        is in ``parameters``,
        when a given ``key=value`` is present in ``parameters``,
        for some value in values.
        Log messages into ``self.result``.

        :param dict parameters: the given parameters
        """
        for key, values, implied_keys in self.IMPLIED_PARAMETERS:
            self.log([u"Checking implied parameters by '%s'='%s'", key, values])
            if (key in parameters) and (parameters[key] in values):
                found = False
                for implied_key in implied_keys:
                    if implied_key in parameters:
                        found = True
                if not found:
                    if len(implied_keys) == 1:
                        msg = u"Parameter '%s' is required when '%s'='%s'." % (implied_keys[0], key, parameters[key])
                    else:
                        msg = u"At least one of [%s] is required when '%s'='%s'." % (",".join(implied_keys), key, parameters[key])
                    self._failed(msg)
                    return
        self.log(u"Passed")

    def _check_required_parameters(
            self,
            required_parameters,
            parameters
    ):
        """
        Check whether the given parameter dictionary contains
        all the required paramenters.
        Log messages into ``self.result``.

        :param list required_parameters: required parameters
        :param dict parameters: parameters specified by the user
        """
        self.log([u"Checking required parameters '%s'", required_parameters])
        self.log(u"Checking input parameters are not empty")
        if (parameters is None) or (len(parameters) == 0):
            self._failed(u"No parameters supplied.")
            return
        self.log(u"Checking no required parameter is missing")
        for req_param in required_parameters:
            if req_param not in parameters:
                self._failed(u"Required parameter '%s' not set." % req_param)
                return
        self.log(u"Checking all parameter values are allowed")
        self._check_allowed_values(parameters)
        self.log(u"Checking all implied parameters are present")
        self._check_implied_parameters(parameters)
        return self.result

    def _check_analyzed_job(self, job, container):
        """
        Check that the job object generated from the given container
        is well formed, that it has at least one task,
        and that the text file of each task has the correct encoding.
        Log messages into ``self.result``.

        :param job: the Job object generated from container
        :type  job: :class:`~aeneas.job.Job`
        :param container: the Container object
        :type  container: :class:`~aeneas.container.Container`
        """
        self.log(u"Checking the Job object generated from container")

        self.log(u"Checking that the Job is not None")
        if job is None:
            self._failed(u"Unable to create a Job from the container.")
            return

        self.log(u"Checking that the Job has at least one Task")
        if len(job) == 0:
            self._failed(u"Unable to create at least one Task from the container.")
            return

        if self.rconf[RuntimeConfiguration.JOB_MAX_TASKS] > 0:
            self.log(u"Checking that the Job does not have too many Tasks")
            if len(job) > self.rconf[RuntimeConfiguration.JOB_MAX_TASKS]:
                self._failed(u"The Job has %d Tasks, more than the maximum allowed (%d)." % (
                    len(job),
                    self.rconf[RuntimeConfiguration.JOB_MAX_TASKS]
                ))
                return

        self.log(u"Checking that each Task text file is well formed")
        for task in job.tasks:
            self.log([u"Checking Task text file '%s'", task.text_file_path])
            text_file_bstring = container.read_entry(task.text_file_path)
            if (text_file_bstring is None) or (len(text_file_bstring) == 0):
                self._failed(u"Text file '%s' is empty" % task.text_file_path)
                return
            self._check_utf8_encoding(text_file_bstring)
            if not self.result.passed:
                self._failed(u"Text file '%s' is not encoded in UTF-8" % task.text_file_path)
                return
            self._check_not_empty(text_file_bstring)
            if not self.result.passed:
                self._failed(u"Text file '%s' is empty" % task.text_file_path)
                return
            self.log([u"Checking Task text file '%s': passed", task.text_file_path])
        self.log(u"Checking each Task text file is well formed: passed")


class ValidatorResult(object):
    """
    A structure to contain the result of a validation.
    """

    TAG = u"ValidatorResult"

    def __init__(self):
        self.passed = True
        self.warnings = []
        self.errors = []

    def __unicode__(self):
        msg = [
            u"Passed: %s" % self.passed,
            self.pretty_print(warnings=True)
        ]
        return u"\n".join(msg)

    def __str__(self):
        return gf.safe_str(self.__unicode__())

    def pretty_print(self, warnings=False):
        """
        Pretty print warnings and errors.

        :param bool warnings: if ``True``, also print warnings.
        :rtype: string
        """
        msg = []
        if (warnings) and (len(self.warnings) > 0):
            msg.append(u"Warnings:")
            for warning in self.warnings:
                msg.append(u"  %s" % warning)
        if len(self.errors) > 0:
            msg.append(u"Errors:")
            for error in self.errors:
                msg.append(u"  %s" % error)
        return u"\n".join(msg)

    @property
    def passed(self):
        """
        The result of a validation.

        Return ``True`` if passed, possibly with emitted warnings.

        Return ``False`` if not passed, that is, at least one error emitted.

        :rtype: bool
        """
        return self.__passed

    @passed.setter
    def passed(self, passed):
        self.__passed = passed

    @property
    def warnings(self):
        """
        The list of emitted warnings.

        :rtype: list of strings
        """
        return self.__warnings

    @warnings.setter
    def warnings(self, warnings):
        self.__warnings = warnings

    @property
    def errors(self):
        """
        The list of emitted errors.

        :rtype: list of strings
        """
        return self.__errors

    @errors.setter
    def errors(self, errors):
        self.__errors = errors

    def add_warning(self, message):
        """
        Add a message to the warnings.

        :param string message: the message to be added
        """
        self.warnings.append(message)

    def add_error(self, message):
        """
        Add a message to the errors.

        :param string message: the message to be added
        """
        self.errors.append(message)
