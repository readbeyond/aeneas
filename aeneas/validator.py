#!/usr/bin/env python
# coding=utf-8

"""
A validator to assess whether user input is well-formed.
"""

import codecs

from aeneas.analyzecontainer import AnalyzeContainer
from aeneas.container import Container
from aeneas.container import ContainerFormat
from aeneas.executetask import AdjustBoundaryAlgorithm
from aeneas.hierarchytype import HierarchyType
from aeneas.idsortingalgorithm import IDSortingAlgorithm
from aeneas.language import Language
from aeneas.logger import Logger
from aeneas.syncmap import SyncMapFormat
from aeneas.syncmap import SyncMapHeadTailFormat
from aeneas.textfile import TextFileFormat
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

class Validator(object):
    """
    A validator to assess whether user input is well-formed.

    Note that all strings are ``str`` objects.
    """

    TAG = "Validator"

    ALLOWED_VALUES = [
        (
            gc.PPN_JOB_LANGUAGE,
            Language.ALLOWED_VALUES
        ),
        (
            gc.PPN_TASK_LANGUAGE,
            Language.ALLOWED_VALUES
        ),
        (
            gc.PPN_JOB_OS_CONTAINER_FORMAT,
            ContainerFormat.ALLOWED_VALUES
        ),
        (
            gc.PPN_JOB_IS_HIERARCHY_TYPE,
            HierarchyType.ALLOWED_VALUES
        ),
        (
            gc.PPN_JOB_OS_HIERARCHY_TYPE,
            HierarchyType.ALLOWED_VALUES
        ),
        (
            gc.PPN_JOB_IS_TEXT_FILE_FORMAT,
            TextFileFormat.ALLOWED_VALUES
        ),
        (
            gc.PPN_TASK_OS_FILE_FORMAT,
            SyncMapFormat.ALLOWED_VALUES
        ),
        (
            gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT,
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
            gc.PPN_JOB_IS_TEXT_FILE_FORMAT,
            [TextFileFormat.UNPARSED],
            [gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT]
        ),
        (
            # is_text_type=unparsed => is_text_unparsed_class_regex or
            #                          is_text_unparsed_id_regex
            gc.PPN_JOB_IS_TEXT_FILE_FORMAT,
            [TextFileFormat.UNPARSED],
            [
                gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX,
                gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX
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
        gc.PPN_JOB_OS_FILE_NAME,
        gc.PPN_JOB_OS_CONTAINER_FORMAT
    ]

    TASK_REQUIRED_PARAMETERS = [
        gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
        gc.PPN_TASK_LANGUAGE,
        gc.PPN_TASK_OS_FILE_NAME,
        gc.PPN_TASK_OS_FILE_FORMAT
    ]

    TASK_REQUIRED_PARAMETERS_EXTERNAL_NAME = [
        gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
        gc.PPN_TASK_LANGUAGE,
        gc.PPN_TASK_OS_FILE_FORMAT
    ]

    TXT_REQUIRED_PARAMETERS = [
        gc.PPN_JOB_IS_HIERARCHY_TYPE,
        gc.PPN_JOB_IS_HIERARCHY_PREFIX,
        gc.PPN_JOB_IS_TEXT_FILE_RELATIVE_PATH,
        gc.PPN_JOB_IS_TEXT_FILE_NAME_REGEX,
        gc.PPN_JOB_IS_TEXT_FILE_FORMAT,
        gc.PPN_JOB_IS_AUDIO_FILE_RELATIVE_PATH,
        gc.PPN_JOB_IS_AUDIO_FILE_NAME_REGEX,
        gc.PPN_JOB_OS_FILE_NAME,
        gc.PPN_JOB_OS_CONTAINER_FORMAT,
        gc.PPN_JOB_OS_HIERARCHY_TYPE,
        gc.PPN_JOB_OS_HIERARCHY_PREFIX,
        gc.PPN_TASK_OS_FILE_NAME,
        gc.PPN_TASK_OS_FILE_FORMAT,
        gc.PPN_JOB_LANGUAGE
    ]

    XML_JOB_REQUIRED_PARAMETERS = [
        gc.PPN_JOB_OS_FILE_NAME,
        gc.PPN_JOB_OS_CONTAINER_FORMAT,
        gc.PPN_JOB_OS_HIERARCHY_TYPE,
        gc.PPN_JOB_OS_HIERARCHY_PREFIX
    ]

    XML_TASK_REQUIRED_PARAMETERS = [
        gc.PPN_TASK_LANGUAGE,
        gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
        gc.PPN_TASK_IS_TEXT_FILE_XML,
        gc.PPN_TASK_IS_AUDIO_FILE_XML,
        gc.PPN_TASK_OS_FILE_NAME,
        gc.PPN_TASK_OS_FILE_FORMAT,
    ]

    def __init__(self, logger=None):
        self.result = None
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def check_file_encoding(self, input_file_path):
        """
        Check whether the given file is UTF-8 encoded.

        :param input_file_path: the path of the file to be checked
        :type  input_file_path: string (path)
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        self._log(["Checking encoding of file '%s'", input_file_path])
        self.result = ValidatorResult()

        self._log("Checking file exists")
        self._check_file_exists(input_file_path)
        if not self.result.passed:
            return self.result

        try:
            file_object = None
            file_object = codecs.open(input_file_path, "r", encoding="utf-8")
            file_object.readlines()
        except UnicodeError:
            self._failed("The given file is not UTF-8 encoded.")
        finally:
            if file_object is not None:
                file_object.close()
        return self.result

    def check_string_well_encoded(self, string):
        """
        Check whether the given string is properly UTF-8 encoded
        and it does not contain reserved characters.

        :param string: the string to be checked
        :type  string: string
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        self._log("Checking that the given string is well encoded")
        self.result = ValidatorResult()
        self._check_string_encoding(string)
        if not self.result.passed:
            return self.result
        self._check_reserved_characters(string)
        return self.result

    def check_job_configuration(self, config_string):
        """
        Check whether the given job configuration string is well-formed
        and it has all the required parameters.

        :param config_string: the string to be checked
        :type  config_string: string
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        self._log("Checking job configuration")
        self.result = ValidatorResult()
        self._check_configuration_string(config_string, self.JOB_REQUIRED_PARAMETERS)
        return self.result

    def check_task_configuration(self, config_string, external_name=False):
        """
        Check whether the given task configuration string is well-formed
        and it has all the required parameters.

        :param config_string: the string to be checked
        :type  config_string: string
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        self._log("Checking task configuration")
        self.result = ValidatorResult()
        if external_name:
            required_parameters = self.TASK_REQUIRED_PARAMETERS_EXTERNAL_NAME
        else:
            required_parameters = self.TASK_REQUIRED_PARAMETERS
        self._check_configuration_string(config_string, required_parameters)
        return self.result

    def check_container(self, container_path, container_format=None):
        """
        Check whether the given container is well-formed.

        :param container_path: the path of the container to be checked
        :type  container_path: string (path)
        :param container_format: the format of the container
        :type  container_format: string (from ContainerFormat enumeration)
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        self._log(["Checking container '%s'", container_path])
        self.result = ValidatorResult()

        self._log("Checking container exists")
        self._check_file_exists(container_path, True)
        if not self.result.passed:
            return self.result

        self._log("Checking container has config file")
        container = Container(container_path, container_format)
        try:
            if container.has_config_xml:
                self._log("Container has XML config file")
                self._check_container_with_xml_config(
                    container=container,
                    config_contents=None
                )
            elif container.has_config_txt:
                self._log("Container has TXT config file")
                self._check_container_with_txt_config_string(
                    container=container,
                    config_string=None
                )
            else:
                self._failed("Container does not have a TXT or XML configuration file.")
        except IOError:
            self._failed("Unable to read the contents of the container.")
        return self.result

    def check_container_from_wizard(
            self,
            container_path,
            config_string,
            container_format=None
        ):
        """
        Check whether the given container and configuration strings
        from the wizard are well-formed.

        :param container_path: the path of the container to be checked
        :type  container_path: string (path)
        :param config_string: the configuration string generated by the wizard
        :type  config_string: string
        :param container_format: the format of the container
        :type  container_format: string (from ContainerFormat enumeration)
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        self._log("Checking container from wizard")
        self.result = ValidatorResult()

        container = Container(container_path, container_format)
        self._check_container_with_txt_config_string(
            container=container,
            config_string=config_string
        )

        self._log(["Checking container from wizard: returning %s", self.result.passed])
        return self.result

    def check_contents_txt_config_file(
            self,
            config_contents,
            convert_to_string=True
        ):
        """
        Check whether the given TXT config contents (or config string)
        is well formed and contains all the requested parameters.

        :param config_contents:
        :type  config_contents: string
        :param convert_to_string: the ``config_contents`` must be converted
                                  to a config string
        :type convert_to_string: bool
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        self._log("Checking contents TXT config file")
        self.result = ValidatorResult()

        config_string = config_contents
        if convert_to_string:
            self._log("Converting file contents to config string")
            config_string = gf.config_txt_to_string(config_contents)

        self._log("Checking that string is well encoded")
        if not self.check_string_well_encoded(config_string):
            self._failed("The TXT config is not well encoded")
            return self.result

        self._log("Checking required parameters")
        parameters = gf.config_string_to_dict(config_string, self.result)
        self._check_required_parameters(self.TXT_REQUIRED_PARAMETERS, parameters)

        return self.result

    def check_contents_xml_config_file(self, config_contents):
        """
        Check whether the given XML config contents
        is well formed and contains all the requested parameters.

        :param config_contents:
        :type  config_contents: string
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        self._log("Checking contents XML config file")
        self.result = ValidatorResult()

        self.check_string_well_encoded(config_contents)
        if not self.result.passed:
            return self.result

        self._log("Checking required parameters for job")
        job_parameters = gf.config_xml_to_dict(
            config_contents,
            self.result,
            parse_job=True
        )
        self._check_required_parameters(
            self.XML_JOB_REQUIRED_PARAMETERS,
            job_parameters
        )
        if not self.result.passed:
            return self.result

        self._log("Checking required parameters for task")
        tasks_parameters = gf.config_xml_to_dict(
            config_contents,
            self.result,
            parse_job=False
        )
        for parameters in tasks_parameters:
            self._log(["Checking required parameters for task: '%s'", parameters])
            self._check_required_parameters(
                self.XML_TASK_REQUIRED_PARAMETERS,
                parameters
            )
            if not self.result.passed:
                return self.result

        return self.result

    def _check_file_exists(self, path, allow_directory=False):
        """
        Check whether a file exists at the given path.

        :param path: the path of the file to be checked
        :type  path: string (path)
        :param allow_directory: if ``True``, allow directory
        :type  allow_directory: bool
        """
        self._log(["Checking file/directory '%s' exists (%s)", path, allow_directory])
        exists = gf.file_exists(path)
        if allow_directory:
            exists = exists or gf.directory_exists(path)
        if not exists:
            self._failed("No file found at path '%s'." % path)

    def _check_configuration_string(self, config_string, required_parameters):
        """
        Check config_string from job or task.

        :param config_string: the string to be checked
        :type  config_string: string
        :param required_parameters: the list of strings representing
                                    the required parameter names
        :type  required_parameters: list of strings
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        self._log(["Checking config_string '%s'", config_string])

        self._log("Checking that string is well encoded")
        self.check_string_well_encoded(config_string)
        if not self.result.passed:
            return self.result

        self._log("Checking required parameters")
        parameters = gf.config_string_to_dict(config_string, self.result)
        self._check_required_parameters(required_parameters, parameters)

        self._log(["Checking config_string: returning %s", self.result.passed])
        return self.result

    def _check_container_with_txt_config_string(
            self,
            container,
            config_string=None
        ):
        """
        Check whether the given container with TXT (INI-like)
        configuration is well-formed.

        :param container: the container
        :type  container: Container
        :param config_string: the TXT (INI-like) configuration string
        :type  config_string: string
        """
        self._log("Checking container with TXT config file")

        # if no config string was passed, try to read it from container
        if config_string is None:
            self._log("Trying to read config file from container")
            config_in_container = True
            config_contents = container.read_entry(container.entry_config_txt)
            if config_contents is None:
                self._failed("Unable to read the contents of TXT config file.")
                return
            self._log("Config file found in container")
        else:
            self._log("Config string passed as parameter")
            config_contents = config_string
            config_in_container = False

        # check the txt config contents or string
        self.check_contents_txt_config_file(
            config_contents,
            config_in_container
        )
        if not self.result.passed:
            return

        # analyze the container
        self._log("Analyze the contents of the container")
        analyzer = AnalyzeContainer(container)
        if config_in_container:
            job = analyzer.analyze()
        else:
            job = analyzer.analyze_from_wizard(config_string)
        self._check_analyzed_job(job, container)

        # return result
        self._log(["Checking container with TXT config file: returning %s", self.result.passed])

    def _check_container_with_xml_config(
            self,
            container,
            config_contents=None
        ):
        """
        Check whether the given container with XML configuration is well-formed.

        :param container: the container
        :type  container: Container
        :param config_contents: the contents of the XML config file
        :type  config_contents: string
        """
        self._log("Checking container with XML config file")

        # if no config contents was passed, try to read them from container
        if config_contents is None:
            self._log("Trying to read config file from container")
            config_contents = container.read_entry(container.entry_config_xml)
            if config_contents is None:
                self._failed("Unable to read the contents of XML config file.")
                return

        # check the txt config contents or string
        self.check_contents_xml_config_file(config_contents)
        if not self.result.passed:
            return

        # analyze the container
        self._log("Analyze the contents of the container")
        analyzer = AnalyzeContainer(container)
        job = analyzer.analyze()
        self._check_analyzed_job(job, container)

        # return result
        self._log(["Checking container: returning %s", self.result.passed])

    def _check_analyzed_job(self, job, container):
        """
        Check that the job object generated from the given container
        is well formed, that it has at least one task,
        and that the text file of each task has the correct encoding.
        Log messages into result.

        :param job: the Job object generated from container
        :type  job: Job
        :param container: the Container object
        :type  container: Container
        """
        self._log("Checking the Job object generated from container")

        # we must have a valid Job object
        self._log("Checking the Job is not None")
        if job is None:
            self._failed("Unable to create a Job from the container.")
            return

        # we must have at least one Task
        self._log("Checking the Job has at least one Task")
        if len(job) == 0:
            self._failed("Unable to create at least one Task from the container.")
            return

        # each Task text file must be well encoded
        self._log("Checking each Task text file is well encoded")
        for task in job.tasks:
            self._log(["Checking Task text file '%s'", task.text_file_path])
            text_file_contents = container.read_entry(task.text_file_path)
            if (text_file_contents is None) or (len(text_file_contents) == 0):
                self._failed("Text file '%s' is empty" % task.text_file_path)
                return
            self._check_string_encoding(text_file_contents)
            if not self.result.passed:
                self._failed("Text file '%s' is not properly encoded" % task.text_file_path)
                return
            self._log(["Checking Task text file '%s': passed", task.text_file_path])
        self._log("Checking each Task text file is well encoded: passed")



    def _failed(self, msg):
        """
        Log a validation failure.

        :param msg: the error message
        :type  msg: string
        """
        self._log(msg)
        self.result.passed = False
        self.result.add_error(msg)
        self._log("Failed")

    def _check_string_encoding(self, string):
        """
        Check whether the given string is UTF-8 encoded.

        :param string: the string to be checked
        :type  string: string
        """
        if not isinstance(string, str):
            self._failed("The given string is not an instance of str")
            return
        if len(string) == 0:
            self._failed("The given string has zero length")
            return
        try:
            string.decode("utf-8")
        except UnicodeDecodeError:
            self._failed("The given string is not UTF-8 encoded.")

    def _check_reserved_characters(self, string):
        """
        Check whether the given string contains reserved characters.

        :param string: the string to be checked
        :type  string: string
        """
        try:
            for char in gc.CONFIG_RESERVED_CHARACTERS:
                if char in string:
                    self._failed("The given string contains the reserved character '%s'." % char)
                    return
        except UnicodeError:
            self._log("Unexpected UnicodeError", Logger.CRITICAL)
            self._failed("The given string contains non-UTF-8 characters")
            return

    def _check_allowed_values(self, parameters):
        """
        Check whether the given parameter value is allowed.
        Log messages into result.

        :param parameters: the given parameters
        :type  parameters: dict
        """
        for cav in self.ALLOWED_VALUES:
            key, allowed_values = cav
            self._log(["Checking allowed values for parameter '%s'", key])
            if key in parameters:
                value = parameters[key]
                if not value in allowed_values:
                    self._failed("Parameter '%s' has value '%s' which is not allowed." % (key, value))
                    return
        self._log("Passed")

    def _check_implied_parameters(self, parameters):
        """
        Check whether at least one of the keys in implied_keys
        is in parameters,
        when a given ``key=value`` is present in parameters,
        for some value in values.
        Log messages into result.

        :param parameters: the given parameters
        :type  parameters: dict
        """
        for cip in self.IMPLIED_PARAMETERS:
            key, values, implied_keys = cip
            self._log(["Checking implied parameters by '%s'='%s'", key, values])
            if (key in parameters) and (parameters[key] in values):
                found = False
                for implied_key in implied_keys:
                    if implied_key in parameters:
                        found = True
                if not found:
                    if len(implied_keys) == 1:
                        msg = "Parameter '%s' is required when '%s'='%s'." % (implied_keys[0], key, parameters[key])
                    else:
                        msg = "At least one of [%s] is required when '%s'='%s'." % (",".join(implied_keys), key, parameters[key])
                    self._failed(msg)
                    return
        self._log("Passed")

    def _check_required_parameters(
            self,
            required_parameters,
            parameters
        ):
        """
        Check whether the given parameters dictionary contains
        all the required paramenters.
        Log messages into result.

        :param required_parameters: required parameters
        :type  required_parameters: list of strings
        :param parameters: parameters specified by the user
        :type  parameters: dict
        """
        self._log(["Checking required parameters '%s'", required_parameters])
        self._log("Checking required parameters")

        self._log("Checking input parameters are not empty")
        if (parameters is None) or (len(parameters) == 0):
            self._failed("No parameters supplied.")
            return

        self._log("Checking no required parameter is missing")
        for req_param in required_parameters:
            if not req_param in parameters:
                self._failed("Required parameter '%s' not set." % req_param)
                return

        self._log("Checking all parameter values are allowed")
        self._check_allowed_values(parameters)

        self._log("Checking all implied parameters are present")
        self._check_implied_parameters(parameters)

        return self.result




class ValidatorResult(object):
    """
    A structure to contain the result of a validation.
    """

    TAG = "ValidatorResult"

    def __init__(self):
        self.passed = True
        self.warnings = []
        self.errors = []

    def __str__(self):
        accumulator = ""
        accumulator += "Passed: %s\n" % self.passed
        accumulator += self.pretty_print(True)
        return accumulator

    def pretty_print(self, warnings=False):
        """
        Pretty print warnings and errors.

        :param warnings: if ``True``, also print warnings.
        :type  warnings: bool
        :rtype: str
        """
        accumulator = ""
        if (warnings) and (len(self.warnings) > 0):
            accumulator += "Warnings:\n"
            for warning in self.warnings:
                accumulator += "  %s\n" % warning
        if len(self.errors) > 0:
            accumulator += "Errors:\n"
            for error in self.errors:
                accumulator += "  %s\n" % error
        return accumulator

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

        :param message: the message to be added
        :type  message: string
        """
        self.warnings.append(message)

    def add_error(self, message):
        """
        Add a message to the errors.

        :param message: the message to be added
        :type  message: string
        """
        self.errors.append(message)



