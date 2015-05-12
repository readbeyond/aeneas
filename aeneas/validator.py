#!/usr/bin/env python
# coding=utf-8

"""
A validator to assess whether user input is well-formed.
"""

import codecs
import os

import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf
from aeneas.analyzecontainer import AnalyzeContainer
from aeneas.container import Container, ContainerFormat
from aeneas.hierarchytype import HierarchyType
from aeneas.language import Language
from aeneas.logger import Logger
from aeneas.syncmap import SyncMapFormat
from aeneas.textfile import TextFileFormat

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl (www.readbeyond.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.0.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class Validator(object):
    """
    A validator to assess whether user input is well-formed.
    """

    TAG = "Validator"

    def __init__(self, logger=None):
        self.logger = logger
        if self.logger == None:
            self.logger = Logger()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def _check_string_encoding(self, string):
        """
        Return ``True`` if the given string is UTF-8 encoded,
        ``False`` otherwise.

        :param string: the string to be checked
        :type  string: string
        :rtype: bool
        """
        result = False
        try:
            self._log("Checking encoding of string")
            string.decode("utf-8")
            result = True
            self._log("Passed")
        except:
            result = False
            self._log("Failed")
        return result

    def _check_reserved_characters(self, string):
        """
        Return ``True`` if the given string
        does not contain reserved characters,
        ``False`` otherwise.

        :param string: the string to be checked
        :type  string: string
        :rtype: bool
        """
        self._log("Checking for reserved characters")
        for char in gc.CONFIG_RESERVED_CHARACTERS:
            if char in string:
                self._log("Failed because of character '%s'" % char)
                return False
        self._log("Passed")
        return True

    def _check_allowed_value(self, parameters, key, allowed_values, result):
        """
        Check whether the given parameter value is allowed.
        Log messages into result.

        :param parameters: the given parameters
        :type  parameters: dict
        :param key: the key name
        :type  key: string
        :param allowed_values: the list of allowed values for key
        :type  allowed_values: list of strings
        :param result: the object where to store validation messages
        :type  result: :class:`aeneas.validator.ValidatorResult`
        """
        self._log("Checking allowed values for parameter '%s'" % key)
        if key in parameters:
            value = parameters[key]
            if (value == None) or (not value in allowed_values):
                msg = "Parameter '%s' has value '%s' which is not allowed." % (key, value)
                result.passed = False
                result.add_error(msg)
                self._log(msg)
                return
        self._log("Passed")

    def _check_implied_parameter(
            self,
            parameters,
            key,
            value,
            implied_keys,
            result
        ):
        """
        Check whether at least one of the keys in implied_keys is in parameters,
        when a given ``key=value`` is present in parameters.
        Log messages into result.

        :param parameters: the given parameters
        :type  parameters: dict
        :param key: the key name
        :type  key: string
        :param value: the value for parameter key
        :type  value: string
        :param implied_keys: the list of keys implied by ``key=value``
        :type  implied_keys: list of strings
        :param result: the object where to store validation messages
        :type  result: :class:`aeneas.validator.ValidatorResult`
        """
        self._log("Checking implied parameters by '%s'='%s'" % (key, value))
        if (key in parameters) and (parameters[key] == value):
            found = False
            for implied_key in implied_keys:
                if implied_key in parameters:
                    found = True
            if not found:
                if len(implied_keys) == 1:
                    msg = "Parameter '%s' is required when '%s'='%s'." % (implied_keys[0], key, value)
                else:
                    msg = "At least one of [%s] is required when '%s'='%s'." % (",".join(implied_keys), key, value)
                result.passed = False
                result.add_error(msg)
                self._log(msg)
                return
        self._log("Passed")

    def _check_required_parameters(
            self,
            required_parameters,
            parameters,
            result
        ):
        """
        Check whether the given parameters dictionary contains
        all the required paramenters.
        Log messages into result.

        :param required_parameters: required parameters
        :type  required_parameters: list of strings
        :param parameters: parameters specified by the user
        :type  parameters: dict
        :param result: the object where to store validation messages
        :type  result: :class:`aeneas.validator.ValidatorResult`
        """
        self._log("Checking required parameters '%s'" % str(required_parameters))

        # check that the input parameters are not empty
        self._log("Checking input parameters are not empty")
        if (parameters == None) or (len(parameters) == 0):
            msg = "No parameters supplied."
            result.passed = False
            result.add_error(msg)
            self._log(msg)
            return

        # check no required parameter is missing
        self._log("Checking no required parameter is missing")
        for req_param in required_parameters:
            if not req_param in parameters:
                msg = "Required parameter '%s' not set." % req_param
                result.passed = False
                result.add_error(msg)
                self._log(msg)
                return

        # check all parameter values are allowed
        self._log("Checking all parameter values are allowed")
        self._check_allowed_value(
            parameters,
            gc.PPN_JOB_LANGUAGE,
            Language.ALLOWED_VALUES,
            result
        )
        self._check_allowed_value(
            parameters,
            gc.PPN_TASK_LANGUAGE,
            Language.ALLOWED_VALUES,
            result
        )
        self._check_allowed_value(
            parameters,
            gc.PPN_JOB_OS_CONTAINER_FORMAT,
            ContainerFormat.ALLOWED_VALUES,
            result
        )
        self._check_allowed_value(
            parameters,
            gc.PPN_JOB_IS_HIERARCHY_TYPE,
            HierarchyType.ALLOWED_VALUES,
            result
        )
        self._check_allowed_value(
            parameters,
            gc.PPN_JOB_OS_HIERARCHY_TYPE,
            HierarchyType.ALLOWED_VALUES,
            result
        )
        self._check_allowed_value(
            parameters,
            gc.PPN_JOB_IS_TEXT_FILE_FORMAT,
            TextFileFormat.ALLOWED_VALUES,
            result
        )
        self._check_allowed_value(
            parameters,
            gc.PPN_TASK_OS_FILE_FORMAT,
            SyncMapFormat.ALLOWED_VALUES,
            result
        )

        # check all parameters implied by other_parameter=value are present
        self._log("Checking all implied parameters are present")
        # is_hierarchy_type=paged => is_task_dir_name_regex
        self._check_implied_parameter(
            parameters,
            gc.PPN_JOB_IS_HIERARCHY_TYPE,
            HierarchyType.PAGED,
            [gc.PPN_JOB_IS_TASK_DIRECTORY_NAME_REGEX],
            result
        )
        # is_text_type=unparsed => is_text_unparsed_id_sort
        self._check_implied_parameter(
            parameters,
            gc.PPN_JOB_IS_TEXT_FILE_FORMAT,
            TextFileFormat.UNPARSED,
            [gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT],
            result
        )
        # is_text_type=unparsed => is_text_unparsed_class_regex or
        #                          is_text_unparsed_id_regex
        self._check_implied_parameter(
            parameters,
            gc.PPN_JOB_IS_TEXT_FILE_FORMAT,
            TextFileFormat.UNPARSED,
            [
                gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX,
                gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX
            ],
            result
        )
        # os_task_file_format=smil => os_task_file_smil_audio_ref
        self._check_implied_parameter(
            parameters,
            gc.PPN_TASK_OS_FILE_FORMAT,
            SyncMapFormat.SMIL,
            [gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF],
            result
        )
        # os_task_file_format=smil => os_task_file_smil_page_ref
        self._check_implied_parameter(
            parameters,
            gc.PPN_TASK_OS_FILE_FORMAT,
            SyncMapFormat.SMIL,
            [gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF],
            result
        )

        # return result
        self._log("Checking required parameters: returning %s" % result.passed)
        return result

    def check_file_encoding(self, input_file_path):
        """
        Check whether the given file is UTF-8 encoded.

        :param input_file_path: the path of the file to be checked
        :type  input_file_path: string (path)
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        result = ValidatorResult()
        try:
            self._log("Checking encoding of file '%s'" % input_file_path)
            file_object = codecs.open(input_file_path, 'r', encoding="utf-8")
            file_object.readlines()
        except:
            msg = "The given file is not UTF-8 encoded."
            result.passed = False
            result.add_error(msg)
            self._log(msg)
            return result
        finally:
            file_object.close()
        self._log("Passed")
        return result

    def check_string_well_encoded(self, string):
        """
        Check whether the given string is properly encoded
        and does not contain reserved characters.
        Log messages into result.

        :param string: the string to be checked
        :type  string: string
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        result = ValidatorResult()
        self._log("Checking that the given string is well encoded")
        if not self._check_string_encoding(string):
            msg = "The given string is not UTF-8 encoded."
            result.passed = False
            result.add_error(msg)
            self._log(msg)
            return result
        if not self._check_reserved_characters(string):
            msg = "The given string contains reserved characters."
            result.passed = False
            result.add_error(msg)
            self._log(msg)
            return result
        self._log("Passed")
        return result

    def check_job_configuration(self, config_string):
        """
        Check whether the given job configuration string is well-formed
        and it has all the required parameters.

        :param config_string: the string to be checked
        :type  config_string: string
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        self._log("Checking job configuration '%s'" % config_string)

        # remove BOM, if any
        #self._log("Removing BOM")
        #config_string = gf.remove_bom(config_string)

        # check if it is well encoded
        self._log("Checking that string is well encoded")
        result = self.check_string_well_encoded(config_string)
        if not result.passed:
            self._log("Failed")
            return result

        # check required parameters
        self._log("Checking required parameters")
        required_parameters = [
            gc.PPN_JOB_LANGUAGE,
            gc.PPN_JOB_OS_FILE_NAME,
            gc.PPN_JOB_OS_CONTAINER_FORMAT
        ]
        parameters = gf.config_string_to_dict(config_string, result)
        self._check_required_parameters(required_parameters, parameters, result)

        # return result
        self._log("Checking job configuration: returning %s" % result.passed)
        return result

    def check_task_configuration(self, config_string):
        """
        Check whether the given task configuration string is well-formed
        and it has all the required parameters.

        :param config_string: the string to be checked
        :type  config_string: string
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        self._log("Checking task configuration '%s'" % config_string)

        # remove BOM, if any
        #self._log("Removing BOM")
        #config_string = gf.remove_bom(config_string)

        # check if it is well encoded
        self._log("Checking that string is well encoded")
        result = self.check_string_well_encoded(config_string)
        if not result.passed:
            self._log("Failed")
            return result

        # check required parameters
        self._log("Checking required parameters")
        required_parameters = [
            gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
            gc.PPN_TASK_LANGUAGE,
            gc.PPN_TASK_OS_FILE_NAME,
            gc.PPN_TASK_OS_FILE_FORMAT
        ]
        parameters = gf.config_string_to_dict(config_string, result)
        self._check_required_parameters(required_parameters, parameters, result)

        # return result
        self._log("Checking task configuration: returning %s" % result.passed)
        return result

    def check_container(self, container_path, container_format=None):
        """
        Check whether the given container is well-formed.

        :param container_path: the path of the container to be checked
        :type  container_path: string (path)
        :param container_format: the format of the container
        :type  container_format: string (from ContainerFormat enumeration)
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        self._log("Checking container file '%s'" % container_path)

        result = ValidatorResult()

        # check the container file exists
        self._log("Checking container file exists")
        if not os.path.exists(container_path):
            msg = "Container file '%s' not found." % container_path
            result.passed = False
            result.add_error(msg)
            self._log(msg)
            return result

        # check if we have config.xml or config.txt
        self._log("Checking container file has config file")
        container = Container(container_path, container_format)
        if container.has_config_xml:
            self._log("Container has XML config file")
            result = self._check_container_with_xml_config(
                container=container,
                config_contents=None
            )
        elif container.has_config_txt:
            self._log("Container has TXT config file")
            result = self._check_container_with_txt_config_string(
                container=container,
                config_string=None
            )
        else:
            msg = "Container does not have a TXT or XML configuration file."
            result.passed = False
            result.add_error(msg)
            self._log(msg)

        # return result
        self._log("Checking container: returning %s" % result.passed)
        return result

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
        container = Container(container_path, container_format)
        return self._check_container_with_txt_config_string(
            container=container,
            config_string=config_string
        )

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
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        self._log("Checking container with TXT config file")

        result = ValidatorResult()
        # if no config string was passed, try to read it from container
        if config_string == None:
            self._log("Trying to read config file from container")
            config_in_container = True
            config_contents = container.read_entry(container.entry_config_txt)
            if config_contents == None:
                msg = "Unable to read the contents of TXT config file."
                result.passed = False
                result.add_error(msg)
                self._log(msg)
                return result
            self._log("Config file found in container")
        else:
            self._log("Config string passed as parameter")
            config_in_container = False

        # check the txt config contents or string
        result = self.check_contents_txt_config_file(
            config_contents,
            config_in_container
        )
        if not result.passed:
            self._log("Failed")
            return result

        # analyze the container
        self._log("Analyze the contents of the container")
        analyzer = AnalyzeContainer(container)
        if config_in_container:
            job = analyzer.analyze()
        else:
            job = analyzer.analyze_from_wizard(config_string)
        self._check_analyzed_job(job, container, result)

        # return result
        self._log("Checking container with TXT config file: returning %s" % result.passed)
        return result

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

        result = ValidatorResult()
        if convert_to_string:
            #self._log("Removing BOM")
            #config_contents = gf.remove_bom(config_contents)
            self._log("Converting file contents to config string")
            config_string = gf.config_txt_to_string(config_contents)
        #else:
            #self._log("Removing BOM")
            #config_string = gf.remove_bom(config_string)

        # check if it is well encoded
        self._log("Checking that string is well encoded")
        if not self.check_string_well_encoded(config_string):
            msg = "The TXT config is not well encoded"
            result.passed = False
            result.add_error(msg)
            self._log(msg)
            return result

        # check required parameters
        self._log("Checking required parameters")
        required_parameters = [
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
        parameters = gf.config_string_to_dict(config_string, result)
        self._check_required_parameters(required_parameters, parameters, result)

        # return result
        self._log("Checking contents TXT config file: returning %s" % result.passed)
        return result

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
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        self._log("Checking container with XML config file")

        result = ValidatorResult()
        # if no config contents was passed, try to read them from container
        if config_contents == None:
            self._log("Trying to read config file from container")
            config_contents = container.read_entry(container.entry_config_xml)
            if config_contents == None:
                msg = "Unable to read the contents of XML config file."
                result.passed = False
                result.add_error(msg)
                self._log(msg)
                return result

        # check the txt config contents or string
        result = self.check_contents_xml_config_file(config_contents)
        if not result.passed:
            self._log("Failed")
            return result

        # analyze the container
        self._log("Analyze the contents of the container")
        analyzer = AnalyzeContainer(container)
        job = analyzer.analyze()
        self._check_analyzed_job(job, container, result)

        # return result
        self._log("Checking container: returning %s" % result.passed)
        return result

    def check_contents_xml_config_file(self, config_contents):
        """
        Check whether the given XML config contents
        is well formed and contains all the requested parameters.

        :param config_contents:
        :type  config_contents: string
        :rtype: :class:`aeneas.validator.ValidatorResult`
        """
        self._log("Checking contents XML config file")

        result = ValidatorResult()

        # remove BOM, if any
        #self._log("Removing BOM")
        #config_contents = gf.remove_bom(config_contents)

        # check if it is well encoded
        result = self.check_string_well_encoded(config_contents)
        if not result.passed:
            self._log("Failed")
            return result

        # check required parameters for the job section
        self._log("Checking required parameters for job")
        required_parameters = [
            gc.PPN_JOB_OS_FILE_NAME,
            gc.PPN_JOB_OS_CONTAINER_FORMAT,
            gc.PPN_JOB_OS_HIERARCHY_TYPE,
            gc.PPN_JOB_OS_HIERARCHY_PREFIX
        ]
        job_parameters = gf.config_xml_to_dict(
            config_contents,
            result,
            parse_job=True
        )
        self._check_required_parameters(
            required_parameters,
            job_parameters,
            result
        )
        if not result.passed:
            self._log("Failed")
            return result

        # check required parameters for each task section
        self._log("Checking required parameters for task")
        required_parameters = [
            gc.PPN_TASK_LANGUAGE,
            gc.PPN_TASK_IS_TEXT_FILE_FORMAT,
            gc.PPN_TASK_IS_TEXT_FILE_XML,
            gc.PPN_TASK_IS_AUDIO_FILE_XML,
            gc.PPN_TASK_OS_FILE_NAME,
            gc.PPN_TASK_OS_FILE_FORMAT,
        ]
        tasks_parameters = gf.config_xml_to_dict(
            config_contents,
            result,
            parse_job=False
        )
        for parameters in tasks_parameters:
            self._log("Checking required parameters for task: '%s'" % str(parameters))
            self._check_required_parameters(
                required_parameters,
                parameters,
                result
            )
            if not result.passed:
                self._log("Failed")
                return result

        # return result
        self._log("Checking contents XML config file: returning %s" % result.passed)
        return result

    def _check_analyzed_job(self, job, container, result):
        """
        Check that the job object generated from the given container
        is well formed, that it has at least one task,
        and that the text file of each task has the correct encoding.
        Log messages into result.

        :param job: the Job object generated from container
        :type  job: Job
        :param container: the Container object
        :type  container: Container
        :param result: the object where to store validation messages
        :type  result: :class:`aeneas.validator.ValidatorResult`
        """
        self._log("Checking the Job object generated from container")

        # we must have a valid Job object
        self._log("Checking the Job is not None")
        if job == None:
            msg = "Unable to create a Job from the container."
            result.passed = False
            result.add_error(msg)
            self._log(msg)
            return

        # we must have at least one Task
        self._log("Checking the Job has at least one Task")
        if len(job) == 0:
            msg = "Unable to create at least one Task from the container."
            result.passed = False
            result.add_error(msg)
            self._log(msg)
            return

        # each Task text file must be well encoded
        self._log("Checking each Task text file is well encoded")
        for task in job.tasks:
            self._log("Checking Task text file '%s'" % task.text_file_path)
            text_file_contents = container.read_entry(task.text_file_path)
            #self._log("Removing BOM")
            #text_file_contents = gf.remove_bom(text_file_contents)
            if ((text_file_contents == None) or
                    (len(text_file_contents) == 0) or
                    (not self._check_string_encoding(text_file_contents))):
                msg = "Text file '%s' seems empty or it has a wrong encoding." % task.text_file_path
                result.passed = False
                result.add_error(msg)
                self._log(msg)
                return
            self._log("Checking Task text file '%s': passed" % task.text_file_path)
        self._log("Checking each Task text file is well encoded: passed")



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
        accumulator += "Warnings:\n"
        for warning in self.warnings:
            accumulator += "%s\n" % warning
        accumulator += "Errors:\n"
        for error in self.errors:
            accumulator += "%s\n" % error
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



