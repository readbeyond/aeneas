#!/usr/bin/env python
# coding=utf-8

"""
Execute a job, that is, execute all of its tasks
and generate the output container
holding the generated sync maps.
"""

import os
import shutil
import tempfile

import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf
from aeneas.analyzecontainer import AnalyzeContainer
from aeneas.container import Container, ContainerFormat
from aeneas.executetask import ExecuteTask
from aeneas.logger import Logger
from aeneas.validator import Validator

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl (www.readbeyond.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.0.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class ExecuteJob(object):
    """
    Execute a job, that is, execute all of its tasks
    and generate the output container
    holding the generated sync maps.

    If you do not provide a job object in the constructor,
    you must manually set it later, or load it from a container
    with ``load_job_from_container``.

    In the first case, you are responsible for setting
    the absolute audio/text/sync map paths of each task of the job,
    to their actual absolute location on the computing machine.
    Moreover, you are responsible for cleaning up
    any temporary files you might have generated around.

    In the second case, you are responsible for
    calling ``clean`` at the end of the job execution,
    to delete the working directory
    created by ``load_job_from_container``
    when creating the job object.

    :param job: the job to be executed
    :type  job: :class:`aeneas.job.Job`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = "ExecuteJob"

    def __init__(self, job=None, logger=None):
        self.job = job
        self.working_directory = None
        self.tmp_directory = None
        self.logger = logger
        if self.logger == None:
            self.logger = Logger()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def load_job(self, job):
        """
        Load the given job.

        NOTE: no sanity check is perfomed by this call,
        and it will always return ``True``.

        :param job: the job to load
        :type  job: :class:`aeneas.job.Job`
        :rtype: bool
        """
        self.job = job
        return True

    def load_job_from_container(self, container_path, config_string=None):
        """
        Validate the given container, and, if it is well formed,
        load the job from it.

        If ``config_string`` is ``None``,
        the container must contain a configuration file;
        otherwise use the provided config string
        (i.e., the wizard case).

        Return ``True`` if the job has been loaded successfully,
        ``False`` otherwise.

        :param container_path: the path to the input container
        :type  container_path: string (path)
        :param config_string: the configuration string (from wizard)
        :type  config_string: string
        :rtype: bool
        """
        self._log("Loading job from container...")

        # validate container
        self._log("Validating container...")
        validator = Validator(logger=self.logger)
        if config_string == None:
            validator_result = validator.check_container(container_path)
        else:
            validator_result = validator.check_container_from_wizard(
                container_path,
                config_string
            )
        if not validator_result.passed:
            self._log("Validating container: failed")
            self._log("Loading job from container: failed")
            return False
        self._log("Validating container: succeeded")

        try:
            # create working directory where the input container
            # will be decompressed
            self.working_directory = tempfile.mkdtemp(dir=gf.custom_tmp_dir())
            self._log("Created working directory '%s'" % self.working_directory)

            # decompress
            self._log("Decompressing input container...")
            input_container = Container(container_path, logger=self.logger)
            input_container.decompress(self.working_directory)
            self._log("Decompressing input container... done")

            # create job from the working directory
            self._log("Creating job from working directory...")
            working_container = Container(
                self.working_directory,
                logger=self.logger
            )
            analyzer = AnalyzeContainer(working_container, logger=self.logger)
            if config_string == None:
                self.job = analyzer.analyze()
            else:
                self.job = analyzer.analyze_from_wizard(config_string)
            self._log("Creating job from working directory... done")

            # set absolute path for text file and audio file
            # for each task in the job
            self._log("Setting absolute paths for tasks...")
            for task in self.job.tasks:
                task.text_file_path_absolute = gf.norm_join(
                    self.working_directory,
                    task.text_file_path
                )
                task.audio_file_path_absolute = gf.norm_join(
                    self.working_directory,
                    task.audio_file_path
                )
            self._log("Setting absolute paths for tasks... done")

            # return
            self._log("Loading job from container: succeeded")
            return True
        except:
            # failure: clean and return
            self.clean()
            self._log("Loading job from container: failed")
            return False

    def write_output_container(self, output_directory_path):
        """
        Write the output container for this job.

        Return a pair ``(bool, string)``, where the bool
        indicates whether the execution succeeded,
        and the string is the path to output container.

        :param output_directory_path: the path to a directory where
                                      the output container must be created
        :type  output_directory_path: string (path)
        :rtype: (bool, string)
        """
        self._log("Writing output container for this job")

        # check if the job has tasks
        if self.job == None:
            self._log("job is None")
            return (False, None)
        if len(self.job) == 0:
            self._log("The job has no tasks")
            return (False, None)

        try:
            # create temporary directory where the sync map files
            # will be created
            # this temporary directory will be compressed into
            # the output container
            self.tmp_directory = tempfile.mkdtemp(dir=gf.custom_tmp_dir())
            self._log("Created temporary directory '%s'" % self.tmp_directory)

            for task in self.job.tasks:
                custom_id = task.configuration.custom_id

                # check if the task has sync map and sync map file path
                if task.sync_map_file_path == None:
                    self._log("Task '%s' has sync_map_file_path not set" % custom_id)
                    return (False, None)
                if task.sync_map == None:
                    self._log("Task '%s' has sync_map not set" % custom_id)
                    return (False, None)

                # output sync map
                self._log("Outputting sync map for task '%s'..." % custom_id)
                task.output_sync_map_file(self.tmp_directory)
                self._log("Outputting sync map for task '%s'... done" % custom_id)

            # get output container info
            output_container_format = self.job.configuration.os_container_format
            self._log("Output container format: '%s'" % output_container_format)
            output_file_name = self.job.configuration.os_file_name
            if ((output_container_format != ContainerFormat.UNPACKED) and
                    (not output_file_name.endswith(output_container_format))):
                self._log("Adding extension to output_file_name")
                output_file_name += "." + output_container_format
            self._log("Output file name: '%s'" % output_file_name)
            output_file_path = gf.norm_join(
                output_directory_path,
                output_file_name
            )
            self._log("Output file path: '%s'" % output_file_path)

            # create output container
            self._log("Compressing...")
            container = Container(
                output_file_path,
                output_container_format,
                logger=self.logger
            )
            container.compress(self.tmp_directory)
            self._log("Compressing... done")
            self._log("Created output file: '%s'" % output_file_path)

            # clean and return
            self.clean(False)
            return (True, output_file_path)
        except:
            self.clean(False)
            return (False, None)

    def execute(self):
        """
        Execute the job, that is, execute all of its tasks.

        Each produced sync map will be stored
        inside the corresponding task object.

        Return ``True`` if the execution succeeded,
        ``False`` otherwise.

        :rtype: bool
        """
        self._log("Executing job")

        # check if the job has tasks
        if self.job == None:
            self._log("job is None")
            return False
        if len(self.job) == 0:
            self._log("The job has no tasks")
            return False
        self._log("Number of tasks: '%s'" % len(self.job))

        # execute tasks
        for task in self.job.tasks:
            custom_id = task.configuration.custom_id
            self._log("Executing task '%s'..." % custom_id)
            executor = ExecuteTask(task, logger=self.logger)
            result = executor.execute()
            self._log("Executing task '%s'... done" % custom_id)
            if not result:
                self._log("Executing task: failed")
                return False
            self._log("Executing task: succeeded")

        # return
        self._log("Executing job: succeeded")
        return True

    def clean(self, remove_working_directory=True):
        """
        Remove the temporary directory.
        If ``remove_working_directory`` is True
        remove the working directory as well,
        otherwise just remove the temporary directory.

        :param remove_working_directory: if ``True``, remove
                                         the working directory as well
        :type  remove_working_directory: bool
        """
        if remove_working_directory:
            self._log("Removing working directory... ")
            self._clean(self.working_directory)
            self.working_directory = None
            self._log("Removing working directory... done")
        self._log("Removing temporary directory... ")
        self._clean(self.tmp_directory)
        self.tmp_directory = None
        self._log("Removing temporary directory... done")

    def _clean(self, path):
        """
        Remove the directory ``path``.

        :param path: the path of the directory to be removed
        :type  path: string (path)
        """
        if (path != None) and (os.path.isdir(path)):
            try:
                self._log("Removing directory '%s'..." % path)
                shutil.rmtree(path)
                self._log("Succeeded")
            except:
                self._log("Failed")



