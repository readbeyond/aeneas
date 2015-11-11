#!/usr/bin/env python
# coding=utf-8

"""
Execute a job, that is, execute all of its tasks
and generate the output container
holding the generated sync maps.
"""

import tempfile

from aeneas.analyzecontainer import AnalyzeContainer
from aeneas.container import Container
from aeneas.container import ContainerFormat
from aeneas.executetask import ExecuteTask
from aeneas.job import Job
from aeneas.logger import Logger
from aeneas.validator import Validator
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

class ExecuteJobInputError(Exception):
    """
    Error raised when the input parameters of the job are invalid or missing.
    """
    pass



class ExecuteJobExecutionError(Exception):
    """
    Error raised when the execution of the job fails for internal reasons.
    """
    pass



class ExecuteJobOutputError(Exception):
    """
    Error raised when the creation of the output container failed.
    """
    pass



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

    :raise ExecuteJobInputError: if ``job`` is not an instance of ``Job``
    """

    TAG = "ExecuteJob"

    def __init__(self, job=None, logger=None):
        self.job = job
        self.working_directory = None
        self.tmp_directory = None
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()
        if job is not None:
            self.load_job(self.job)

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def load_job(self, job):
        """
        Load the job from the given ``Job`` object.

        :param job: the job to load
        :type  job: :class:`aeneas.job.Job`

        :raise ExecuteJobInputError: if ``job`` is not an instance of ``Job``
        """
        if not isinstance(job, Job):
            self._failed("job is not an instance of Job", "input")
        self.job = job

    def load_job_from_container(self, container_path, config_string=None):
        """
        Load the job from the given ``Container`` object.

        If ``config_string`` is ``None``,
        the container must contain a configuration file;
        otherwise use the provided config string
        (i.e., the wizard case).

        :param container_path: the path to the input container
        :type  container_path: string (path)
        :param config_string: the configuration string (from wizard)
        :type  config_string: string

        :raise ExecuteJobInputError: if the given container does not contain a valid ``Job``
        """
        self._log("Loading job from container...")

        # create working directory where the input container
        # will be decompressed
        self.working_directory = tempfile.mkdtemp(dir=gf.custom_tmp_dir())
        self._log(["Created working directory '%s'", self.working_directory])

        try:
            self._log("Decompressing input container...")
            input_container = Container(container_path, logger=self.logger)
            input_container.decompress(self.working_directory)
            self._log("Decompressing input container... done")
        except Exception as exc:
            self.clean()
            self._failed("Unable to decompress container '%s': %s" % (container_path, str(exc)), "input")

        try:
            self._log("Creating job from working directory...")
            working_container = Container(
                self.working_directory,
                logger=self.logger
            )
            analyzer = AnalyzeContainer(working_container, logger=self.logger)
            if config_string is None:
                self.job = analyzer.analyze()
            else:
                self.job = analyzer.analyze_from_wizard(config_string)
            self._log("Creating job from working directory... done")
        except Exception as exc:
            self.clean()
            self._failed("Unable to analyze container '%s': %s" % (container_path, str(exc)), "input")

        if self.job is None:
            self._failed("The container '%s' does not contain a valid Job" % container_path, "input")

        try:
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

            self._log("Loading job from container: succeeded")
        except Exception as exc:
            self.clean()
            self._failed("Error while setting absolute paths for tasks: %s" % str(exc), "input")

    def execute(self):
        """
        Execute the job, that is, execute all of its tasks.

        Each produced sync map will be stored
        inside the corresponding task object.

        :raise ExecuteJobExecutionError: if there is a problem during the job execution
        """
        self._log("Executing job")

        if self.job is None:
            self._failed("The job object is None", "execution")
        if len(self.job) == 0:
            self._failed("The job has no tasks", "execution")
        self._log(["Number of tasks: '%d'", len(self.job)])

        for task in self.job.tasks:
            try:
                custom_id = task.configuration.custom_id
                self._log(["Executing task '%s'...", custom_id])
                executor = ExecuteTask(task, logger=self.logger)
                executor.execute()
                self._log(["Executing task '%s'... done", custom_id])
            except Exception as exc:
                self._failed("Error while executing task '%s': %s" % (custom_id, str(exc)), "execution")
            self._log("Executing task: succeeded")

        self._log("Executing job: succeeded")

    def write_output_container(self, output_directory_path):
        """
        Write the output container for this job.

        Return the path to output container.

        :param output_directory_path: the path to a directory where
                                      the output container must be created
        :type  output_directory_path: string (path)
        :rtype: string
        """
        self._log("Writing output container for this job")

        if self.job is None:
            self._failed("The job object is None", "output")
        if len(self.job) == 0:
            self._failed("The job has no tasks", "output")
        self._log(["Number of tasks: '%d'", len(self.job)])

        # create temporary directory where the sync map files
        # will be created
        # this temporary directory will be compressed into
        # the output container
        self.tmp_directory = tempfile.mkdtemp(dir=gf.custom_tmp_dir())
        self._log(["Created temporary directory '%s'", self.tmp_directory])

        for task in self.job.tasks:
            custom_id = task.configuration.custom_id

            # check if the task has sync map and sync map file path
            if task.sync_map_file_path is None:
                self._failed("Task '%s' has sync_map_file_path not set" % custom_id, "output")
            if task.sync_map is None:
                self._failed("Task '%s' has sync_map not set" % custom_id, "output")

            try:
                # output sync map
                self._log(["Outputting sync map for task '%s'...", custom_id])
                task.output_sync_map_file(self.tmp_directory)
                self._log(["Outputting sync map for task '%s'... done", custom_id])
            except Exception as exc:
                self._failed("Error while outputting sync map for task '%s': %s" % (custom_id, str(exc)), "output")

        # get output container info
        output_container_format = self.job.configuration.os_container_format
        self._log(["Output container format: '%s'", output_container_format])
        output_file_name = self.job.configuration.os_file_name
        if ((output_container_format != ContainerFormat.UNPACKED) and
                (not output_file_name.endswith(output_container_format))):
            self._log("Adding extension to output_file_name")
            output_file_name += "." + output_container_format
        self._log(["Output file name: '%s'", output_file_name])
        output_file_path = gf.norm_join(
            output_directory_path,
            output_file_name
        )
        self._log(["Output file path: '%s'", output_file_path])

        try:
            self._log("Compressing...")
            container = Container(
                output_file_path,
                output_container_format,
                logger=self.logger
            )
            container.compress(self.tmp_directory)
            self._log("Compressing... done")
            self._log(["Created output file: '%s'", output_file_path])
            self._log("Writing output container for this job: succeeded")
            self.clean(False)
            return output_file_path
        except Exception as exc:
            self.clean(False)
            self._failed(str(exc), "output")
            return None

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
        if remove_working_directory is not None:
            self._log("Removing working directory... ")
            gf.delete_directory(self.working_directory)
            self.working_directory = None
            self._log("Removing working directory... done")
        self._log("Removing temporary directory... ")
        gf.delete_directory(self.tmp_directory)
        self.tmp_directory = None
        self._log("Removing temporary directory... done")

    def _failed(self, msg, during="execution"):
        """ Bubble exception up """
        if during == "input":
            self._log(msg, Logger.CRITICAL)
            raise ExecuteJobInputError(msg)
        elif during == "output":
            self._log(msg, Logger.CRITICAL)
            raise ExecuteJobOutputError(msg)
        else:
            self._log(msg, Logger.CRITICAL)
            raise ExecuteJobExecutionError(msg)



