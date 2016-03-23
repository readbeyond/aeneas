#!/usr/bin/env python
# coding=utf-8

"""
This module contains the following classes:

* :class:`~aeneas.executejob.ExecuteJob`, a class to process a job;
* :class:`~aeneas.executejob.ExecuteJobExecutionError`,
* :class:`~aeneas.executejob.ExecuteJobInputError`, and
* :class:`~aeneas.executejob.ExecuteJobOutputError`,
  representing errors generated while processing jobs.
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.analyzecontainer import AnalyzeContainer
from aeneas.container import Container
from aeneas.container import ContainerFormat
from aeneas.executetask import ExecuteTask
from aeneas.job import Job
from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration
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

class ExecuteJobExecutionError(Exception):
    """
    Error raised when the execution of the job fails for internal reasons.
    """
    pass



class ExecuteJobInputError(Exception):
    """
    Error raised when the input parameters of the job are invalid or missing.
    """
    pass



class ExecuteJobOutputError(Exception):
    """
    Error raised when the creation of the output container failed.
    """
    pass



class ExecuteJob(Loggable):
    """
    Execute a job, that is, execute all of its tasks
    and generate the output container
    holding the generated sync maps.

    If you do not provide a job object in the constructor,
    you must manually set it later, or load it from a container
    with :func:`~aeneas.executejob.ExecuteJob.load_job_from_container`.

    In the first case, you are responsible for setting
    the absolute audio/text/sync map paths of each task of the job,
    to their actual absolute location on the computing machine.
    Moreover, you are responsible for cleaning up
    any temporary files you might have generated around.

    In the second case, you are responsible for
    calling :func:`~aeneas.executejob.ExecuteJob.clean`
    at the end of the job execution,
    to delete the working directory
    created by :func:`~aeneas.executejob.ExecuteJob.load_job_from_container`
    when creating the job object.

    :param job: the job to be executed
    :type  job: :class:`~aeneas.job.Job`
    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    :raises: :class:`~aeneas.executejob.ExecuteJobInputError`: if ``job`` is not an instance of ``Job``
    """

    TAG = u"ExecuteJob"

    def __init__(self, job=None, rconf=None, logger=None):
        super(ExecuteJob, self).__init__(rconf=rconf, logger=logger)
        self.job = job
        self.working_directory = None
        self.tmp_directory = None
        if job is not None:
            self.load_job(self.job)

    def load_job(self, job):
        """
        Load the job from the given ``Job`` object.

        :param job: the job to load
        :type  job: :class:`~aeneas.job.Job`
        :raises: :class:`~aeneas.executejob.ExecuteJobInputError`: if ``job`` is not an instance of :class:`~aeneas.job.Job`
        """
        if not isinstance(job, Job):
            self.log_exc(u"job is not an instance of Job", None, True, ExecuteJobInputError)
        self.job = job

    def load_job_from_container(self, container_path, config_string=None):
        """
        Load the job from the given :class:`aeneas.container.Container` object.

        If ``config_string`` is ``None``,
        the container must contain a configuration file;
        otherwise use the provided config string
        (i.e., the wizard case).

        :param string container_path: the path to the input container
        :param string config_string: the configuration string (from wizard)
        :raises: :class:`~aeneas.executejob.ExecuteJobInputError`: if the given container does not contain a valid :class:`~aeneas.job.Job`
        """
        self.log(u"Loading job from container...")

        # create working directory where the input container
        # will be decompressed
        self.working_directory = gf.tmp_directory(root=self.rconf[RuntimeConfiguration.TMP_PATH])
        self.log([u"Created working directory '%s'", self.working_directory])

        try:
            self.log(u"Decompressing input container...")
            input_container = Container(container_path, logger=self.logger)
            input_container.decompress(self.working_directory)
            self.log(u"Decompressing input container... done")
        except Exception as exc:
            self.clean()
            self.log_exc(u"Unable to decompress container '%s': %s" % (container_path, exc), None, True, ExecuteJobInputError)

        try:
            self.log(u"Creating job from working directory...")
            working_container = Container(
                self.working_directory,
                logger=self.logger
            )
            analyzer = AnalyzeContainer(working_container, logger=self.logger)
            self.job = analyzer.analyze(config_string=config_string)
            self.log(u"Creating job from working directory... done")
        except Exception as exc:
            self.clean()
            self.log_exc(u"Unable to analyze container '%s': %s" % (container_path, exc), None, True, ExecuteJobInputError)

        if self.job is None:
            self.log_exc(u"The container '%s' does not contain a valid Job" % (container_path), None, True, ExecuteJobInputError)

        try:
            # set absolute path for text file and audio file
            # for each task in the job
            self.log(u"Setting absolute paths for tasks...")
            for task in self.job.tasks:
                task.text_file_path_absolute = gf.norm_join(
                    self.working_directory,
                    task.text_file_path
                )
                task.audio_file_path_absolute = gf.norm_join(
                    self.working_directory,
                    task.audio_file_path
                )
            self.log(u"Setting absolute paths for tasks... done")

            self.log(u"Loading job from container: succeeded")
        except Exception as exc:
            self.clean()
            self.log_exc(u"Error while setting absolute paths for tasks", exc, True, ExecuteJobInputError)

    def execute(self):
        """
        Execute the job, that is, execute all of its tasks.

        Each produced sync map will be stored
        inside the corresponding task object.

        :raises: :class:`~aeneas.executejob.ExecuteJobExecutionError`: if there is a problem during the job execution
        """
        self.log(u"Executing job")

        if self.job is None:
            self.log_exc(u"The job object is None", None, True, ExecuteJobExecutionError)
        if len(self.job) == 0:
            self.log_exc(u"The job has no tasks", None, True, ExecuteJobExecutionError)
        job_max_tasks = self.rconf[RuntimeConfiguration.JOB_MAX_TASKS]
        if (job_max_tasks > 0) and (len(self.job) > job_max_tasks):
            self.log_exc(u"The Job has %d Tasks, more than the maximum allowed (%d)." % (len(self.job), job_max_tasks), None, True, ExecuteJobExecutionError)
        self.log([u"Number of tasks: '%d'", len(self.job)])

        for task in self.job.tasks:
            try:
                custom_id = task.configuration["custom_id"]
                self.log([u"Executing task '%s'...", custom_id])
                executor = ExecuteTask(task, rconf=self.rconf, logger=self.logger)
                executor.execute()
                self.log([u"Executing task '%s'... done", custom_id])
            except Exception as exc:
                self.log_exc(u"Error while executing task '%s'" % (custom_id), exc, True, ExecuteJobExecutionError)
            self.log(u"Executing task: succeeded")

        self.log(u"Executing job: succeeded")

    def write_output_container(self, output_directory_path):
        """
        Write the output container for this job.

        Return the path to output container,
        which is the concatenation of ``output_directory_path``
        and of the output container file or directory name.

        :param string output_directory_path: the path to a directory where
                                             the output container must be created
        :rtype: string
        :raises: :class:`~aeneas.executejob.ExecuteJobOutputError`: if there is a problem while writing the output container
        """
        self.log(u"Writing output container for this job")

        if self.job is None:
            self.log_exc(u"The job object is None", None, True, ExecuteJobOutputError)
        if len(self.job) == 0:
            self.log_exc(u"The job has no tasks", None, True, ExecuteJobOutputError)
        self.log([u"Number of tasks: '%d'", len(self.job)])

        # create temporary directory where the sync map files
        # will be created
        # this temporary directory will be compressed into
        # the output container
        self.tmp_directory = gf.tmp_directory(root=self.rconf[RuntimeConfiguration.TMP_PATH])
        self.log([u"Created temporary directory '%s'", self.tmp_directory])

        for task in self.job.tasks:
            custom_id = task.configuration["custom_id"]

            # check if the task has sync map and sync map file path
            if task.sync_map_file_path is None:
                self.log_exc(u"Task '%s' has sync_map_file_path not set" % (custom_id), None, True, ExecuteJobOutputError)
            if task.sync_map is None:
                self.log_exc(u"Task '%s' has sync_map not set" % (custom_id), None, True, ExecuteJobOutputError)

            try:
                # output sync map
                self.log([u"Outputting sync map for task '%s'...", custom_id])
                task.output_sync_map_file(self.tmp_directory)
                self.log([u"Outputting sync map for task '%s'... done", custom_id])
            except Exception as exc:
                self.log_exc(u"Error while outputting sync map for task '%s'" % (custom_id), None, True, ExecuteJobOutputError)

        # get output container info
        output_container_format = self.job.configuration["o_container_format"]
        self.log([u"Output container format: '%s'", output_container_format])
        output_file_name = self.job.configuration["o_name"]
        if ((output_container_format != ContainerFormat.UNPACKED) and
                (not output_file_name.endswith(output_container_format))):
            self.log(u"Adding extension to output_file_name")
            output_file_name += "." + output_container_format
        self.log([u"Output file name: '%s'", output_file_name])
        output_file_path = gf.norm_join(
            output_directory_path,
            output_file_name
        )
        self.log([u"Output file path: '%s'", output_file_path])

        try:
            self.log(u"Compressing...")
            container = Container(
                output_file_path,
                output_container_format,
                logger=self.logger
            )
            container.compress(self.tmp_directory)
            self.log(u"Compressing... done")
            self.log([u"Created output file: '%s'", output_file_path])
            self.log(u"Writing output container for this job: succeeded")
            self.clean(False)
            return output_file_path
        except Exception as exc:
            self.clean(False)
            self.log_exc(u"Error while compressing", exc, True, ExecuteJobOutputError)
            return None

    def clean(self, remove_working_directory=True):
        """
        Remove the temporary directory.
        If ``remove_working_directory`` is ``True``
        remove the working directory as well,
        otherwise just remove the temporary directory.

        :param bool remove_working_directory: if ``True``, remove
                                              the working directory as well
        """
        if remove_working_directory is not None:
            self.log(u"Removing working directory... ")
            gf.delete_directory(self.working_directory)
            self.working_directory = None
            self.log(u"Removing working directory... done")
        self.log(u"Removing temporary directory... ")
        gf.delete_directory(self.tmp_directory)
        self.tmp_directory = None
        self.log(u"Removing temporary directory... done")



