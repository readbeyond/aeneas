#!/usr/bin/env python
# coding=utf-8

"""
Wrapper around ``ffmpeg`` to convert audio files.
"""

import os
import subprocess

from aeneas.logger import Logger
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

class FFMPEGWrapper(object):
    """
    Wrapper around ``ffmpeg`` to convert audio files.

    It will perform a call like::

        $ ffmpeg -i /path/to/input.mp3 [parameters] /path/to/output.wav

    :param parameters: list of parameters (not counting input and output paths)
                       to be passed to ``ffmpeg``.
                       Default: ``FFMPEG_PARAMETERS_DEFAULT``.
    :type  parameters: list of strings
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    FFMPEG_SAMPLE_16000 = ["-ar", "16000"]
    """ Single parameter for ``ffmpeg``: 16000 Hz sampling """

    FFMPEG_SAMPLE_22050 = ["-ar", "22050"]
    """ Single parameter for ``ffmpeg``: 22050 Hz sampling """

    FFMPEG_SAMPLE_44100 = ["-ar", "44100"]
    """ Single parameter for ``ffmpeg``: 44100 Hz sampling """

    FFMPEG_MONO = ["-ac", "1"]
    """ Single parameter for ``ffmpeg``: mono (1 channel) """

    FFMPEG_STEREO = ["-ac", "2"]
    """ Single parameter for ``ffmpeg``: stereo (2 channels) """

    FFMPEG_OVERWRITE = ["-y"]
    """ Single parameter for ``ffmpeg``: force overwriting output file """

    FFMPEG_FORMAT_WAV = ["-f", "wav"]
    """ Single parameter for ``ffmpeg``: produce output in ``wav`` format
    (must be the second to last argument to ``ffmpeg``,
    just before path of the output file) """

    FFMPEG_PARAMETERS_SAMPLE_KEEP = (
        FFMPEG_MONO + FFMPEG_OVERWRITE + FFMPEG_FORMAT_WAV
    )
    """ Set of parameters for ``ffmpeg`` without changing the sampling rate """

    FFMPEG_PARAMETERS_SAMPLE_16000 = (
        FFMPEG_MONO + FFMPEG_SAMPLE_16000 + FFMPEG_OVERWRITE + FFMPEG_FORMAT_WAV
    )
    """ Set of parameters for ``ffmpeg`` with 16000 Hz sampling """

    FFMPEG_PARAMETERS_SAMPLE_22050 = (
        FFMPEG_MONO + FFMPEG_SAMPLE_22050 + FFMPEG_OVERWRITE + FFMPEG_FORMAT_WAV
    )
    """ Set of parameters for ``ffmpeg`` with 22050 Hz sampling """

    FFMPEG_PARAMETERS_SAMPLE_44100 = (
        FFMPEG_MONO + FFMPEG_SAMPLE_44100 + FFMPEG_OVERWRITE + FFMPEG_FORMAT_WAV
    )
    """ Set of parameters for ``ffmpeg`` with 44100 Hz sampling """

    FFMPEG_PARAMETERS_DEFAULT = FFMPEG_PARAMETERS_SAMPLE_22050
    """ Default set of parameters for ``ffmpeg`` """

    TAG = "FFMPEGWrapper"

    def __init__(self, parameters=FFMPEG_PARAMETERS_DEFAULT, logger=None):
        self.parameters = parameters
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()
        self._log(["Initialized with parameters '%s'", self.parameters])

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    @property
    def parameters(self):
        """
        The parameters to be passed to ffmpeg,
        not including ``-i input_file.mp3`` and ``output_file.wav``.

        :rtype: list of strings
        """
        return self.__parameters

    @parameters.setter
    def parameters(self, parameters):
        self.__parameters = parameters

    def convert(
            self,
            input_file_path,
            output_file_path,
            head_length=None,
            process_length=None
        ):
        """
        Convert the audio file at ``input_file_path``
        into ``output_file_path``,
        using the parameters set in the constructor
        or through the ``parameters`` property.

        You can skip the beginning of the audio file
        by specifying ``head_length`` seconds to skip
        (if it is ``None``, start at time zero),
        and you can specify to convert
        only ``process_length`` seconds
        (if it is ``None``, process the entire input file length).

        By specifying both ``head_length`` and ``process_length``,
        you can skip a portion at the beginning and at the end
        of the original input file.

        :param input_file_path: the path of the audio file to convert
        :type  input_file_path: string
        :param output_file_path: the path of the converted audio file
        :type  output_file_path: string
        :param head_length: skip these many seconds
                            from the beginning of the audio file
        :type  head_length: float
        :param process_length: process these many seconds of the audio file
        :type  process_length: float

        :raise IOError: if ``input_file_path`` does not exist or ``output_file_path`` cannot be written
        """
        # test if we can read the input file
        if not gf.file_exists(input_file_path):
            self._log(["Input file '%s' does not exist", input_file_path], Logger.CRITICAL)
            raise IOError("Input audio file does not exist")

        # test if we can write the output file
        if not gf.file_can_be_written(output_file_path):
            self._log(["Output file '%s' cannot be written", output_file_path], Logger.CRITICAL)
            raise IOError("Output audio file cannot be written")

        # call ffmpeg
        arguments = []
        arguments += [gc.FFMPEG_PATH]
        arguments += ["-i", input_file_path]
        if head_length is not None:
            arguments += ["-ss", head_length]
        if process_length is not None:
            arguments += ["-t", process_length]
        arguments += self.parameters
        arguments += [output_file_path]
        self._log(["Calling with arguments '%s'", arguments])
        proc = subprocess.Popen(
            arguments,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE)
        proc.communicate()
        proc.stdout.close()
        proc.stdin.close()
        proc.stderr.close()
        self._log("Call completed")

        # check if the output file exists
        if not gf.file_exists(output_file_path):
            self._log(["Output file '%s' was not written", output_file_path], Logger.CRITICAL)
            raise IOError("Output file was not written")

        # returning the output file path
        self._log(["Returning output file path '%s'", output_file_path])
        return output_file_path



