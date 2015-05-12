#!/usr/bin/env python
# coding=utf-8

"""
A class representing an audio file.
"""

import os

import aeneas.globalfunctions as gf
from aeneas.ffprobewrapper import FFPROBEWrapper
from aeneas.logger import Logger

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl (www.readbeyond.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.0.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class AudioFile(object):
    """
    A class representing an audio file.

    The properties of the audio file
    (length, format, etc.)
    will be set by the constructor
    invoking an audio file probe.
    (Currently,
    :class:`aeneas.ffprobewrapper.FFPROBEWrapper`
    )

    :param file_path: the path to the audio file
    :type  file_path: string (path)
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = "AudioFile"

    def __init__(self, file_path, logger=None):
        self.logger = logger
        if self.logger == None:
            self.logger = Logger()
        self.file_path = file_path
        self.file_size = None
        self.audio_length = None
        self.audio_format = None
        self.audio_sample_rate = None
        self.audio_channels = None
        self._read_properties()

    def _log(self, message, severity=Logger.DEBUG):
        self.logger.log(message, severity, self.TAG)

    def __str__(self):
        accumulator = ""
        accumulator += "File path:         %s\n" % self.file_path
        accumulator += "File size (bytes): %s\n" % gf.safe_int(self.file_size)
        accumulator += "Audio length (s):  %s\n" % gf.safe_float(self.audio_length)
        accumulator += "Audio format:      %s\n" % self.audio_format
        accumulator += "Audio sample rate: %s\n" % gf.safe_int(self.audio_sample_rate)
        accumulator += "Audio channels:    %s" % gf.safe_int(self.audio_channels)
        return accumulator

    @property
    def file_path(self):
        """
        The path of the audio file.

        :rtype: string
        """
        return self.__file_path
    @file_path.setter
    def file_path(self, file_path):
        self.__file_path = file_path

    @property
    def file_size(self):
        """
        The size, in bytes, of the audio file.

        :rtype: int
        """
        return self.__file_size
    @file_size.setter
    def file_size(self, file_size):
        self.__file_size = file_size

    @property
    def audio_length(self):
        """
        The length, in seconds, of the audio file.

        :rtype: float
        """
        return self.__audio_length
    @audio_length.setter
    def audio_length(self, audio_length):
        self.__audio_length = audio_length

    @property
    def audio_format(self):
        """
        The format of the audio file.

        :rtype: string
        """
        return self.__audio_format
    @audio_format.setter
    def audio_format(self, audio_format):
        self.__audio_format = audio_format

    @property
    def audio_sample_rate(self):
        """
        The sample rate of the audio file.

        :rtype: int
        """
        return self.__audio_sample_rate
    @audio_sample_rate.setter
    def audio_sample_rate(self, audio_sample_rate):
        self.__audio_sample_rate = audio_sample_rate

    @property
    def audio_channels(self):
        """
        The number of channels of the audio file.

        :rtype: int
        """
        return self.__audio_channels
    @audio_channels.setter
    def audio_channels(self, audio_channels):
        self.__audio_channels = audio_channels

    def _read_properties(self):
        """
        Populate this object by reading
        the audio properties of the file at the given path.

        Currently this function uses
        :class:`aeneas.ffprobewrapper.FFPROBEWrapper`
        to get the audio file properties.
        """

        self._log("Reading properties")

        # check the file can be read
        if not os.path.isfile(self.file_path):
            msg = "File '%s' cannot be read" % self.file_path
            self._log(msg, Logger.CRITICAL)
            raise OSError(msg)

        # get the file size
        self._log("Getting file size for '%s'" % self.file_path)
        self.file_size = os.path.getsize(self.file_path)
        self._log("File size for '%s' is '%d'" % (self.file_path, self.file_size))

        # get the audio properties
        self._log("Reading properties with FFPROBEWrapper...")
        prober = FFPROBEWrapper(logger=self.logger)
        properties = prober.read_properties(self.file_path)
        self._log("Reading properties with FFPROBEWrapper... done")

        # save relevant properties in results inside the audiofile object
        self.audio_length = gf.safe_float(properties[FFPROBEWrapper.STDOUT_DURATION])
        self._log("Stored audio_length: '%s'" % self.audio_length)
        self.audio_format = properties[FFPROBEWrapper.STDOUT_CODEC_NAME]
        self._log("Stored audio_format: '%s'" % self.audio_format)
        self.audio_sample_rate = gf.safe_int(properties[FFPROBEWrapper.STDOUT_SAMPLE_RATE])
        self._log("Stored audio_sample_rate: '%s'" % self.audio_sample_rate)
        self.audio_channels = gf.safe_int(properties[FFPROBEWrapper.STDOUT_CHANNELS])
        self._log("Stored audio_channels: '%s'" % self.audio_channels)



