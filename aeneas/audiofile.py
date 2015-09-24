#!/usr/bin/env python
# coding=utf-8

"""
A class representing an audio file.
"""

import os
from scikits.audiolab import wavread

import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf
from aeneas.ffprobewrapper import FFPROBEWrapper
from aeneas.logger import Logger
from aeneas.mfcc import MFCC

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.1.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class AudioFile(object):
    """
    A class representing an audio file.

    The properties of the audio file
    (length, format, etc.)
    will be set by ``read_properties()``
    which will invoke an audio file probe.
    (Currently,
    :class:`aeneas.ffprobewrapper.FFPROBEWrapper`
    )

    If the file is a monoaural WAVE file,
    its data can be read and MFCCs can be extracted.

    :param file_path: the path to the audio file
    :type  file_path: string (path)
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = "AudioFile"

    def __init__(self, file_path, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()
        self.file_path = file_path
        self.file_size = None
        self.audio_data = None
        self.audio_length = None
        self.audio_format = None
        self.audio_sample_rate = None
        self.audio_channels = None
        self.audio_mfcc = None

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
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

    @property
    def audio_mfcc(self):
        """
        The MFCCs of the audio file.

        :rtype: numpy 2D array
        """
        return self.__audio_mfcc

    @audio_mfcc.setter
    def audio_mfcc(self, audio_mfcc):
        self.__audio_mfcc = audio_mfcc

    def read_properties(self):
        """
        Populate this object by reading
        the audio properties of the file at the given path.

        Currently this function uses
        :class:`aeneas.ffprobewrapper.FFPROBEWrapper`
        to get the audio file properties.
        """

        self._log("Reading properties")

        # check the file can be read
        if self.file_path is None:
            raise AttributeError("File path is None")
        if not os.path.isfile(self.file_path):
            self._log(["File '%s' cannot be read", self.file_path], Logger.CRITICAL)
            raise OSError("File cannot be read")

        # get the file size
        self._log(["Getting file size for '%s'", self.file_path])
        self.file_size = os.path.getsize(self.file_path)
        self._log(["File size for '%s' is '%d'", self.file_path, self.file_size])

        # get the audio properties
        self._log("Reading properties with FFPROBEWrapper...")
        prober = FFPROBEWrapper(logger=self.logger)
        properties = prober.read_properties(self.file_path)
        self._log("Reading properties with FFPROBEWrapper... done")

        # save relevant properties in results inside the audiofile object
        self.audio_length = gf.safe_float(properties[FFPROBEWrapper.STDOUT_DURATION])
        self._log(["Stored audio_length: '%s'", self.audio_length])
        self.audio_format = properties[FFPROBEWrapper.STDOUT_CODEC_NAME]
        self._log(["Stored audio_format: '%s'", self.audio_format])
        self.audio_sample_rate = gf.safe_int(properties[FFPROBEWrapper.STDOUT_SAMPLE_RATE])
        self._log(["Stored audio_sample_rate: '%s'", self.audio_sample_rate])
        self.audio_channels = gf.safe_int(properties[FFPROBEWrapper.STDOUT_CHANNELS])
        self._log(["Stored audio_channels: '%s'", self.audio_channels])

    def load_data(self):
        """
        Load the audio file data (works only for mono wav files)
        """
        self._log("Loading audio data")

        # check the file can be read
        if self.file_path is None:
            raise AttributeError("File path is None")
        if not os.path.isfile(self.file_path):
            self._log(["File '%s' cannot be read", self.file_path], Logger.CRITICAL)
            raise OSError("File cannot be read")

        self._log("Loading wav file...")
        self.audio_data, self.audio_sample_rate, self.audio_format = wavread(self.file_path)
        self.audio_length = (float(len(self.audio_data)) / self.audio_sample_rate)
        self._log(["Sample length: %f", self.audio_length])
        self._log(["Sample rate:   %f", self.audio_sample_rate])
        self._log(["Audio format:  %s", self.audio_format])
        self._log("Loading wav file... done")

    def extract_mfcc(self, frame_rate=gc.MFCC_FRAME_RATE):
        """
        Extract MFCCs from the given audio file.

        :param frame_rate: the MFCC frame rate, in frames per second. Default:
                           :class:`aeneas.globalconstants.MFCC_FRAME_RATE`
        :type  frame_rate: int
        """
        if gc.USE_C_EXTENSIONS:
            self._log("C extensions enabled in gc")
            if gf.can_run_c_extension("cmfcc"):
                self._log("C extensions enabled in gc and cmfcc can be loaded")
                try:
                    self._compute_mfcc_c_extension(frame_rate)
                    return
                except:
                    self._log(
                        "An error occurred running cmfcc",
                        severity=Logger.WARNING
                    )
            else:
                self._log("C extensions enabled in gc, but cmfcc cannot be loaded")
        else:
            self._log("C extensions disabled in gc")
        self._log("Running the pure Python code")
        try:
            self._compute_mfcc_pure_python(frame_rate)
        except:
            self._log(
                "An error occurred running _compute_mfcc_pure_python",
                severity=Logger.WARNING
            )

    def clear_data(self):
        """
        Clear the audio data, freeing memory.
        """
        self.audio_data = None

    def _compute_mfcc_c_extension(self, frame_rate):
        """
        Compute MFCCs using the Python C extension cmfcc.
        """
        self._log("Computing MFCCs using C extension...")
        self._log("Importing cmfcc...")
        import aeneas.cmfcc
        self._log("Importing cmfcc... done")
        self.audio_mfcc = aeneas.cmfcc.cmfcc_compute_mfcc(
            self.audio_data,
            self.audio_sample_rate,
            frame_rate,
            40,
            13,
            512,
            133.3333,
            6855.4976,
            0.97,
            0.0256
        ).transpose()
        self._log("Computing MFCCs using C extension... done")

    def _compute_mfcc_pure_python(self, frame_rate):
        """
        Compute MFCCs using the pure Python code.
        """
        self._log("Computing MFCCs using pure Python code...")
        extractor = MFCC(samprate=self.audio_sample_rate, frate=frame_rate)
        self.audio_mfcc = extractor.sig2s2mfc(self.audio_data).transpose()
        self._log("Computing MFCCs using pure Python code... done")



