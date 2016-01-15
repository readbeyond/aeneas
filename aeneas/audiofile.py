#!/usr/bin/env python
# coding=utf-8

"""
A class representing an audio file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy

from aeneas.ffprobewrapper import FFPROBEParsingError
from aeneas.ffprobewrapper import FFPROBEUnsupportedFormatError
from aeneas.ffprobewrapper import FFPROBEWrapper
from aeneas.logger import Logger
from aeneas.mfcc import MFCC
from aeneas.wavfile import read as scipywavread
from aeneas.wavfile import write as scipywavwrite
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.4.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class AudioFileUnsupportedFormatError(Exception):
    """
    Error raised when the format of the given file cannot be decoded.
    """
    pass



class AudioFile(object):
    """
    A class representing an audio file.

    The properties of the audio file (length, format, etc.)
    are set by invoking the ``read_properties()`` function,
    which calls an audio file probe.
    (Currently, the probe is :class:`aeneas.ffprobewrapper.FFPROBEWrapper`)

    :param file_path: the path of the audio file
    :type  file_path: Unicode string (path)
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = u"AudioFile"

    def __init__(self, file_path=None, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()
        self.file_path = file_path
        self.file_size = None
        self.audio_length = None
        self.audio_format = None
        self.audio_sample_rate = None
        self.audio_channels = None

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def __unicode__(self):
        msg = [
            u"File path:         %s" % self.file_path,
            u"File size (bytes): %s" % gf.safe_int(self.file_size),
            u"Audio length (s):  %s" % gf.safe_float(self.audio_length),
            u"Audio format:      %s" % self.audio_format,
            u"Audio sample rate: %s" % gf.safe_int(self.audio_sample_rate),
            u"Audio channels:    %s" % gf.safe_int(self.audio_channels),
        ]
        return u"\n".join(msg)

    def __str__(self):
        return gf.safe_str(self.__unicode__())

    @property
    def file_path(self):
        """
        The path of the audio file.

        :rtype: Unicode string
        """
        return self.__file_path
    @file_path.setter
    def file_path(self, file_path):
        self.__file_path = file_path

    @property
    def file_size(self):
        """
        The size of the audio file, in bytes.

        :rtype: int
        """
        return self.__file_size
    @file_size.setter
    def file_size(self, file_size):
        self.__file_size = file_size

    @property
    def audio_length(self):
        """
        The length of the audio file, in seconds.

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

        :rtype: Unicode string
        """
        return self.__audio_format
    @audio_format.setter
    def audio_format(self, audio_format):
        self.__audio_format = audio_format

    @property
    def audio_sample_rate(self):
        """
        The sample rate of the audio file, in samples per second.

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

    def read_properties(self):
        """
        Populate this object by reading
        the audio properties of the file at the given path.

        Currently this function uses
        :class:`aeneas.ffprobewrapper.FFPROBEWrapper`
        to get the audio file properties.

        :raises AudioFileUnsupportedFormatError: if the audio file has a format not supported
        :raises OSError: if the audio file cannot be read
        """

        self._log(u"Reading properties...")

        # check the file can be read
        if not gf.file_can_be_read(self.file_path):
            self._log([u"File '%s' cannot be read", self.file_path], Logger.CRITICAL)
            raise OSError(u"File '%s' cannot be read" % self.file_path)

        # get the file size
        self._log([u"Getting file size for '%s'", self.file_path])
        self.file_size = gf.file_size(self.file_path)
        self._log([u"File size for '%s' is '%d'", self.file_path, self.file_size])

        # get the audio properties using FFPROBEWrapper
        try:
            self._log(u"Reading properties with FFPROBEWrapper...")
            properties = FFPROBEWrapper(logger=self.logger).read_properties(self.file_path)
            self._log(u"Reading properties with FFPROBEWrapper... done")
        except FFPROBEUnsupportedFormatError:
            self._log(u"Reading properties with FFPROBEWrapper... failed", Logger.CRITICAL)
            self._log(u"Unsupported audio file format", Logger.CRITICAL)
            raise AudioFileUnsupportedFormatError("Unsupported audio file format")
        except FFPROBEParsingError:
            self._log(u"Reading properties with FFPROBEWrapper... failed", Logger.CRITICAL)
            self._log(u"Failed while parsing the ffprobe output", Logger.CRITICAL)
            raise AudioFileUnsupportedFormatError("Unsupported audio file format")

        # save relevant properties in results inside the audiofile object
        self.audio_length = gf.safe_float(properties[FFPROBEWrapper.STDOUT_DURATION])
        self.audio_format = properties[FFPROBEWrapper.STDOUT_CODEC_NAME]
        self.audio_sample_rate = gf.safe_int(properties[FFPROBEWrapper.STDOUT_SAMPLE_RATE])
        self.audio_channels = gf.safe_int(properties[FFPROBEWrapper.STDOUT_CHANNELS])
        self._log([u"Stored audio_length: '%s'", self.audio_length])
        self._log([u"Stored audio_format: '%s'", self.audio_format])
        self._log([u"Stored audio_sample_rate: '%s'", self.audio_sample_rate])
        self._log([u"Stored audio_channels: '%s'", self.audio_channels])
        self._log(u"Reading properties... done")



class AudioFileMonoWAVE(AudioFile):
    """
    A monoaural (single-channel) WAVE audio file.

    Its data can be read from and write to file, set from a ``numpy`` 1D array.

    It supports append, prepend, reverse, and trim operations.

    It can also extract MFCCs and store them internally,
    also after the audio data has been discarded.

    NOTE
    At the moment, the state of this object might be inconsistent
    (e.g., setting a new path after loading audio data will not flush the audio data).
    Use this class with care.

    :param file_path: the path of the audio file
    :type  file_path: Unicode string (path)
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = u"AudioFileMonoWAVE"

    def __init__(self, file_path=None, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()
        self.audio_data = None
        self.audio_mfcc = None
        AudioFile.__init__(self, file_path=file_path, logger=logger)

    @property
    def audio_data(self):
        """
        The audio data.

        :rtype: numpy 1D array
        """
        return self.__audio_data
    @audio_data.setter
    def audio_data(self, audio_data):
        self.__audio_data = audio_data

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

    def load_data(self):
        """
        Load the audio file data.

        :raises AudioFileUnsupportedFormatError: if the audio file is not a mono WAVE file
        :raises OSError: if the audio file cannot be read
        """
        self._log(u"Loading audio data...")

        # check the file can be read
        if not gf.file_can_be_read(self.file_path):
            self._log([u"File '%s' cannot be read", self.file_path], Logger.CRITICAL)
            raise OSError("File '%s' cannot be read" % self.file_path)

        try:
            self.audio_format = "pcm16"
            self.audio_sample_rate, self.audio_data = scipywavread(self.file_path)
            # scipy => [-32768..32767]
            self.audio_data = self.audio_data.astype("float64") / 32768
        except ValueError:
            self._log(u"Unsupported audio file format", Logger.CRITICAL)
            raise AudioFileUnsupportedFormatError("Unsupported audio file format")

        self._update_length()
        self._log([u"Sample length: %f", self.audio_length])
        self._log([u"Sample rate:   %f", self.audio_sample_rate])
        self._log([u"Audio format:  %s", self.audio_format])
        self._log(u"Loading audio data... done")

    def append_data(self, new_data):
        """
        Append the given new data to the current audio data.

        If audio data is not loaded, create an empty data structure
        and then append to it.

        :param new_data: the new data to be appended
        :type  new_data: numpy 1D array

        .. versionadded:: 1.2.1
        """
        self._log(u"Appending audio data...")
        self._audio_data_is_initialized(load=False)
        self.audio_data = numpy.append(self.audio_data, new_data)
        self._update_length()
        self._log(u"Appending audio data... done")

    def prepend_data(self, new_data):
        """
        Prepend the given new data to the current audio data.

        If audio data is not loaded, create an empty data structure
        and then preppend to it.

        :param new_data: the new data to be prepended
        :type  new_data: numpy 1D array

        .. versionadded:: 1.2.1
        """
        self._log(u"Prepending audio data...")
        self._audio_data_is_initialized(load=False)
        self.audio_data = numpy.append(new_data, self.audio_data)
        self._update_length()
        self._log(u"Prepending audio data... done")

    def extract_mfcc(
            self,
            frame_rate=gc.MFCC_FRAME_RATE,
            force_pure_python=False
    ):
        """
        Extract MFCCs from the given audio file.

        If audio data is not loaded, load it, extract MFCCs,
        store them internally, and discard the audio data immediately.

        :param frame_rate: the MFCC frame rate, in frames per second.
                           Default: :class:`aeneas.globalconstants.MFCC_FRAME_RATE`.
        :type  frame_rate: int
        :param force_pure_python: force using the pure Python version
        :type  force_pure_python: bool
        :raise RuntimeError: if both the C extension and
                             the pure Python code did not succeed.
        """
        had_audio_data = self._audio_data_is_initialized(load=True)
        gf.run_c_extension_with_fallback(
            self._log,
            "cmfcc",
            self._compute_mfcc_c_extension,
            self._compute_mfcc_pure_python,
            (frame_rate,),
            force_pure_python=force_pure_python
        )
        if not had_audio_data:
            self._log(u"Audio data was not loaded, clearing it")
            self.clear_data()
        else:
            self._log(u"Audio data was loaded, not clearing it")

    def reverse(self):
        """
        Reverse the audio data.

        If audio data is not loaded, load it and then reverse it.

        .. versionadded:: 1.2.0
        """
        self._log(u"Reversing...")
        self._audio_data_is_initialized(load=True)
        self.audio_data = self.audio_data[::-1]
        self._log(u"Reversing... done")

    def trim(self, begin=None, length=None):
        """
        Get a slice of the audio data of ``length`` seconds,
        starting from ``begin`` seconds.

        If audio data is not loaded, load it and then slice it.

        :param begin: the start position, in seconds
        :type  begin: float
        :param length: the  position, in seconds
        :type  length: float

        .. versionadded:: 1.2.0
        """
        self._log(u"Trimming...")
        if (begin is None) and (length is None):
            self._log(u"begin and length are both None: nothing to do")
        else:
            self._audio_data_is_initialized(load=True)
            self._log([u"audio_length is %.3f", self.audio_length])
            if begin is None:
                begin = 0
                self._log([u"begin was None, now set to %.3f", begin])
            begin = min(max(0, begin), self.audio_length)
            self._log([u"begin is %.3f", begin])
            if length is None:
                length = self.audio_length - begin
                self._log([u"length was None, now set to %.3f", length])
            length = min(max(0, length), self.audio_length - begin)
            self._log([u"length is %.3f", length])
            begin_index = int(begin * self.audio_sample_rate)
            end_index = int((begin + length) * self.audio_sample_rate)
            self.audio_data = self.audio_data[begin_index:end_index]
            self._update_length()
        self._log(u"Trimming... done")

    def write(self, file_path):
        """
        Write the audio data to file.
        Return ``True`` on success, or ``False`` otherwise.

        :param file_path: the path of the output file to be written
        :type  file_path: Unicode string (path)

        .. versionadded:: 1.2.0
        """
        self._log([u"Writing audio file '%s'...", file_path])
        self._audio_data_is_initialized(load=False)
        try:
            # scipy => [-32768..32767]
            data = (self.audio_data * 32768).astype("int16")
            scipywavwrite(file_path, self.audio_sample_rate, data)
        except:
            self._log(u"Error writing audio file", severity=Logger.CRITICAL)
            raise OSError("Error writing audio file to '%s'" % file_path)
        self._log([u"Writing audio file '%s'... done", file_path])

    def clear_data(self):
        """
        Clear the audio data, freeing memory.
        """
        self._log(u"Clear audio_data")
        self.audio_data = None

    def _update_length(self):
        """
        Update the audio length property,
        according to the length of the current audio data
        and audio sample rate.

        This function fails silently if one of the two is None.
        """
        if (self.audio_sample_rate is not None) and (self.audio_data is not None):
            self.audio_length = len(self.audio_data) / self.audio_sample_rate

    def _audio_data_is_initialized(self, load=True):
        """
        Check if audio data is loaded:
        if so, return True.

        Otherwise, either load or initialize the audio data
        and return False.

        :param load: if True, load from file; if False, initialize to empty
        :type  load: bool
        :rtype: bool
        """
        if self.audio_data is not None:
            self._log(u"audio data is not None: returning True")
            return True
        if load:
            self._log(u"No audio data: loading it from file")
            self.load_data()
        else:
            self._log(u"No audio data: initializing it to an empty data structure")
            self.audio_data = numpy.array([])
        self._log(u"audio data was None: returning False")
        return False

    def _compute_mfcc_c_extension(self, frame_rate):
        """
        Compute MFCCs using the Python C extension cmfcc.
        """
        self._log(u"Computing MFCCs using C extension...")
        try:
            self._log(u"Importing cmfcc...")
            import aeneas.cmfcc
            self._log(u"Importing cmfcc... done")
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
            self._log(u"Computing MFCCs using C extension... done")
            return (True, None)
        except Exception as exc:
            self._log(u"Computing MFCCs using C extension... failed")
            self._log(u"An unexpected exception occurred while running cmfcc:", Logger.WARNING)
            self._log([u"%s", exc], Logger.WARNING)
        return (False, None)

    def _compute_mfcc_pure_python(self, frame_rate):
        """
        Compute MFCCs using the pure Python code.
        """
        self._log(u"Computing MFCCs using pure Python code...")
        try:
            self.audio_mfcc = MFCC(
                samprate=self.audio_sample_rate,
                frate=frame_rate
            ).sig2s2mfc(self.audio_data).transpose()
            self._log(u"Computing MFCCs using pure Python code... done")
            return (True, None)
        except Exception as exc:
            self._log(u"Computing MFCCs using pure Python code... failed")
            self._log(u"An unexpected exception occurred while running pure Python code:", Logger.WARNING)
            self._log([u"%s", exc], Logger.WARNING)
        return (False, None)



