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
from aeneas.ffprobewrapper import FFPROBEPathError
from aeneas.ffprobewrapper import FFPROBEUnsupportedFormatError
from aeneas.ffprobewrapper import FFPROBEWrapper
from aeneas.logger import Logger
from aeneas.mfcc import MFCC
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.wavfile import read as scipywavread
from aeneas.wavfile import write as scipywavwrite
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

class AudioFileProbeError(Exception):
    """
    Error raised when the probe executable cannot be executed.
    """
    pass



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

    :param string file_path: the path of the audio file
    :param rconf: a runtime configuration. Default: ``None``, meaning that
                  default settings will be used.
    :type  rconf: :class:`aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = u"AudioFile"

    def __init__(self, file_path=None, rconf=None, logger=None):
        self.logger = logger if logger is not None else Logger()
        self.rconf = rconf if rconf is not None else RuntimeConfiguration()
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

        :rtype: string
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

        :rtype: string
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

        :raises AudioFileProbeError: if the path to the ``ffprobe`` executable cannot be called
        :raises AudioFileUnsupportedFormatError: if the audio file has a format not supported
        :raises OSError: if the audio file cannot be read
        """

        self._log(u"Reading properties...")

        # check the file can be read
        if not gf.file_can_be_read(self.file_path):
            self._log([u"File '%s' cannot be read", self.file_path], Logger.CRITICAL)
            raise OSError("File '%s' cannot be read" % self.file_path)

        # get the file size
        self._log([u"Getting file size for '%s'", self.file_path])
        self.file_size = gf.file_size(self.file_path)
        self._log([u"File size for '%s' is '%d'", self.file_path, self.file_size])

        # get the audio properties using FFPROBEWrapper
        try:
            self._log(u"Reading properties with FFPROBEWrapper...")
            properties = FFPROBEWrapper(rconf=self.rconf, logger=self.logger).read_properties(self.file_path)
            self._log(u"Reading properties with FFPROBEWrapper... done")
        except FFPROBEPathError:
            self._log(u"Reading properties with FFPROBEWrapper... failed", Logger.CRITICAL)
            self._log(u"Unable to call ffprobe executable", Logger.CRITICAL)
            raise AudioFileProbeError("Unable to call the audio probe executable")
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



class AudioFileMonoWAVENotInitialized(Exception):
    """
    Error raised when trying to access audio samples from
    an AudioFileMonoWAVE not initialized yet.
    """
    pass



class AudioFileMonoWAVE(AudioFile):
    """
    A monoaural (single-channel) WAVE audio file,
    represented as a numpy 1D array of float64 values in [-1, 1].

    The audio samples can be read from file and written to file.

    It supports append, prepend, reverse, and trim operations.

    Memory can be pre-allocated to avoid memory trashing while
    performing many append operations.

    If ``file_path`` is not ``None``, the audio samples
    will be read upon creation of the object.

    :param string file_path: the path of the audio file
    :param rconf: a runtime configuration. Default: ``None``, meaning that
                  default settings will be used.
    :type  rconf: :class:`aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = u"AudioFileMonoWAVE"

    def __init__(self, file_path=None, rconf=None, logger=None):
        self.logger = logger if logger is not None else Logger()
        self.rconf = rconf if rconf is not None else RuntimeConfiguration()
        self.__samples_capacity = 0
        self.__samples_length = 0
        self.__samples = None
        AudioFile.__init__(self, file_path=file_path, rconf=rconf, logger=logger)
        if self.file_path is not None:
            self.read_samples_from_file()

    def __unicode__(self):
        msg = [
            u"File path:         %s" % self.file_path,
            u"File size (bytes): %s" % gf.safe_int(self.file_size),
            u"Audio length (s):  %s" % gf.safe_float(self.audio_length),
            u"Audio format:      %s" % self.audio_format,
            u"Audio sample rate: %s" % gf.safe_int(self.audio_sample_rate),
            u"Audio channels:    %s" % gf.safe_int(self.audio_channels),
            u"Samples capacity:  %s" % gf.safe_int(self.__samples_capacity),
            u"Samples length:    %s" % gf.safe_int(self.__samples_length),
        ]
        return u"\n".join(msg)

    def __str__(self):
        return gf.safe_str(self.__unicode__())

    @property
    def audio_samples(self):
        """
        The audio audio_samples, that is, an array of float64 values,
        each representing an audio sample.

        Note that this function returns a view into the
        first ``self.__samples_length`` elements of ``self.__samples``.
        If you want to clone the values,
        you must use e.g. ``numpy.array(audiofile.audio_samples)``.

        :rtype: numpy 1D array view
        :raises AudioFileMonoWAVENotInitialized: if the audio file is not initialized yet 
        """
        if self.__samples is None:
            raise AudioFileMonoWAVENotInitialized("The AudioFileMonoWAVE is not initialized")
        return self.__samples[0:self.__samples_length]

    def read_samples_from_file(self):
        """
        Load the audio samples from file.

        :raises AudioFileUnsupportedFormatError: if the audio file is not a mono WAVE file
        :raises OSError: if the audio file cannot be read
        """
        self._log(u"Loading audio data...")

        # check the file can be read
        if not gf.file_can_be_read(self.file_path):
            self._log([u"File '%s' cannot be read", self.file_path], Logger.CRITICAL)
            raise OSError("File '%s' cannot be read" % self.file_path)

        # TODO allow calling C extension cwave
        try:
            self.audio_format = "pcm16"
            self.audio_sample_rate, self.__samples = scipywavread(self.file_path)
            # scipy reads a sample as an int16_t, that is, a number in [-32768, 32767]
            # so we convert it to a float64 in [-1, 1]
            self.__samples = self.__samples.astype("float64") / 32768
            self.__samples_capacity = len(self.__samples)
            self.__samples_length = self.__samples_capacity
            # set the precise audio length
            self.audio_length = float(self.__samples_length) / self.audio_sample_rate
        except ValueError:
            self._log(u"Unsupported audio file format", Logger.CRITICAL)
            raise AudioFileUnsupportedFormatError("Unsupported audio file format")

        self._update_length()
        self._log([u"Sample length: %f", self.audio_length])
        self._log([u"Sample rate:   %f", self.audio_sample_rate])
        self._log([u"Audio format:  %s", self.audio_format])
        self._log(u"Loading audio data... done")

    def preallocate_memory(self, new_capacity):
        """
        Preallocate memory to store audio samples,
        to avoid repeated new allocations and copies
        while performing several consecutive append operations.

        If ``self.__samples`` is not initialized,
        it will become an array of ``new_capacity`` zeros.

        If ``new_capacity`` is larger than the current capacity,
        the current ``self.__samples`` will be extended with zeros.

        If ``new_capacity`` is smaller than the current capacity,
        the first ``new_capacity`` values of ``self.__samples``
        will be retained.

        :param int new_capacity: the new capacity, in number of samples

        :raises ValueError: if ``new_capacity`` is negative

        .. versionadded:: 1.5.0
        """
        if new_capacity < 0:
            raise ValueError("The capacity value cannot be negative")
        if self.__samples is None:
            self._log(u"self.__samples is not initialized")
            self.__samples = numpy.zeros(new_capacity)
            self.__samples_length = 0
        else:
            self._log([u"Previous sample capacity was %d samples", self.__samples_capacity])
            self._log([u"Previous sample length was %d samples", self.__samples_length])
            self.__samples = numpy.resize(self.__samples, new_capacity)
            self.__samples_length = min(self.__samples_length, new_capacity)
        self._log([u"New sample length   is %d", self.__samples_length])
        self._log([u"New sample capacity is %d", new_capacity])
        self.__samples_capacity = new_capacity

    def append(self, new_data, reverse=False):
        """
        Append the given new data to the current audio data.

        This function initializes the memory if no audio data
        is present already.

        If ``reverse`` is ``True``, the new audio data
        will be reversed and then appended.

        :param new_data: the new data to be appended
        :type  new_data: numpy 1D array
        :param bool reverse: if ``True``, append data reversed

        .. versionadded:: 1.2.1
        """
        self._log(u"Appending audio data...")
        new_data_length = len(new_data)
        current_length = self.__samples_length
        future_length = current_length + new_data_length
        if (self.__samples is None) or (self.__samples_capacity < future_length):
            self.preallocate_memory(2 * future_length)
        if reverse:
            self.__samples[current_length:future_length] = new_data[::-1]
        else:
            self.__samples[current_length:future_length] = new_data[:]
        self.__samples_length = future_length
        self._update_length()
        self._log(u"Appending audio data... done")

    def prepend(self, new_data):
        """
        Prepend the given new data to the current audio data.

        If no audio data is present, then at the end
        the audio data will be equal to ``new_data``.

        :param new_data: the new data to be prepended
        :type  new_data: numpy 1D array

        .. versionadded:: 1.2.1
        """
        self._log(u"Prepending audio data...")
        new_data_length = len(new_data)
        current_length = self.__samples_length
        future_length = current_length + new_data_length
        if (self.__samples is None) or (self.__samples_capacity < future_length):
            self.preallocate_memory(2 * future_length)
        self.__samples[new_data_length:future_length] = self.__samples[0:current_length]
        self.__samples[0:new_data_length] = new_data[:]
        self.__samples_length = future_length
        self._update_length()
        self._log(u"Prepending audio data... done")

    def reverse(self):
        """
        Reverse the audio data.

        .. versionadded:: 1.2.0
        """
        if self.__samples is None:
            raise AudioFileMonoWAVENotInitialized(u"The AudioFileMonoWAVE is not initialized")
        self._log(u"Reversing...")
        self.__samples[0:self.__samples_length] = numpy.flipud(self.__samples[0:self.__samples_length])
        self._log(u"Reversing... done")

    def trim(self, begin=None, length=None):
        """
        Get a slice of the audio data of ``length`` seconds,
        starting from ``begin`` seconds.

        If audio data is not loaded, load it and then slice it.

        :param float begin: the start position, in seconds
        :param float length: the  position, in seconds

        .. versionadded:: 1.2.0
        """
        self._log(u"Trimming...")
        if (begin is None) and (length is None):
            self._log(u"begin and length are both None: nothing to do")
        else:
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
            new_idx = end_index - begin_index
            self.__samples[0:new_idx] = self.__samples[begin_index:end_index]
            self.__samples_length = new_idx
            self._update_length()
        self._log(u"Trimming... done")

    def write(self, file_path):
        """
        Write the audio data to file.
        Return ``True`` on success, or ``False`` otherwise.

        :param string file_path: the path of the output file to be written

        .. versionadded:: 1.2.0
        """
        if self.__samples is None:
            raise AudioFileMonoWAVENotInitialized("The AudioFileMonoWAVE is not initialized")
        self._log([u"Writing audio file '%s'...", file_path])
        try:
            # our value is a float64 in [-1, 1]
            # scipy writes the sample as an int16_t, that is, a number in [-32768, 32767]
            data = (self.audio_samples * 32768).astype("int16")
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
        self.__samples_capacity = 0
        self.__samples_length = 0
        self.__samples = None

    def _update_length(self):
        """
        Update the audio length property,
        according to the length of the current audio data
        and audio sample rate.

        This function fails silently if one of the two is None.
        """
        if (self.audio_sample_rate is not None) and (self.__samples is not None):
            self.audio_length = self.__samples_length / self.audio_sample_rate



