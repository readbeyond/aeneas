#!/usr/bin/env python
# coding=utf-8

"""
A class representing an audio file.
"""

#from scikits.audiolab import wavread
#from scikits.audiolab import wavwrite
import numpy
import os

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
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.3.2"
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

    The properties of the audio file
    (length, format, etc.)
    will be set by ``read_properties()``
    which will invoke an audio file probe.
    (Currently,
    :class:`aeneas.ffprobewrapper.FFPROBEWrapper`
    )

    :param file_path: the path to the audio file
    :type  file_path: string (path)
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = "AudioFile"

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

    def read_properties(self):
        """
        Populate this object by reading
        the audio properties of the file at the given path.

        Currently this function uses
        :class:`aeneas.ffprobewrapper.FFPROBEWrapper`
        to get the audio file properties.

        :raises AudioFileUnsupportedFormatError: if the audio file has a format not supported
        :raises IOError: if the audio file cannot be read
        """

        self._log("Reading properties...")

        # check the file can be read
        if not gf.file_exists(self.file_path):
            self._log(["File '%s' does not exist", self.file_path], Logger.CRITICAL)
            raise IOError("Input audio file '%s' does not exist" % self.file_path)

        # get the file size
        self._log(["Getting file size for '%s'", self.file_path])
        self.file_size = os.path.getsize(self.file_path)
        self._log(["File size for '%s' is '%d'", self.file_path, self.file_size])

        # get the audio properties
        try:
            self._log("Reading properties with FFPROBEWrapper...")
            prober = FFPROBEWrapper(logger=self.logger)
            properties = prober.read_properties(self.file_path)
            self._log("Reading properties with FFPROBEWrapper... done")
        except (FFPROBEUnsupportedFormatError, FFPROBEParsingError):
            self._log("Reading properties with FFPROBEWrapper... failed", Logger.CRITICAL)
            self._log("Unsupported audio file format", Logger.CRITICAL)
            raise AudioFileUnsupportedFormatError("Unsupported audio file format")

        # save relevant properties in results inside the audiofile object
        self.audio_length = gf.safe_float(properties[FFPROBEWrapper.STDOUT_DURATION])
        self._log(["Stored audio_length: '%s'", self.audio_length])
        self.audio_format = properties[FFPROBEWrapper.STDOUT_CODEC_NAME]
        self._log(["Stored audio_format: '%s'", self.audio_format])
        self.audio_sample_rate = gf.safe_int(properties[FFPROBEWrapper.STDOUT_SAMPLE_RATE])
        self._log(["Stored audio_sample_rate: '%s'", self.audio_sample_rate])
        self.audio_channels = gf.safe_int(properties[FFPROBEWrapper.STDOUT_CHANNELS])
        self._log(["Stored audio_channels: '%s'", self.audio_channels])
        self._log("Reading properties... done")



class AudioFileMonoWAV(AudioFile):
    """
    A monoaural WAV audio file.

    Its data can be read from and write to file,
    set from a `numpy` 1D array.

    It supports append, prepend, reverse, and trim
    operations.

    It can also extract MFCCs and store them internally,
    also after the audio data has been discarded.

    Note that, at the moment, the state of this object
    might be inconsistent (e.g., setting a new path
    after loading audio data will not flush the audio data).
    Use this class with care.

    :param file_path: the path to the audio file
    :type  file_path: string (path)
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = "AudioFileMonoWAV"

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

        :raises AudioFileUnsupportedFormatError: if the audio file is not a mono wav file
        :raises IOError: if the audio file cannot be read
        """
        self._log("Loading audio data...")

        # check the file can be read
        if not gf.file_exists(self.file_path):
            self._log(["File '%s' does not exist", self.file_path], Logger.CRITICAL)
            raise IOError("Input audio file '%s' does not exist" % self.file_path)

        try:
            # old way using scikits.audiolab
            #self.audio_data, self.audio_sample_rate, self.audio_format = wavread(self.file_path)
            # new way using wavfile from scipy.io
            self.audio_format = "pcm16"
            self.audio_sample_rate, self.audio_data = scipywavread(self.file_path)
            # scikits => [-1..1], scipy => [-32768..32767]
            self.audio_data = self.audio_data.astype("float64") / 32768
        except ValueError:
            self._log("Unsupported audio file format", Logger.CRITICAL)
            raise AudioFileUnsupportedFormatError("Unsupported audio file format")

        self._update_length()
        self._log(["Sample length: %f", self.audio_length])
        self._log(["Sample rate:   %f", self.audio_sample_rate])
        self._log(["Audio format:  %s", self.audio_format])
        self._log("Loading audio data... done")

    def append_data(self, new_data):
        """
        Append the given new_data to the current audio_data.

        :param new_data: the new data to be appended
        :type  new_data: numpy 1D array

        .. versionadded:: 1.2.1
        """
        self._log("Appending audio data...")
        self._ensure_audio_data(load=False)
        self.audio_data = numpy.append(self.audio_data, new_data)
        self._update_length()
        self._log("Appending audio data... done")

    def prepend_data(self, new_data):
        """
        Prepend the given new_data to the current audio_data.

        :param new_data: the new data to be prepended
        :type  new_data: numpy 1D array

        .. versionadded:: 1.2.1
        """
        self._log("Prepending audio data...")
        self._ensure_audio_data(load=False)
        self.audio_data = numpy.append(new_data, self.audio_data)
        self._update_length()
        self._log("Prepending audio data... done")

    def extract_mfcc(self, frame_rate=gc.MFCC_FRAME_RATE):
        """
        Extract MFCCs from the given audio file.

        If audio data is not loaded, load it, extract MFCCs,
        and then clear it.

        :param frame_rate: the MFCC frame rate, in frames per second. Default:
                           :class:`aeneas.globalconstants.MFCC_FRAME_RATE`
        :type  frame_rate: int
        """
        # remember if we have audio data already
        had_audio_data = self._ensure_audio_data(load=True)

        if gc.USE_C_EXTENSIONS:
            self._log("C extensions enabled in gc")
            if gf.can_run_c_extension("cmfcc"):
                self._log("C extensions enabled in gc and cmfcc can be loaded")
                try:
                    self._compute_mfcc_c_extension(frame_rate)
                    # if we did not have audio data, clear it immediately
                    if not had_audio_data:
                        self.clear_data()
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

        # if we did not have audio data, clear it immediately
        if not had_audio_data:
            self.clear_data()

    def reverse(self):
        """
        Reverse the audio data.

        If audio data is not loaded, load it and then reverse it.

        .. versionadded:: 1.2.0
        """
        self._log("Reversing...")
        self._ensure_audio_data(load=True)
        self.audio_data = self.audio_data[::-1]
        self._log("Reversing... done")

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
        self._log("Trimming...")
        if (begin is None) and (length is None):
            self._log("begin and length are both None: nothing to do")
        else:
            self._ensure_audio_data(load=True)
            self._log(["audio_length is %.3f", self.audio_length])
            if begin is None:
                begin = 0
                self._log(["begin was None, now set to %.3f", begin])
            begin = min(max(0, begin), self.audio_length)
            self._log(["begin is %.3f", begin])
            if length is None:
                length = self.audio_length - begin
                self._log(["length was None, now set to %.3f", length])
            length = min(max(0, length), self.audio_length - begin)
            self._log(["length is %.3f", length])
            begin_index = int(begin * self.audio_sample_rate)
            end_index = int((begin + length) * self.audio_sample_rate)
            self.audio_data = self.audio_data[begin_index:end_index]
            self._update_length()
        self._log("Trimming... done")

    def write(self, file_path):
        """
        Write the audio data to file.
        Return ``True`` on success, or ``False`` otherwise.

        :param file_path: the path of the output file to be written
        :type  file_path: string (path)

        .. versionadded:: 1.2.0
        """
        self._log(["Writing audio file '%s'...", file_path])
        self._ensure_audio_data(load=False)
        try:
            # old way using scikits.audiolab
            #wavwrite(self.audio_data, file_path, self.audio_sample_rate, self.audio_format)
            # new way using wavfile from scipy.io
            # scikits => [-1..1], scipy => [-32768..32767]
            data = (self.audio_data * 32768).astype("int16")
            scipywavwrite(file_path, self.audio_sample_rate, data)
        except:
            self._log("Error writing audio file", severity=Logger.CRITICAL)
            raise IOError("Error writing audio file to '%s'" % file_path) 
        self._log(["Writing audio file '%s'... done", file_path])

    def clear_data(self):
        """
        Clear the audio data, freeing memory.
        """
        self._log("Clear audio_data")
        self.audio_data = None

    def _update_length(self):
        """
        Update audio length
        """
        if (self.audio_sample_rate is not None) and (self.audio_data is not None):
            self.audio_length = (float(len(self.audio_data)) / self.audio_sample_rate)

    def _ensure_audio_data(self, load=True):
        """
        Check if audio_data is loaded:
        if so, return True.

        Otherwise, either load or initialize audio_data
        and return False.

        :param load: if True, load from file; if False, initialize to empty
        :type  load: bool
        :rtype: bool
        """
        if self.audio_data is not None:
            self._log("audio data is not None: returning True")
            return True
        if load:
            self._log("No audio data: loading it")
            self.load_data()
        else:
            self._log("No audio data: initializing it")
            self.audio_data = numpy.array([])
        self._log("audio data was None: returning False")
        return False

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



