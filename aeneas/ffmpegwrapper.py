#!/usr/bin/env python
# coding=utf-8

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2016, Alberto Pettarin (www.albertopettarin.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
This module contains the following classes:

* :class:`~aeneas.ffmpegwrapper.FFMPEGWrapper`, a wrapper around ``ffmpeg`` to convert audio files;
* :class:`~aeneas.ffmpegwrapper.FFMPEGPathError`, representing a failure to locate the ``ffmpeg`` executable.
"""

from __future__ import absolute_import
from __future__ import print_function
import subprocess

from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration
import aeneas.globalfunctions as gf


class FFMPEGPathError(Exception):
    """
    Error raised when the path to ``ffmpeg`` is not a valid executable.

    .. versionadded:: 1.4.1
    """
    pass


class FFMPEGWrapper(Loggable):
    """
    A wrapper around ``ffmpeg`` to convert audio files.

    In abstract terms, it will perform a call like::

        $ ffmpeg -i /path/to/input.mp3 [parameters] /path/to/output.wav

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    FFMPEG_SAMPLE_8000 = ["-ar", "8000"]
    """ Single parameter for ``ffmpeg``: 8000 Hz sampling """

    FFMPEG_SAMPLE_16000 = ["-ar", "16000"]
    """ Single parameter for ``ffmpeg``: 16000 Hz sampling """

    FFMPEG_SAMPLE_22050 = ["-ar", "22050"]
    """ Single parameter for ``ffmpeg``: 22050 Hz sampling """

    FFMPEG_SAMPLE_44100 = ["-ar", "44100"]
    """ Single parameter for ``ffmpeg``: 44100 Hz sampling """

    FFMPEG_SAMPLE_48000 = ["-ar", "48000"]
    """ Single parameter for ``ffmpeg``: 48000 Hz sampling """

    FFMPEG_MONO = ["-ac", "1"]
    """ Single parameter for ``ffmpeg``: mono (1 channel) """

    FFMPEG_STEREO = ["-ac", "2"]
    """ Single parameter for ``ffmpeg``: stereo (2 channels) """

    FFMPEG_OVERWRITE = ["-y"]
    """ Single parameter for ``ffmpeg``: force overwriting output file """

    FFMPEG_PLAIN_HEADER = ["-map_metadata", "-1", "-flags", "+bitexact"]
    """ Single parameter for ``ffmpeg``: generate WAVE header
    without extra chunks (e.g., the INFO chunk) """

    FFMPEG_FORMAT_WAVE = ["-f", "wav"]
    """ Single parameter for ``ffmpeg``: produce output in ``wav`` format
    (must be the second to last argument to ``ffmpeg``,
    just before path of the output file) """

    FFMPEG_PARAMETERS_SAMPLE_KEEP = (
        FFMPEG_MONO +
        FFMPEG_OVERWRITE +
        FFMPEG_PLAIN_HEADER +
        FFMPEG_FORMAT_WAVE
    )
    """ Set of parameters for ``ffmpeg`` without changing the sampling rate """

    FFMPEG_PARAMETERS_SAMPLE_8000 = (
        FFMPEG_MONO +
        FFMPEG_SAMPLE_8000 +
        FFMPEG_OVERWRITE +
        FFMPEG_PLAIN_HEADER +
        FFMPEG_FORMAT_WAVE
    )
    """ Set of parameters for ``ffmpeg`` with 8000 Hz sampling """

    FFMPEG_PARAMETERS_SAMPLE_16000 = (
        FFMPEG_MONO +
        FFMPEG_SAMPLE_16000 +
        FFMPEG_OVERWRITE +
        FFMPEG_PLAIN_HEADER +
        FFMPEG_FORMAT_WAVE
    )
    """ Set of parameters for ``ffmpeg`` with 16000 Hz sampling """

    FFMPEG_PARAMETERS_SAMPLE_22050 = (
        FFMPEG_MONO +
        FFMPEG_SAMPLE_22050 +
        FFMPEG_OVERWRITE +
        FFMPEG_PLAIN_HEADER +
        FFMPEG_FORMAT_WAVE
    )
    """ Set of parameters for ``ffmpeg`` with 22050 Hz sampling """

    FFMPEG_PARAMETERS_SAMPLE_44100 = (
        FFMPEG_MONO +
        FFMPEG_SAMPLE_44100 +
        FFMPEG_OVERWRITE +
        FFMPEG_PLAIN_HEADER +
        FFMPEG_FORMAT_WAVE
    )
    """ Set of parameters for ``ffmpeg`` with 44100 Hz sampling """

    FFMPEG_PARAMETERS_SAMPLE_48000 = (
        FFMPEG_MONO +
        FFMPEG_SAMPLE_48000 +
        FFMPEG_OVERWRITE +
        FFMPEG_PLAIN_HEADER +
        FFMPEG_FORMAT_WAVE
    )
    """ Set of parameters for ``ffmpeg`` with 48000 Hz sampling """

    FFMPEG_PARAMETERS_MAP = {
        8000: FFMPEG_PARAMETERS_SAMPLE_8000,
        16000: FFMPEG_PARAMETERS_SAMPLE_16000,
        22050: FFMPEG_PARAMETERS_SAMPLE_22050,
        44100: FFMPEG_PARAMETERS_SAMPLE_44100,
        48000: FFMPEG_PARAMETERS_SAMPLE_48000
    }
    """ Map sample rate to parameter list """

    FFMPEG_PARAMETERS_DEFAULT = FFMPEG_PARAMETERS_SAMPLE_16000
    """ Default set of parameters for ``ffmpeg`` """

    TAG = u"FFMPEGWrapper"

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

        :param string input_file_path: the path of the audio file to convert
        :param string output_file_path: the path of the converted audio file
        :param float head_length: skip these many seconds
                                  from the beginning of the audio file
        :param float process_length: process these many seconds of the audio file
        :raises: :class:`~aeneas.ffmpegwrapper.FFMPEGPathError`: if the path to the ``ffmpeg`` executable cannot be called
        :raises: OSError: if ``input_file_path`` does not exist
                          or ``output_file_path`` cannot be written
        """
        # test if we can read the input file
        if not gf.file_can_be_read(input_file_path):
            self.log_exc(u"Input file '%s' cannot be read" % (input_file_path), None, True, OSError)

        # test if we can write the output file
        if not gf.file_can_be_written(output_file_path):
            self.log_exc(u"Output file '%s' cannot be written" % (output_file_path), None, True, OSError)

        # call ffmpeg
        arguments = [self.rconf[RuntimeConfiguration.FFMPEG_PATH]]
        arguments.extend(["-i", input_file_path])
        if head_length is not None:
            arguments.extend(["-ss", head_length])
        if process_length is not None:
            arguments.extend(["-t", process_length])
        if self.rconf.sample_rate in self.FFMPEG_PARAMETERS_MAP:
            arguments.extend(self.FFMPEG_PARAMETERS_MAP[self.rconf.sample_rate])
        else:
            arguments.extend(self.FFMPEG_PARAMETERS_DEFAULT)
        arguments.append(output_file_path)
        self.log([u"Calling with arguments '%s'", arguments])
        try:
            proc = subprocess.Popen(
                arguments,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            proc.communicate()
            proc.stdout.close()
            proc.stdin.close()
            proc.stderr.close()
        except OSError as exc:
            self.log_exc(u"Unable to call the '%s' ffmpeg executable" % (self.rconf[RuntimeConfiguration.FFMPEG_PATH]), exc, True, FFMPEGPathError)
        self.log(u"Call completed")

        # check if the output file exists
        if not gf.file_exists(output_file_path):
            self.log_exc(u"Output file '%s' was not written" % (output_file_path), None, True, OSError)

        # returning the output file path
        self.log([u"Returning output file path '%s'", output_file_path])
        return output_file_path
