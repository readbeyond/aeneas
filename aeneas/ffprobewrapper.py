#!/usr/bin/env python
# coding=utf-8

"""
This module contains the following classes:

* :class:`~aeneas.ffprobewrapper.FFPROBEWrapper`, a wrapper around ``ffprobe`` to read the properties of an audio file;
* :class:`~aeneas.ffprobewrapper.FFPROBEParsingError`,
* :class:`~aeneas.ffprobewrapper.FFPROBEPathError`, and
* :class:`~aeneas.ffprobewrapper.FFPROBEUnsupportedFormatError`,
  representing errors while reading the properties of audio files.
"""

from __future__ import absolute_import
from __future__ import print_function
import re
import subprocess

from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.timevalue import TimeValue
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

class FFPROBEParsingError(Exception):
    """
    Error raised when the call to ``ffprobe`` does not produce any output.
    """
    pass



class FFPROBEPathError(Exception):
    """
    Error raised when the path to ``ffprobe`` is not a valid executable.

    .. versionadded:: 1.4.1
    """
    pass


class FFPROBEUnsupportedFormatError(Exception):
    """
    Error raised when ``ffprobe`` cannot decode the format of the given file.
    """
    pass



class FFPROBEWrapper(Loggable):
    """
    Wrapper around ``ffprobe`` to read the properties of an audio file.

    It will perform a call like::

        $ ffprobe -select_streams a -show_streams /path/to/audio/file.mp3

    and it will parse the first ``[STREAM]`` element returned::

            [STREAM]
            index=0
            codec_name=mp3
            codec_long_name=MP3 (MPEG audio layer 3)
            profile=unknown
            codec_type=audio
            codec_time_base=1/44100
            codec_tag_string=[0][0][0][0]
            codec_tag=0x0000
            sample_fmt=s16p
            sample_rate=44100
            channels=1
            channel_layout=mono
            bits_per_sample=0
            id=N/A
            r_frame_rate=0/0
            avg_frame_rate=0/0
            time_base=1/14112000
            start_pts=0
            start_time=0.000000
            duration_ts=1545083190
            duration=109.487188
            bit_rate=128000
            max_bit_rate=N/A
            bits_per_raw_sample=N/A
            nb_frames=N/A
            nb_read_frames=N/A
            nb_read_packets=N/A
            DISPOSITION:default=0
            DISPOSITION:dub=0
            DISPOSITION:original=0
            DISPOSITION:comment=0
            DISPOSITION:lyrics=0
            DISPOSITION:karaoke=0
            DISPOSITION:forced=0
            DISPOSITION:hearing_impaired=0
            DISPOSITION:visual_impaired=0
            DISPOSITION:clean_effects=0
            DISPOSITION:attached_pic=0
            [/STREAM]

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    FFPROBE_PARAMETERS = [
        "-select_streams",
        "a",
        "-show_streams"
    ]
    """ ``ffprobe`` parameters """

    STDERR_DURATION_REGEX = re.compile(r"Duration: ([0-9]*):([0-9]*):([0-9]*)\.([0-9]*)")
    """ Regex to match ``ffprobe`` stderr duration values """

    STDOUT_BEGIN_STREAM = "[STREAM]"
    """ ``ffprobe`` stdout begin stream tag """

    STDOUT_CHANNELS = "channels"
    """ ``ffprobe`` stdout channels keyword """

    STDOUT_CODEC_NAME = "codec_name"
    """ ``ffprobe`` stdout codec name (format) keyword """

    STDOUT_END_STREAM = "[/STREAM]"
    """ ``ffprobe`` stdout end stream tag """

    STDOUT_DURATION = "duration"
    """ ``ffprobe`` stdout duration keyword """

    STDOUT_SAMPLE_RATE = "sample_rate"
    """ ``ffprobe`` stdout sample rate keyword """

    TAG = u"FFPROBEWrapper"

    def read_properties(self, audio_file_path):
        """
        Read the properties of an audio file
        and return them as a dictionary.

        Example: ::

            d["index"]=0
            d["codec_name"]=mp3
            d["codec_long_name"]=MP3 (MPEG audio layer 3)
            d["profile"]=unknown
            d["codec_type"]=audio
            d["codec_time_base"]=1/44100
            d["codec_tag_string"]=[0][0][0][0]
            d["codec_tag"]=0x0000
            d["sample_fmt"]=s16p
            d["sample_rate"]=44100
            d["channels"]=1
            d["channel_layout"]=mono
            d["bits_per_sample"]=0
            d["id"]=N/A
            d["r_frame_rate"]=0/0
            d["avg_frame_rate"]=0/0
            d["time_base"]=1/14112000
            d["start_pts"]=0
            d["start_time"]=0.000000
            d["duration_ts"]=1545083190
            d["duration"]=109.487188
            d["bit_rate"]=128000
            d["max_bit_rate"]=N/A
            d["bits_per_raw_sample"]=N/A
            d["nb_frames"]=N/A
            d["nb_read_frames"]=N/A
            d["nb_read_packets"]=N/A
            d["DISPOSITION:default"]=0
            d["DISPOSITION:dub"]=0
            d["DISPOSITION:original"]=0
            d["DISPOSITION:comment"]=0
            d["DISPOSITION:lyrics"]=0
            d["DISPOSITION:karaoke"]=0
            d["DISPOSITION:forced"]=0
            d["DISPOSITION:hearing_impaired"]=0
            d["DISPOSITION:visual_impaired"]=0
            d["DISPOSITION:clean_effects"]=0
            d["DISPOSITION:attached_pic"]=0

        :param string audio_file_path: the path of the audio file to analyze
        :rtype: dict
        :raises: TypeError: if ``audio_file_path`` is None
        :raises: OSError: if the file at ``audio_file_path`` cannot be read
        :raises: FFPROBEParsingError: if the call to ``ffprobe`` does not produce any output
        :raises: FFPROBEPathError: if the path to the ``ffprobe`` executable cannot be called
        :raises: FFPROBEUnsupportedFormatError: if the file has a format not supported by ``ffprobe``
        """

        # test if we can read the file at audio_file_path
        if audio_file_path is None:
            self.log_exc(u"The audio file path is None", None, True, TypeError)
        if not gf.file_can_be_read(audio_file_path):
            self.log_exc(u"Input file '%s' cannot be read" % (audio_file_path), None, True, OSError)

        # call ffprobe
        arguments = [self.rconf[RuntimeConfiguration.FFPROBE_PATH]]
        arguments.extend(self.FFPROBE_PARAMETERS)
        arguments.append(audio_file_path)
        self.log([u"Calling with arguments '%s'", arguments])
        try:
            proc = subprocess.Popen(
                arguments,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            (stdoutdata, stderrdata) = proc.communicate()
            proc.stdout.close()
            proc.stdin.close()
            proc.stderr.close()
        except OSError as exc:
            self.log_exc(u"Unable to call the '%s' ffprobe executable" % (self.rconf[RuntimeConfiguration.FFPROBE_PATH]), exc, True, FFPROBEPathError)
        self.log(u"Call completed")

        # check there is some output
        if (stdoutdata is None) or (len(stderrdata) == 0):
            self.log_exc(u"ffprobe produced no output", None, True, FFPROBEParsingError)

        # decode stdoutdata and stderrdata to Unicode string
        try:
            stdoutdata = gf.safe_unicode(stdoutdata)
            stderrdata = gf.safe_unicode(stderrdata)
        except UnicodeDecodeError as exc:
            self.log_exc(u"Unable to decode ffprobe out/err", exc, True, FFPROBEParsingError)

        # dictionary for the results
        results = {
            self.STDOUT_CHANNELS : None,
            self.STDOUT_CODEC_NAME : None,
            self.STDOUT_DURATION : None,
            self.STDOUT_SAMPLE_RATE : None
        }

        # scan the first audio stream the ffprobe stdout output
        # TODO more robust parsing
        # TODO deal with multiple audio streams
        for line in stdoutdata.splitlines():
            if line == self.STDOUT_END_STREAM:
                self.log(u"Reached end of the stream")
                break
            elif len(line.split("=")) == 2:
                key, value = line.split("=")
                results[key] = value
                self.log([u"Found property '%s'='%s'", key, value])

        try:
            self.log([u"Duration found in stdout: '%s'", results[self.STDOUT_DURATION]])
            results[self.STDOUT_DURATION] = TimeValue(results[self.STDOUT_DURATION])
            self.log(u"Valid duration")
        except:
            self.log_warn(u"Invalid duration")
            results[self.STDOUT_DURATION] = None
            # try scanning ffprobe stderr output
            for line in stderrdata.splitlines():
                match = self.STDERR_DURATION_REGEX.search(line)
                if match is not None:
                    self.log([u"Found matching line '%s'", line])
                    results[self.STDOUT_DURATION] = gf.time_from_hhmmssmmm(line)
                    self.log([u"Extracted duration '%.3f'", results[self.STDOUT_DURATION]])
                    break

        if results[self.STDOUT_DURATION] is None:
            self.log_exc(u"No duration found in stdout or stderr. Unsupported audio file format?", None, True, FFPROBEUnsupportedFormatError)

        # return dictionary
        self.log(u"Returning dict")
        return results



