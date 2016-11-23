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
A wrapper for the ``speect`` TTS engine.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import speect
import speect.audio
import speect.audio_riff

from aeneas.audiofile import AudioFile
from aeneas.language import Language
from aeneas.exacttiming import TimeValue
from aeneas.ttswrappers.basettswrapper import BaseTTSWrapper
import aeneas.globalfunctions as gf


class CustomTTSWrapper(BaseTTSWrapper):
    """
    A wrapper for the ``speect`` TTS engine.

    This wrapper supports calling the TTS engine
    only via Python.

    To use this TTS engine, specify ::

        "tts=custom|tts_path=/path/to/this/file.py"

    in the ``RuntimeConfiguration`` object.

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    #
    # NOTE in this example we load an English voice,
    #      hence we support only English language,
    #      and we map it to a dummy voice code
    #
    ENG = Language.ENG
    """ English """
    LANGUAGE_TO_VOICE_CODE = {
        ENG: ENG
    }
    DEFAULT_LANGUAGE = ENG

    #
    # NOTE in this example we load a voice producing
    #      audio data in PCM16 mono WAVE (RIFF) format at 16000 Hz
    #
    OUTPUT_AUDIO_FORMAT = ("pcm_s16le", 1, 16000)

    #
    # NOTE in this example we call speect through
    #      Python, via the speect Python package
    #
    HAS_PYTHON_CALL = True

    TAG = u"CustomTTSWrapperSPEECT"

    def __init__(self, rconf=None, logger=None):
        super(CustomTTSWrapper, self).__init__(rconf=rconf, logger=logger)

    def _synthesize_single_python_helper(
        self,
        text,
        voice_code,
        output_file_path=None,
        return_audio_data=True
    ):
        """
        This is an helper function to synthesize a single text fragment via a Python call.

        If ``output_file_path`` is ``None``,
        the audio data will not persist to file at the end of the method.

        :rtype: tuple (result, (duration, sample_rate, encoding, data))
        """
        # return zero if text is the empty string
        if len(text) == 0:
            #
            # NOTE values of sample_rate, encoding, data
            #      do not matter if the duration is 0.000,
            #      so set them to None instead of the more precise:
            #      return (True, (TimeValue("0.000"), 16000, "pcm_s16le", numpy.array([])))
            #
            self.log(u"len(text) is zero: returning 0.000")
            return (True, (TimeValue("0.000"), None, None, None))

        #
        # NOTE in this example, we assume that the Speect voice data files
        #      are located in the same directory of this .py source file
        #      and that the voice JSON file is called "voice.json"
        #
        # NOTE the voice_code value is ignored in this example,
        #      since we have only one TTS voice,
        #      but in general one might select a voice file to load,
        #      depending on voice_code;
        #      in fact, we could have created the ``voice`` object
        #      only once, in the constructor, instead of creating it
        #      each time this function is invoked,
        #      achieving slightly faster synthesis
        #
        voice_json_path = gf.safe_str(gf.absolute_path("voice.json", __file__))
        voice = speect.SVoice(voice_json_path)
        utt = voice.synth(text)
        audio = utt.features["audio"]
        if output_file_path is None:
            self.log(u"output_file_path is None => not saving to file")
        else:
            self.log(u"output_file_path is not None => saving to file...")
            # NOTE apparently, save_riff needs the path to be a byte string
            audio.save_riff(gf.safe_str(output_file_path))
            self.log(u"output_file_path is not None => saving to file... done")

        # return immediately if returning audio data is not needed
        if not return_audio_data:
            self.log(u"return_audio_data is True => return immediately")
            return (True, None)

        # get length and data using speect Python API
        self.log(u"return_audio_data is True => read and return audio data")
        waveform = audio.get_audio_waveform()
        audio_sample_rate = int(waveform["samplerate"])
        audio_length = TimeValue(audio.num_samples() / audio_sample_rate)
        audio_format = "pcm16"
        audio_samples = numpy.fromstring(
            waveform["samples"],
            dtype=numpy.int16
        ).astype("float64") / 32768
        return (True, (
            audio_length,
            audio_sample_rate,
            audio_format,
            audio_samples
        ))
