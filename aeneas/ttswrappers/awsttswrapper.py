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

* :class:`~aeneas.ttswrappers.awsttswrapper.AWSTTSWrapper`,
  a wrapper for the AWS Polly TTS API engine.

Please refer to
https://aws.amazon.com/polly/
for further details.

.. note:: This module requires Python module ``boto3`` (``pip install boto3``).

.. warning:: You will be billed according to your AWS account plan.
             Your AWS credentials and configuration settings
             to access the AWS Polly service must be
             either stored on disk
             (e.g., in ``~/.aws/credentials`` and ``~/.aws/config``)
             or set in environment variables.
             Please refer to
             http://boto3.readthedocs.io/en/latest/guide/configuration.html
             for further details.

.. warning:: This module is experimental, use at your own risk.

.. versionadded:: 1.7.0
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import time
import uuid

from aeneas.audiofile import AudioFile
from aeneas.exacttiming import TimeValue
from aeneas.language import Language
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.ttswrappers.basettswrapper import BaseTTSWrapper
import aeneas.globalfunctions as gf


class AWSTTSWrapper(BaseTTSWrapper):
    """
    A wrapper for the AWS Polly TTS API.

    This wrapper supports calling the TTS engine
    only via Python.

    In abstract terms, it performs one or more calls to the
    AWS Polly TTS API service, and concatenate the resulting WAVE files,
    returning their anchor times.

    To use this TTS engine, specify ::

        "tts=aws"

    in the ``RuntimeConfiguration`` object.

    Your AWS credentials and configuration settings
    to access the AWS Polly service must be
    either stored on disk
    (e.g., in ``~/.aws/credentials`` and ``~/.aws/config``)
    or set in environment variables.
    Please refer to
    http://boto3.readthedocs.io/en/latest/guide/configuration.html
    for further details.

    You might also want to enable the TTS caching,
    to reduce the number of API calls ::

        "tts=aws|tts_cache=True"

    See :class:`~aeneas.ttswrappers.basettswrapper.BaseTTSWrapper`
    for the available functions.
    Below are listed the languages supported by this wrapper.

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    CYM = Language.CYM
    """ Welsh """

    DAN = Language.DAN
    """ Danish """

    DEU = Language.DEU
    """ German """

    ENG = Language.ENG
    """ English """

    FRA = Language.FRA
    """ French """

    ISL = Language.ISL
    """ Icelandic """

    ITA = Language.ITA
    """ Italian """

    JPN = Language.JPN
    """ Japanese """

    NLD = Language.NLD
    """ Dutch """

    NOR = Language.NOR
    """ Norwegian """

    POL = Language.POL
    """ Polish """

    POR = Language.POR
    """ Portuguese """

    RON = Language.RON
    """ Romanian """

    RUS = Language.RUS
    """ Russian """

    SPA = Language.SPA
    """ Spanish """

    SWE = Language.SWE
    """ Swedish """

    TUR = Language.TUR
    """ Turkish """

    ENG_AUS = "eng-AUS"
    """ English (Australia) """

    ENG_GBR = "eng-GBR"
    """ English (GB) """

    ENG_IND = "eng-IND"
    """ English (India) """

    ENG_USA = "eng-USA"
    """ English (USA) """

    ENG_WLS = "eng-WLS"
    """ English (Wales) """

    FRA_CAN = "fra-CAN"
    """ French (Canada) """

    FRA_FRA = "fra-FRA"
    """ French (France) """

    POR_BRA = "por-BRA"
    """ Portuguese (Brazil) """

    POR_PRT = "por-PRT"
    """ Portuguese (Portugal) """

    SPA_ESP = "spa-ESP"
    """ Spanish (Spain) """

    SPA_USA = "spa-USA"
    """ Spanish (USA) """

    CODE_TO_HUMAN = {
        CYM: u"Welsh",
        DAN: u"Danish",
        DEU: u"German",
        ENG: u"English",
        FRA: u"French",
        ISL: u"Icelandic",
        ITA: u"Italian",
        JPN: u"Japanese",
        NLD: u"Dutch",
        NOR: u"Norwegian",
        POL: u"Polish",
        POR: u"Portuguese",
        RON: u"Romanian",
        RUS: u"Russian",
        SPA: u"Spanish",
        SWE: u"Swedish",
        TUR: u"Turkish",
        ENG_AUS: u"English (Australia)",
        ENG_GBR: u"English (GB)",
        ENG_IND: u"English (India)",
        ENG_USA: u"English (USA)",
        ENG_WLS: u"English (Wales)",
        FRA_CAN: u"French (Canada)",
        FRA_FRA: u"French (France)",
        POR_BRA: u"Portuguese (Brazil)",
        POR_PRT: u"Portuguese (Portugal)",
        SPA_ESP: u"Spanish (Spain)",
        SPA_USA: u"Spanish (USA)",
    }

    CODE_TO_HUMAN_LIST = sorted([u"%s\t%s" % (k, v) for k, v in CODE_TO_HUMAN.items()])

    LANGUAGE_TO_VOICE_CODE = {
        CYM: "Gwyneth",         # F
        DAN: "Naja",            # F, M: Mads
        DEU: "Marlene",         # F, M: Hans
        ENG: "Joanna",          # F
        FRA: "Celine",          # F, M: Mathieu
        ISL: "Dora",            # F, M: Karl
        ITA: "Carla",           # F, M: Giorgio
        JPN: "Mizuki",          # F
        NLD: "Lotte",           # F, M: Ruben
        NOR: "Liv",             # F
        POL: "Maja",            # F, F: Ewa, M: Jan, Jacek
        POR: "Ines",            # F
        RON: "Carmen",          # F
        RUS: "Tatyana",         # F, M: Maxim
        SPA: "Conchita",        # F, M: Enrique
        SWE: "Astrid",          # F
        TUR: "Filiz",           # F
        ENG_AUS: "Nicole",      # F, M: Russell
        ENG_GBR: "Emma",        # F, F: Amy, M: Brian
        ENG_IND: "Raveena",     # F
        ENG_USA: "Joanna",      # F, F: Salli, Kimberly, Kendra, Ivy, M: Justin, Joey
        ENG_WLS: "Geraint",     # M
        FRA_FRA: "Celine",      # F, M: Mathieu
        FRA_CAN: "Chantal",     # F
        POR_BRA: "Vitoria",     # F, M: Ricardo
        POR_PRT: "Ines",        # F, M: Cristiano
        SPA_ESP: "Conchita",    # F, M: Enrique
        SPA_USA: "Penelope",    # F, M: Miguel
    }
    DEFAULT_LANGUAGE = ENG_USA

    OUTPUT_AUDIO_FORMAT = ("pcm_s16le", 1, 16000)

    HAS_PYTHON_CALL = True

    SAMPLE_FORMAT = "pcm"
    """ Synthesize 16kHz PCM16 mono """

    SAMPLE_RATE = 16000
    """ Synthesize 16kHz PCM16 mono """

    TAG = u"AWSTTSWrapper"

    def __init__(self, rconf=None, logger=None):
        super(AWSTTSWrapper, self).__init__(rconf=rconf, logger=logger)

    def _synthesize_single_python_helper(self, text, voice_code, output_file_path=None, return_audio_data=True):
        self.log(u"Importing boto3...")
        import boto3
        self.log(u"Importing boto3... done")

        # prepare client
        polly_client = boto3.client("polly")

        # post request
        sleep_delay = self.rconf[RuntimeConfiguration.TTS_API_SLEEP]
        attempts = self.rconf[RuntimeConfiguration.TTS_API_RETRY_ATTEMPTS]
        self.log([u"Sleep delay:    %.3f", sleep_delay])
        self.log([u"Retry attempts: %d", attempts])

        while attempts > 0:
            self.log(u"Sleeping to throttle API usage...")
            time.sleep(sleep_delay)
            self.log(u"Sleeping to throttle API usage... done")
            self.log(u"Posting...")
            try:
                response = polly_client.synthesize_speech(
                    Text=text,
                    OutputFormat=self.SAMPLE_FORMAT,
                    SampleRate="%d" % self.SAMPLE_RATE,
                    VoiceId=voice_code
                )
            except Exception as exc:
                self.log_exc(u"Unexpected exception on HTTP POST. Are you offline?", exc, True, ValueError)
            self.log(u"Posting... done")
            self.log(u"Reading response...")
            try:
                status_code = response["ResponseMetadata"]["HTTPStatusCode"]
                response_content = response["AudioStream"].read()
            except Exception as exc:
                self.log_warn(u"Error while reading the response status code or the response content")
                status_code = 999
            self.log(u"Reading response... done")
            self.log([u"Status code: %d", status_code])
            if status_code == 200:
                self.log(u"Got status code 200, break")
                break
            else:
                self.log_warn(u"Got status code other than 200, retry")
                attempts -= 1

        if attempts < 0:
            self.log_exc(u"All API requests returned status code != 200", None, True, ValueError)

        # save to file if requested
        if output_file_path is None:
            self.log(u"output_file_path is None => not saving to file")
        else:
            self.log(u"output_file_path is not None => saving to file...")
            import wave
            output_file = wave.open(output_file_path, "wb")
            output_file.setframerate(self.SAMPLE_RATE)  # sample rate
            output_file.setnchannels(1)                 # 1 channel, i.e. mono
            output_file.setsampwidth(2)                 # 16 bit/sample, i.e. 2 bytes/sample
            output_file.writeframes(response_content)
            output_file.close()
            self.log(u"output_file_path is not None => saving to file... done")

        # get length and data
        audio_sample_rate = self.SAMPLE_RATE
        number_of_frames = len(response_content) / 2
        audio_length = TimeValue(number_of_frames / audio_sample_rate)
        self.log([u"Response (bytes): %d", len(response_content)])
        self.log([u"Number of frames: %d", number_of_frames])
        self.log([u"Audio length (s): %.3f", audio_length])
        audio_format = "pcm16"
        audio_samples = numpy.fromstring(response_content, dtype=numpy.int16).astype("float64") / 32768

        # return data
        return (True, (audio_length, audio_sample_rate, audio_format, audio_samples))
