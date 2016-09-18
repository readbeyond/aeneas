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

* :class:`~aeneas.ttswrappers.nuancettswrapper.NuanceTTSWrapper`,
  a wrapper for the Nuance TTS API engine.

.. note:: This module requires Python module ``requests`` (``pip install requests``).

.. warning:: You will be billed according to your Nuance Developers account plan.

.. warning:: This module is experimental, use at your own risk.

.. versionadded:: 1.5.0
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import time
import uuid

from aeneas.audiofile import AudioFile
from aeneas.language import Language
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.timevalue import TimeValue
from aeneas.ttswrappers.basettswrapper import BaseTTSWrapper
import aeneas.globalfunctions as gf


class NuanceTTSWrapper(BaseTTSWrapper):
    """
    A wrapper for the Nuance Developers TTS API.

    This wrapper supports calling the TTS engine
    only via Python.

    In abstract terms, it performs one or more calls to the
    Nuance TTS API service, and concatenate the resulting WAVE files,
    returning their anchor times.

    To use this TTS engine, specify ::

        "tts=nuance|nuance_tts_api_id=...|nuance_tts_api_key=..."

    in the ``RuntimeConfiguration`` object,
    substituting your Nuance Developer API ID and Key.

    See :class:`~aeneas.ttswrappers.basettswrapper.BaseTTSWrapper`
    for the available functions.
    Below are listed the languages supported by this wrapper.

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    ARA = Language.ARA
    """ Arabic """

    CAT = Language.CAT
    """ Catalan """

    CES = Language.CES
    """ Czech """

    CMN = Language.CMN
    """ Mandarin Chinese """

    DAN = Language.DAN
    """ Danish """

    DEU = Language.DEU
    """ German """

    ELL = Language.ELL
    """ Greek (Modern) """

    ENG = Language.ENG
    """ English """

    EUS = Language.EUS
    """ Basque """

    FIN = Language.FIN
    """ Finnish """

    FRA = Language.FRA
    """ French """

    GLG = Language.GLG
    """ Galician """

    HEB = Language.HEB
    """ Hebrew """

    HIN = Language.HIN
    """ Hindi """

    HUN = Language.HUN
    """ Hungarian """

    IND = Language.IND
    """ Indonesian """

    ITA = Language.ITA
    """ Italian """

    JPN = Language.JPN
    """ Japanese """

    KOR = Language.KOR
    """ Korean """

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

    SLK = Language.SLK
    """ Slovak """

    SPA = Language.SPA
    """ Spanish """

    SWE = Language.SWE
    """ Swedish """

    THA = Language.THA
    """ Thai """

    TUR = Language.TUR
    """ Turkish """

    YUE = Language.YUE
    """ Yue Chinese """

    NLD_BEL = "nld-BEL"
    """ Dutch (Belgium) """

    FRA_CAN = "fra-CAN"
    """ French (Canada) """

    CMN_CHN = "cmn-CHN"
    """ Mandarin Chinese (China) """

    CMN_TWN = "cmn-TWN"
    """ Mandarin Chinese (Taiwan) """

    POR_BRA = "por-BRA"
    """ Portuguese (Brazil) """

    POR_PRT = "por-PRT"
    """ Portuguese (Portugal) """

    SPA_ESP = "spa-ESP"
    """ Spanish (Castillian) """

    SPA_COL = "spa-COL"
    """ Spanish (Colombia) """

    SPA_MEX = "spa-MEX"
    """ Spanish (Mexico) """

    ENG_AUS = "eng-AUS"
    """ English (Australia) """

    ENG_GBR = "eng-GBR"
    """ English (GB) """

    ENG_IND = "eng-IND"
    """ English (India) """

    ENG_IRL = "eng-IRL"
    """ English (Ireland) """

    ENG_SCT = "eng-SCT"
    """ English (Scotland) """

    ENG_ZAF = "eng-ZAF"
    """ English (South Africa) """

    ENG_USA = "eng-USA"
    """ English (USA) """

    LANGUAGE_TO_VOICE_CODE = {
        CMN_CHN: "Tian-Tian",
        CMN_TWN: "Mei-Jia",
        FRA_CAN: "Amelie",      # F: Chantal
        ENG_AUS: "Karen",
        ENG_GBR: "Kate",        # F: Serena
        ENG_IND: "Veena",
        ENG_IRL: "Moira",
        ENG_SCT: "Fiona",
        ENG_USA: "Ava",         # F: Allison, F: Samantha, F: Susan, F: Zoe
        ENG_ZAF: "Tessa",
        NLD_BEL: "Ellen",
        POR_BRA: "Luciana",
        POR_PRT: "Catarina",    # F: Joana
        SPA_COL: "Soledad",
        SPA_ESP: "Monica",
        SPA_MEX: "Angelica",    # F: Paulina, F: Empar (Valencian)
        ARA: "Laila",
        CAT: "Jordi",           # M, F: Montserrat
        CES: "Iveta",           # F: Zuzana
        CMN: "Tian-Tian",
        DAN: "Ida",
        DEU: "Anna-ML",         # F-ML, F-ML: Petra-ML
        ELL: "Melina",
        ENG: "Kate",
        EUS: "Miren",
        FIN: "Satu",
        FRA: "Audrey-ML",       # F-ML, F: Aurelie
        GLG: "Carmela",
        HEB: "Carmit",
        HIN: "Lekha",
        HUN: "Mariska",
        IND: "Damayanti",
        ITA: "Alice-ML",        # F-ML, F: Federica, F: Paola
        JPN: "Kyoko",
        KOR: "Sora",
        NLD: "Claire",
        NOR: "Nora",
        POL: "Ewa",             # F: Zosia
        POR: "Catarina",
        RON: "Ioana",
        RUS: "Katya",           # F: Milena
        SLK: "Laura",
        SPA: "Monica",
        SWE: "Alva",
        THA: "Kanya",
        TUR: "Cem",             # M, F: Yelda
        YUE: "Sin-Ji",
    }
    DEFAULT_LANGUAGE = ENG_GBR

    OUTPUT_AUDIO_FORMAT = ("pcm_s16le", 1, 16000)

    HAS_PYTHON_CALL = True

    # Nuance TTS API specific
    END_POINT = "NMDPTTSCmdServlet/tts"
    """ Nuance TTS API end point """

    SAMPLE_RATE = 16000
    """ Synthesize 16kHz PCM16 mono """

    URL = "https://tts.nuancemobility.net"
    """ Nuance TTS API URL """

    TAG = u"NuanceTTSWrapper"

    def __init__(self, rconf=None, logger=None):
        super(NuanceTTSWrapper, self).__init__(rconf=rconf, logger=logger)

    def _synthesize_single_python_helper(self, text, voice_code, output_file_path=None):
        self.log(u"Importing requests...")
        import requests
        self.log(u"Importing requests... done")

        # prepare request header and contents
        request_id = str(uuid.uuid4()).replace("-", "")[0:16]
        headers = {
            u"Content-Type": u"text/plain; charset=utf-8",
            u"Accept": u"audio/x-wav;codec=pcm;bit=16;rate=%d" % self.SAMPLE_RATE
        }
        text_to_synth = text.encode("utf-8")
        url = "%s/%s?appId=%s&appKey=%s&id=%s&voice=%s" % (
            self.URL,
            self.END_POINT,
            self.rconf[RuntimeConfiguration.NUANCE_TTS_API_ID],
            self.rconf[RuntimeConfiguration.NUANCE_TTS_API_KEY],
            request_id,
            voice_code
        )

        # DEBUG
        # print(self.rconf[RuntimeConfiguration.NUANCE_TTS_API_ID])
        # print(self.rconf[RuntimeConfiguration.NUANCE_TTS_API_KEY])
        # #print(headers)
        # print(url)
        # print(voice_code)
        # raise ValueError("Stop execution here")

        # post request
        sleep_delay = self.rconf[RuntimeConfiguration.NUANCE_TTS_API_SLEEP]
        attempts = self.rconf[RuntimeConfiguration.NUANCE_TTS_API_RETRY_ATTEMPTS]
        self.log([u"Sleep delay:    %.3f", sleep_delay])
        self.log([u"Retry attempts: %d", attempts])
        while attempts > 0:
            self.log(u"Sleeping to throttle API usage...")
            time.sleep(sleep_delay)
            self.log(u"Sleeping to throttle API usage... done")
            self.log(u"Posting...")
            try:
                response = requests.post(url, data=text_to_synth, headers=headers)
            except Exception as exc:
                self.log_exc(u"Unexpected exception on HTTP POST. Are you offline?", exc, True, ValueError)
            self.log(u"Posting... done")
            status_code = response.status_code
            self.log([u"Status code: %d", status_code])
            if status_code == 200:
                self.log(u"Got status code 200, break")
                break
            else:
                self.log_warn(u"Got status code other than 200, retry")
                attempts -= 1

        if attempts < 0:
            self.log_exc(u"All HTTP POST requests returned status code != 200", None, True, ValueError)

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
            output_file.writeframes(response.content)
            output_file.close()
            self.log(u"output_file_path is not None => saving to file... done")

        # get length and data
        audio_sample_rate = self.SAMPLE_RATE
        number_of_frames = len(response.content) / 2
        audio_length = TimeValue(number_of_frames / audio_sample_rate)
        self.log([u"Response (bytes): %d", len(response.content)])
        self.log([u"Number of frames: %d", number_of_frames])
        self.log([u"Audio length (s): %.3f", audio_length])
        audio_format = "pcm16"
        audio_samples = numpy.fromstring(response.content, dtype=numpy.int16).astype("float64") / 32768

        # return data
        return (True, (audio_length, audio_sample_rate, audio_format, audio_samples))
