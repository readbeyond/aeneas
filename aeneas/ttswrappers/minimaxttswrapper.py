#!/usr/bin/env python
# coding=utf-8

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2017, Alberto Pettarin (www.albertopettarin.it)
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
Synthesize text with the MiniMax synchronous speech API.

Set ``MINIMAX_API_KEY`` in the environment, then use::

    tts=minimax|tts_voice_code=YOUR_VOICE_ID|tts_cache=True

``minimax_tts_model`` selects the speech model and ``minimax_tts_region``
selects ``global_en`` (default) or ``cn_zh``. API usage is billed to your
MiniMax account. Cached fragments avoid repeated synthesis requests.

See https://platform.minimax.io/docs/api-reference/speech-t2a-http.
"""

from __future__ import absolute_import
from __future__ import division
import binascii
import io
import json
import os
import time
import wave

import numpy

try:
    from urllib.request import Request, urlopen
except ImportError:
    from urllib2 import Request, urlopen

from aeneas.exacttiming import TimeValue
from aeneas.language import Language
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.ttswrappers.basettswrapper import BaseTTSWrapper


class MiniMaxTTSWrapper(BaseTTSWrapper):
    """A Python wrapper returning mono PCM16 audio and fragment timings."""

    ENDPOINTS = {
        'global_en': 'https://api.minimax.io/v1/t2a_v2',
        'cn_zh': 'https://api.minimaxi.com/v1/t2a_v2',
    }
    SAMPLE_RATE = 32000
    OUTPUT_AUDIO_FORMAT = ("pcm_s16le", 1, SAMPLE_RATE)
    HAS_PYTHON_CALL = True
    DEFAULT_LANGUAGE = Language.ENG
    LANGUAGE_TO_VOICE_CODE = dict.fromkeys([
        Language.AFR, Language.ARA, Language.BUL, Language.CAT,
        Language.CES, Language.CMN, Language.DAN, Language.DEU,
        Language.ELL, Language.ENG, Language.FIN, Language.FRA,
        Language.HEB, Language.HIN, Language.HRV, Language.HUN,
        Language.IND, Language.ITA, Language.JPN, Language.KOR,
        Language.MSA, Language.NLD, Language.NOR, Language.POL,
        Language.POR, Language.RON, Language.RUS, Language.SLK,
        Language.SLV, Language.SPA, Language.SWE, Language.THA,
        Language.TUR, Language.UKR, Language.VIE,
    ])
    TAG = u"MiniMaxTTSWrapper"

    def __init__(self, rconf=None, logger=None):
        super(MiniMaxTTSWrapper, self).__init__(rconf=rconf, logger=logger)
        region = self.rconf[RuntimeConfiguration.MINIMAX_TTS_REGION]
        if region not in self.ENDPOINTS:
            raise ValueError("minimax_tts_region must be global_en or cn_zh")
        if not self.rconf[RuntimeConfiguration.TTS_VOICE_CODE]:
            raise ValueError("MiniMax requires a tts_voice_code voice ID")
        self.endpoint = self.ENDPOINTS[region]

    def _synthesize_single_python_helper(self, text, voice_code, output_file_path=None, return_audio_data=True):
        if not text:
            return (True, (TimeValue("0"), self.SAMPLE_RATE, "pcm16", numpy.zeros(0)))
        if len(text) >= 10000:
            raise ValueError("MiniMax text fragments must be shorter than 10000 characters")
        api_key = os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError("Set MINIMAX_API_KEY before synthesizing text")
        payload = {
            "model": self.rconf[RuntimeConfiguration.MINIMAX_TTS_MODEL],
            "text": text,
            "stream": False,
            "output_format": "hex",
            "voice_setting": {"voice_id": voice_code},
            "audio_setting": {
                "format": "wav", "sample_rate": self.SAMPLE_RATE, "channel": 1,
            },
        }
        request = Request(self.endpoint, json.dumps(payload).encode("utf-8"), {
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        })
        time.sleep(float(self.rconf[RuntimeConfiguration.TTS_API_SLEEP]))
        try:
            response = urlopen(request, timeout=60)
            try:
                result = json.loads(response.read().decode("utf-8"))
            finally:
                response.close()
        except Exception:
            # Do not echo response bodies, request headers, or credentials.
            raise ValueError("MiniMax speech request failed")
        if not isinstance(result, dict):
            raise ValueError("Invalid MiniMax speech response")
        base = result.get("base_resp") or {}
        data = result.get("data") or {}
        if not isinstance(base, dict) or base.get("status_code") != 0:
            raise ValueError("MiniMax speech API reported an error")
        if not isinstance(data, dict) or data.get("status") != 2:
            raise ValueError("MiniMax speech audio is incomplete")
        try:
            audio = binascii.unhexlify(data["audio"])
        except (KeyError, TypeError, ValueError, binascii.Error):
            raise ValueError("MiniMax returned invalid hex audio")
        decoded = self._decode_audio(audio)
        if output_file_path is not None:
            with open(output_file_path, "wb") as output_file:
                output_file.write(audio)
        return (True, decoded if return_audio_data else None)

    def _read_audio_data(self, file_path):
        # Keep cached samples at the synthesis rate, independently of the
        # analysis sample rate used elsewhere in the runtime configuration.
        with open(file_path, "rb") as audio_file:
            return (True, self._decode_audio(audio_file.read()))

    def _decode_audio(self, audio):
        try:
            reader = wave.open(io.BytesIO(audio), "rb")
            try:
                if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getcomptype()) != (1, 2, self.SAMPLE_RATE, "NONE"):
                    raise ValueError("Unexpected audio format")
                frame_count = reader.getnframes()
                pcm = reader.readframes(frame_count)
                if frame_count == 0 or len(pcm) != frame_count * 2:
                    raise ValueError("Empty or truncated audio")
            finally:
                reader.close()
        except Exception:
            raise ValueError("MiniMax did not return valid mono PCM16 WAVE audio")
        samples = numpy.frombuffer(pcm, dtype="<i2").astype("float64") / 32768
        duration = TimeValue(frame_count) / TimeValue(self.SAMPLE_RATE)
        return (duration, self.SAMPLE_RATE, "pcm16", samples)
