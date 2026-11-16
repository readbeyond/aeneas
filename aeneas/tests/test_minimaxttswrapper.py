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

from __future__ import absolute_import
import binascii
import io
import json
import os
import unittest
import wave

import numpy

from aeneas.exacttiming import TimeValue
from aeneas.language import Language
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.synthesizer import Synthesizer
from aeneas.textfile import TextFile, TextFragment
from aeneas.ttswrappers.minimaxttswrapper import MiniMaxTTSWrapper
import aeneas.globalfunctions as gf
import aeneas.ttswrappers.minimaxttswrapper as minimax


class TestMiniMaxTTSWrapper(unittest.TestCase):

    def setUp(self):
        self.previous_key = os.environ.get("MINIMAX_API_KEY")
        os.environ["MINIMAX_API_KEY"] = "test-key"
        self.previous_urlopen = minimax.urlopen
        minimax.urlopen = self.urlopen
        self.rconf = RuntimeConfiguration("tts=minimax|tts_voice_code=test-voice|tts_api_sleep=0")
        self.calls = []
        self.closed = False
        self.body = {"base_resp": {"status_code": 0}, "data": {
            "status": 2, "audio": binascii.hexlify(self.wave_bytes()).decode("ascii"),
        }}

    def tearDown(self):
        minimax.urlopen = self.previous_urlopen
        if self.previous_key is None:
            os.environ.pop("MINIMAX_API_KEY", None)
        else:
            os.environ["MINIMAX_API_KEY"] = self.previous_key

    def wave_bytes(self, rate=32000, channels=1):
        buffer = io.BytesIO()
        writer = wave.open(buffer, "wb")
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(numpy.array([0, 16384, -32768, 32767], dtype="<i2").tobytes())
        writer.close()
        return buffer.getvalue()

    def urlopen(self, request, timeout):
        self.calls.append((request, timeout))
        return self

    def read(self):
        return json.dumps(self.body).encode("utf-8")

    def close(self):
        self.closed = True

    def synthesize(self, **kwargs):
        return MiniMaxTTSWrapper(self.rconf)._synthesize_single_python_helper(u"Hello", "test-voice", **kwargs)

    def test_registry_and_request(self):
        synth = Synthesizer(self.rconf)
        self.assertIsInstance(synth.tts_engine, MiniMaxTTSWrapper)
        self.assertIn("minimax", Synthesizer.ALLOWED_VALUES)
        computed, data = self.synthesize()
        self.assertTrue(computed)
        self.assertEqual(data[:3], (TimeValue("0.000125"), 32000, "pcm16"))
        numpy.testing.assert_allclose(data[3], [0, 0.5, -1, 32767 / 32768.0])
        request, timeout = self.calls[0]
        self.assertEqual(request.get_full_url(), MiniMaxTTSWrapper.ENDPOINTS["global_en"])
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(timeout, 60)
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], self.rconf[RuntimeConfiguration.MINIMAX_TTS_MODEL])
        self.assertEqual(payload["voice_setting"], {"voice_id": "test-voice"})
        self.assertEqual(payload["audio_setting"], {"format": "wav", "sample_rate": 32000, "channel": 1})
        self.assertEqual(payload["text"], "Hello")
        self.assertEqual(payload["output_format"], "hex")
        self.assertFalse(payload["stream"])
        self.assertTrue(self.closed)

    def test_region_and_model_override(self):
        self.rconf[RuntimeConfiguration.MINIMAX_TTS_REGION] = "cn_zh"
        self.rconf[RuntimeConfiguration.MINIMAX_TTS_MODEL] = "custom-speech-model"
        self.synthesize()
        request = self.calls[0][0]
        self.assertEqual(request.get_full_url(), MiniMaxTTSWrapper.ENDPOINTS["cn_zh"])
        self.assertEqual(json.loads(request.data.decode("utf-8"))["model"], "custom-speech-model")

    def test_invalid_configuration(self):
        self.rconf[RuntimeConfiguration.MINIMAX_TTS_REGION] = "invalid"
        with self.assertRaises(ValueError):
            MiniMaxTTSWrapper(self.rconf)
        self.rconf[RuntimeConfiguration.MINIMAX_TTS_REGION] = "global_en"
        self.rconf[RuntimeConfiguration.TTS_VOICE_CODE] = None
        with self.assertRaises(ValueError):
            MiniMaxTTSWrapper(self.rconf)
        self.assertEqual(self.calls, [])

    def test_missing_key(self):
        os.environ.pop("MINIMAX_API_KEY")
        with self.assertRaises(ValueError):
            self.synthesize()
        self.assertEqual(self.calls, [])

    def test_api_failure_and_incomplete_audio(self):
        for body in [None, {}, {"base_resp": {"status_code": 1004}, "data": None},
                     {"base_resp": {"status_code": 0}, "data": None},
                     {"base_resp": {"status_code": 0}, "data": {"status": 1}}]:
            self.body = body
            with self.assertRaises(ValueError):
                self.synthesize()

    def test_malformed_audio(self):
        for audio in [None, "", "not-hex", "1234", binascii.hexlify(self.wave_bytes(rate=16000)).decode("ascii"),
                      binascii.hexlify(self.wave_bytes(channels=2)).decode("ascii")]:
            self.body["data"]["audio"] = audio
            with self.assertRaises(ValueError):
                self.synthesize()

    def test_transport_failure(self):
        def fail(request, timeout):
            raise IOError("private transport details")
        minimax.urlopen = fail
        with self.assertRaises(ValueError) as error:
            self.synthesize()
        self.assertEqual(str(error.exception), "MiniMax speech request failed")

    def test_invalid_json_closes_response(self):
        self.read = lambda: b"invalid json"
        with self.assertRaises(ValueError):
            self.synthesize()
        self.assertTrue(self.closed)

    def test_save_without_returning_samples(self):
        handler, path = gf.tmp_file(suffix=".wav")
        try:
            self.assertEqual(self.synthesize(output_file_path=path, return_audio_data=False), (True, None))
            with open(path, "rb") as audio:
                self.assertEqual(audio.read(), self.wave_bytes())
        finally:
            gf.delete_file(handler, path)

    def test_empty_and_oversized_fragments(self):
        wrapper = MiniMaxTTSWrapper(self.rconf)
        self.assertEqual(wrapper._synthesize_single_python_helper(u"", "test-voice")[1][0], TimeValue("0"))
        with self.assertRaises(ValueError):
            wrapper._synthesize_single_python_helper(u"a" * 10000, "test-voice")
        self.assertEqual(self.calls, [])

    def test_multiple_fragments_and_cache(self):
        self.rconf[RuntimeConfiguration.TTS_CACHE] = True
        text = TextFile()
        text.add_fragment(TextFragment(identifier="f1", language=Language.ENG, lines=[u"Hello"], filtered_lines=[u"Hello"]))
        text.add_fragment(TextFragment(identifier="f2", language=Language.ENG, lines=[u"Hello"], filtered_lines=[u"Hello"]))
        synth = Synthesizer(self.rconf)
        handler, path = gf.tmp_file(suffix=".wav")
        try:
            anchors, duration, characters = synth.synthesize(text, path)
            self.assertEqual([anchor[0] for anchor in anchors], [TimeValue("0"), TimeValue("0.000125")])
            self.assertEqual(duration, TimeValue("0.000250"))
            self.assertEqual(characters, 10)
            self.assertEqual(len(self.calls), 1)
            reader = wave.open(path, "rb")
            self.assertEqual(reader.getnframes(), 8)
            reader.close()
        finally:
            synth.clear_cache()
            gf.delete_file(handler, path)


if __name__ == "__main__":
    unittest.main()
