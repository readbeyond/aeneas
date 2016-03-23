#!/usr/bin/env python
# coding=utf-8

"""
This module contains the following classes:

* :class:`~aeneas.nuancettsapiwrapper.NuanceTTSAPIWrapper`, a wrapper for the Nuance TTS API engine.

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
from aeneas.ttswrapper import TTSWrapper
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

class NuanceTTSAPIWrapper(TTSWrapper):
    """
    A wrapper for the Nuance Developers TTS API.

    This wrapper supports calling the TTS engine
    only via Python.

    In abstract terms, it performs one or more calls to the
    Nuance TTS API service, and concatenate the resulting WAVE files,
    returning their anchor times.

    To use this TTS engine, specify ::

        "tts=nuance|nuance_tts_api_id=...|nuance_tts_api_key=..."

    in the ``rconf`` object, substituting your Nuance Developer API ID and Key.

    See :class:`~aeneas.ttswrapper.TTSWrapper` for the available functions.
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
        FRA_CAN: "Amelie", # F: Chantal
        ENG_AUS: "Karen",
        ENG_GBR: "Kate", # F: Serena
        ENG_IND: "Veena",
        ENG_IRL: "Moira",
        ENG_SCT: "Fiona",
        ENG_USA: "Ava", # F: Allison, F: Samantha, F: Susan, F: Zoe
        ENG_ZAF: "Tessa",
        NLD_BEL: "Ellen",
        POR_BRA: "Luciana",
        POR_PRT: "Catarina", # F: Joana
        SPA_COL: "Soledad",
        SPA_ESP: "Monica",
        SPA_MEX: "Angelica", # F: Paulina, F: Empar (Valencian)
        ARA: "Laila",
        CAT: "Jordi", # M, F: Montserrat
        CES: "Iveta", # F: Zuzana
        CMN: "Tian-Tian",
        DAN: "Ida",
        DEU: "Anna-ML", # F-ML, F-ML: Petra-ML
        ELL: "Melina",
        ENG: "Kate",
        EUS: "Miren",
        FIN: "Satu",
        FRA: "Audrey-ML", # F-ML, F: Aurelie
        GLG: "Carmela",
        HEB: "Carmit",
        HIN: "Lekha",
        HUN: "Mariska",
        IND: "Damayanti",
        ITA: "Alice-ML", # F-ML, F: Federica, F: Paola
        JPN: "Kyoko",
        KOR: "Sora",
        NLD: "Claire",
        NOR: "Nora",
        POL: "Ewa", # F: Zosia
        POR: "Catarina",
        RON: "Ioana",
        RUS: "Katya", # F: Milena
        SLK: "Laura",
        SPA: "Monica",
        SWE: "Alva",
        THA: "Kanya",
        TUR: "Cem", # M, F: Yelda
        YUE: "Sin-Ji",
    }
    DEFAULT_LANGUAGE = ENG_GBR

    OUTPUT_MONO_WAVE = True

    # Nuance TTS API specific
    END_POINT = "NMDPTTSCmdServlet/tts"
    """ Nuance TTS API end point """

    SAMPLE_RATE = 16000
    """ Synthesize 16kHz PCM16 """

    URL = "https://tts.nuancemobility.net"
    """ Nuance TTS API URL """

    TAG = u"NuanceTTSAPIWrapper"

    def __init__(self, rconf=None, logger=None):
        super(NuanceTTSAPIWrapper, self).__init__(
            has_subprocess_call=False,
            has_c_extension_call=False,
            has_python_call=True,
            rconf=rconf,
            logger=logger)

    def _synthesize_multiple_python(self, text_file, output_file_path, quit_after=None, backwards=False):
        """
        Synthesize multiple text fragments, via Python call.

        Return a tuple (anchors, total_time, num_chars).

        :rtype: (bool, (list, TimeValue, int))
        """
        #
        # generating wave data for each fragment,
        # and concatenating them together
        #
        self.log(u"Calling TTS engine via Python...")
        try:
            # open output file
            output_file = AudioFile(rconf=self.rconf, logger=self.logger)
            output_file.audio_format = "pcm16"
            output_file.audio_channels = 1
            output_file.audio_sample_rate = self.SAMPLE_RATE

            # create output
            anchors = []
            current_time = TimeValue("0.000")
            num = 0
            num_chars = 0
            fragments = text_file.fragments
            if backwards:
                fragments = fragments[::-1]
            for fragment in fragments:
                # language to voice code
                voice_code = self._language_to_voice_code(fragment.language)
                # synthesize and get the duration of the output file
                self.log([u"Synthesizing fragment %d", num])
                duration, sr_nu, enc_nu, data = self._synthesize_single_helper(
                    text=(fragment.filtered_text + u" "),
                    voice_code=voice_code
                )
                # store for later output
                anchors.append([current_time, fragment.identifier, fragment.text])
                # increase the character counter
                num_chars += fragment.characters
                # append new data
                self.log([u"Fragment %d starts at: %.3f", num, current_time])
                if duration > 0:
                    self.log([u"Fragment %d duration: %.3f", num, duration])
                    current_time += duration
                    # if backwards, we append the data reversed
                    output_file.add_samples(data, reverse=backwards)
                else:
                    self.log([u"Fragment %d has zero duration", num])
                # increment fragment counter
                num += 1
                # check if we must stop synthesizing because we have enough audio
                if (quit_after is not None) and (current_time > quit_after):
                    self.log([u"Quitting after reached duration %.3f", current_time])
                    break

            # if backwards, we need to reverse the audio samples again
            if backwards:
                output_file.reverse()

            # write output file
            self.log([u"Writing audio file '%s'", output_file_path])
            output_file.write(file_path=output_file_path)
        except Exception as exc:
            self.log_exc(u"Unexpected exception while calling TTS engine via Python", exc, None, type(exc))
            return (False, None)

        # return output
        # NOTE anchors do not make sense if backwards
        self.log([u"Returning %d time anchors", len(anchors)])
        self.log([u"Current time %.3f", current_time])
        self.log([u"Synthesized %d characters", num_chars])
        self.log(u"Calling TTS engine via Python... done")
        return (True, (anchors, current_time, num_chars))

    def _synthesize_single_python(self, text, voice_code, output_file_path):
        """
        Synthesize a single text fragment via Python call.

        :rtype: tuple (result, (duration, sample_rate, encoding, data))
        """
        self.log(u"Synthesizing using Python call...")
        data = self._synthesize_single_helper(text, voice_code, output_file_path)
        return (True, data)

    def _synthesize_single_helper(self, text, voice_code, output_file_path=None):
        """
        This is an helper function to synthesize a single text fragment via Python call.

        The caller can choose whether the output file should be written to disk or not.

        :rtype: tuple (result, (duration, sample_rate, encoding, data))
        """
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
        #print(self.rconf[RuntimeConfiguration.NUANCE_TTS_API_ID])
        #print(self.rconf[RuntimeConfiguration.NUANCE_TTS_API_KEY])
        ##print(headers)
        #print(url)
        #print(voice_code)
        #raise ValueError("Stop execution here")

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
        if output_file_path is not None:
            self.log(u"Saving to file...")
            import wave
            output_file = wave.open(output_file_path, "wb")
            output_file.setframerate(self.SAMPLE_RATE) # sample rate
            output_file.setnchannels(1)                # 1 channel, i.e. mono
            output_file.setsampwidth(2)                # 16 bit/sample, i.e. 2 bytes/sample
            output_file.writeframes(response.content)
            output_file.close()
            self.log(u"Saving to file... done")

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
        return (audio_length, audio_sample_rate, audio_format, audio_samples)



