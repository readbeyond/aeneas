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

* :class:`~aeneas.ttswrappers.festivalttswrapper.FESTIVALTTSWrapper`,
  a wrapper for the ``Festival`` TTS engine.
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.language import Language
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.ttswrappers.basettswrapper import BaseTTSWrapper


class FESTIVALTTSWrapper(BaseTTSWrapper):
    """
    A wrapper for the ``Festival`` TTS engine.

    This wrapper supports calling the TTS engine
    via ``subprocess`` only.

    In abstract terms, it performs one or more calls like ::

        $ echo text | text2wave -eval (language_italian) -o output_file.wav

    To use this TTS engine, specify ::

        "tts=festival|tts_path=/path/to/wave2text"

    in the ``RuntimeConfiguration`` object.

    See :class:`~aeneas.ttswrappers.basettswrapper.BaseTTSWrapper`
    for the available functions.
    Below are listed the languages supported by this wrapper.

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    CES = Language.CES
    """ Czech """

    CYM = Language.CYM
    """ Welsh """

    ENG = Language.ENG
    """ English """

    FIN = Language.FIN
    """ Finnish """

    ITA = Language.ITA
    """ Italian """

    RUS = Language.RUS
    """ Russian """

    SPA = Language.SPA
    """ Spanish """

    ENG_GBR = "eng-GBR"
    """ English (GB) """

    ENG_SCT = "eng-SCT"
    """ English (Scotland) """

    ENG_USA = "eng-USA"
    """ English (USA) """

    LANGUAGE_TO_VOICE_CODE = {
        CES: CES,
        CYM: CYM,
        ENG: ENG,
        ENG_GBR: ENG_GBR,
        ENG_SCT: ENG_SCT,
        ENG_USA: ENG_USA,
        SPA: SPA,
        FIN: FIN,
        ITA: ITA,
        RUS: RUS
    }
    DEFAULT_LANGUAGE = ENG

    VOICE_CODE_TO_SUBPROCESS = {
        CES: u"(language_czech)",
        CYM: u"(language_welsh)",
        ENG: u"(language_english)",
        ENG_GBR: u"(language_british_english)",
        ENG_SCT: u"(language_scots_gaelic)",
        ENG_USA: u"(language_american_english)",
        SPA: u"(language_castillian_spanish)",
        FIN: u"(language_finnish)",
        ITA: u"(language_italian)",
        RUS: u"(language_russian)",
    }

    OUTPUT_AUDIO_FORMAT = ("pcm_s16le", 1, 16000)

    TAG = u"FESTIVALTTSWrapper"

    def __init__(self, rconf=None, logger=None):
        super(FESTIVALTTSWrapper, self).__init__(
            has_subprocess_call=True,
            has_c_extension_call=False,
            has_python_call=False,
            rconf=rconf,
            logger=logger
        )
        self.set_subprocess_arguments([
            self.rconf[RuntimeConfiguration.TTS_PATH],
            self.CLI_PARAMETER_VOICE_CODE_FUNCTION,
            u"-o",
            self.CLI_PARAMETER_WAVE_PATH,
            self.CLI_PARAMETER_TEXT_STDIN
        ])

    def _voice_code_to_subprocess(self, voice_code):
        return [u"-eval", self.VOICE_CODE_TO_SUBPROCESS[voice_code]]
