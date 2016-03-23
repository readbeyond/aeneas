#!/usr/bin/env python
# coding=utf-8

"""
This module contains the following classes:

* :class:`~aeneas.festivalwrapper.FESTIVALWrapper`, a wrapper for the ``Festival`` TTS engine.
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.language import Language
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.ttswrapper import TTSWrapper

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

class FESTIVALWrapper(TTSWrapper):
    """
    A wrapper for the ``Festival`` TTS engine.

    This wrapper supports calling the TTS engine
    via ``subprocess`` only.

    In abstract terms, it performs one or more calls like ::

        $ echo text | text2wave -eval (language_italian) -o output_file.wav

    To use this TTS engine, specify ::

        "tts=festival|tts_path=/path/to/wave2text"

    in the ``RuntimeConfiguration`` object.

    See :class:`~aeneas.ttswrapper.TTSWrapper` for the available functions.
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
        CES : CES,
        CYM : CYM,
        ENG : ENG,
        ENG_GBR : ENG_GBR,
        ENG_SCT : ENG_SCT,
        ENG_USA : ENG_USA,
        SPA : SPA,
        FIN : FIN,
        ITA : ITA,
        RUS : RUS
    }
    DEFAULT_LANGUAGE = ENG

    VOICE_CODE_TO_SUBPROCESS = {
        CES : u"(language_czech)",
        CYM : u"(language_welsh)",
        ENG : u"(language_english)",
        ENG_GBR : u"(language_british_english)",
        ENG_SCT : u"(language_scots_gaelic)",
        ENG_USA : u"(language_american_english)",
        SPA : u"(language_castillian_spanish)",
        FIN : u"(language_finnish)",
        ITA : u"(language_italian)",
        RUS : u"(language_russian)",
    }

    OUTPUT_MONO_WAVE = True

    TAG = u"FESTIVALWrapper"

    def __init__(self, rconf=None, logger=None):
        super(FESTIVALWrapper, self).__init__(
            has_subprocess_call=True,
            has_c_extension_call=False,
            has_python_call=False,
            rconf=rconf,
            logger=logger
        )
        self.set_subprocess_arguments([
            self.rconf[RuntimeConfiguration.TTS_PATH],
            TTSWrapper.CLI_PARAMETER_VOICE_CODE_FUNCTION,
            u"-o",
            TTSWrapper.CLI_PARAMETER_WAVE_PATH,
            TTSWrapper.CLI_PARAMETER_TEXT_STDIN
        ])

    def _voice_code_to_subprocess(self, voice_code):
        return [u"-eval", self.VOICE_CODE_TO_SUBPROCESS[voice_code]]



