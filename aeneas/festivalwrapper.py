#!/usr/bin/env python
# coding=utf-8

"""
A wrapper for the ``festival`` TTS engine.
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
    A wrapper for the ``festival`` TTS engine.

    It will perform one or more calls like ::

        $ echo text | text2wave -eval (language_italian) -o output_file.wav

    This wrapper supports calling the TTS engine
    only via ``subprocess``.

    To use this TTS engine, specify ::

        "tts=festival|tts_path=/path/to/wave2text"

    in the ``RuntimeConfiguration`` object.

    :param rconf: a runtime configuration. Default: ``None``, meaning that
                  default settings will be used.
    :type  rconf: :class:`aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = u"FESTIVALWrapper"

    LANGUAGE_TO_VOICE = {
        Language.CS    : u"(language_czech)",
        Language.CY    : u"(language_welsh)",
        Language.EN    : u"(language_english)",
        Language.EN_GB : u"(language_british_english)",
        Language.EN_SC : u"(language_scots_gaelic)",
        Language.EN_US : u"(language_american_english)",
        Language.ES    : u"(language_castillian_spanish)",
        Language.FI    : u"(language_finnish)",
        Language.IT    : u"(language_italian)",
        Language.RU    : u"(language_russian)",
    }

    OUTPUT_MONO_WAVE = True

    SUPPORTED_LANGUAGES = [
        Language.CS,
        Language.CY,
        Language.EN,
        Language.EN_GB,
        Language.EN_SC,
        Language.EN_US,
        Language.ES,
        Language.FI,
        Language.IT,
        Language.RU
    ]

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

    def _language_to_voice_code(self, language):
        voice_code = language
        if language == Language.UK:
            voice_code = Language.RU
        self._log([u"Language to voice code: '%s' => '%s'", language, voice_code])
        return voice_code

    def _voice_code_to_subprocess(self, voice_code):
        if voice_code in self.LANGUAGE_TO_VOICE:
            return [u"-eval", self.LANGUAGE_TO_VOICE[voice_code]]
        return []



