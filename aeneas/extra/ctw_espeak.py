#!/usr/bin/env python
# coding=utf-8

"""
A wrapper for a custom TTS engine.
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.language import Language
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

class CustomTTSWrapper(TTSWrapper):
    """
    A wrapper for the ``espeak`` TTS engine,
    to illustrate the use of custom TTS wrapper
    loading at runtime.

    It will perform one or more calls like ::

        $ echo "text to be synthesized" | espeak -v en -w output_file.wav

    This wrapper supports calling the TTS engine
    only via ``subprocess``.

    To use this TTS engine, specify ::

        "tts=custom|tts_path=/path/to/this/file.py"

    in the ``rconf`` object.

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    TAG = u"CustomTTSWrapper"

    #
    # NOTE create aliases for the language codes
    #      supported by this TTS: in this example,
    #      English, Italian, Russian and Ukrainian
    #
    ENG = Language.ENG
    """ English """

    ITA = Language.ITA
    """ Italian """

    RUS = Language.RUS
    """ Russian """

    UKR = Language.UKR
    """ Ukrainian """

    #
    # NOTE LANGUAGE_TO_VOICE_CODE maps a language code
    #      to the corresponding voice code
    #      supported by this custom TTS wrapper;
    #      mock support for Ukrainian with Russian voice
    #
    LANGUAGE_TO_VOICE_CODE = {
        ENG : "en",
        ITA : "it",
        RUS : "ru",
        UKR : "ru",
    }
    DEFAULT_LANGUAGE = ENG

    #
    # NOTE eSpeak always outputs to PCM16 mono WAVE (RIFF)
    #
    OUTPUT_MONO_WAVE = True

    def __init__(self, rconf=None, logger=None):
        #
        # NOTE custom TTS wrappers must be implemented
        #      in a class named CustomTTSWrapper
        #      otherwise the Synthesizer will not work
        #
        # NOTE this custom TTS wrapper implements
        #      only the subprocess call method
        #      hence we set the following init parameters
        #
        super(CustomTTSWrapper, self).__init__(
            has_subprocess_call=True,
            has_c_extension_call=False,
            has_python_call=False,
            rconf=rconf,
            logger=logger
        )
        #
        # NOTE this example is minimal, as we implement only
        #      the subprocess call method
        #      hence, all we need to do is to specify
        #      how to map the command line arguments of the TTS engine
        #
        # NOTE if our TTS engine was callable via Python or a Python C extension,
        #      we would have needed to write a _synthesize_multiple_python()
        #      or a _synthesize_multiple_c_extension() function,
        #      with the same I/O interface of
        #      _synthesize_multiple_c_extension() in espeakwrapper.py
        #
        # NOTE on a command line, you will use eSpeak
        #      to synthesize some text to a WAVE file as follows:
        #
        #      $ echo "text to be synthesized" | espeak -v en -w output_file.wav
        #
        #      Observe that text is read from stdin, while the audio data
        #      is written to a file specified by a given output path,
        #      introduced by the "-w" switch.
        #      Also, there is a parameter to select the English voice ("en"),
        #      introduced by the "-v" switch.
        #
        self.set_subprocess_arguments([
            u"/usr/bin/espeak",                         # path of espeak executable; you can use just "espeak" if it is in your PATH
            u"-v",                                      # append "-v"
            TTSWrapper.CLI_PARAMETER_VOICE_CODE_STRING, # it will be replaced by the actual voice code
            u"-w",                                      # append "-w"
            TTSWrapper.CLI_PARAMETER_WAVE_PATH,         # it will be replaced by the actual output file path
            TTSWrapper.CLI_PARAMETER_TEXT_STDIN         # text is read from stdin
        ])
        #
        # NOTE if your TTS engine only reads text from a file
        #      you can use the TTSWrapper.CLI_PARAMETER_TEXT_PATH placeholder.
        #
        # NOTE if your TTS engine only writes audio data to stdout
        #      you can use the TTSWrapper.CLI_PARAMETER_WAVE_STDOUT placeholder.
        #
        # NOTE if your TTS engine needs a more complex parameter
        #      for selecting the voice, e.g. Festival needs '-eval "(language_italian)"',
        #      you can implement a _voice_code_to_subprocess() function
        #      and use the TTSWrapper.CLI_PARAMETER_VOICE_CODE_FUNCTION placeholder
        #      instead of the TTSWrapper.CLI_PARAMETER_VOICE_CODE_STRING placeholder.
        #      See the aeneas/festivalwrapper.py file for an example.
        #



