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
A wrapper for a custom TTS engine.
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.language import Language
from aeneas.ttswrappers.basettswrapper import BaseTTSWrapper


class CustomTTSWrapper(BaseTTSWrapper):
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
        ENG: "en",
        ITA: "it",
        RUS: "ru",
        UKR: "ru",
    }
    DEFAULT_LANGUAGE = ENG

    #
    # NOTE eSpeak always outputs to PCM16 mono WAVE (RIFF) at 22050 Hz
    #
    OUTPUT_AUDIO_FORMAT = ("pcm_s16le", 1, 22050)

    #
    # NOTE calling eSpeak via subprocess
    #
    HAS_SUBPROCESS_CALL = True

    TAG = u"CustomTTSWrapperESPEAK"

    def __init__(self, rconf=None, logger=None):
        #
        # NOTE custom TTS wrappers must be implemented
        #      in a class named CustomTTSWrapper
        #      otherwise the Synthesizer will not work
        #
        super(CustomTTSWrapper, self).__init__(rconf=rconf, logger=logger)
        #
        # NOTE this example is minimal, as we implement only
        #      the subprocess call method
        #      hence, all we need to do is to specify
        #      how to map the command line arguments of the TTS engine
        #
        # NOTE if our TTS engine was callable via Python
        #      or a Python C extension,
        #      we would have needed to write a _synthesize_multiple_python()
        #      or a _synthesize_multiple_c_extension() function,
        #      with the same I/O interface of
        #      _synthesize_multiple_c_extension() in espeakwrapper.py
        #
        # NOTE on a command line, you will use eSpeak
        #      to synthesize some text to a WAVE file as follows:
        #
        #      $ echo "text to synthesize" | espeak -v en -w output_file.wav
        #
        #      Observe that text is read from stdin, while the audio data
        #      is written to a file specified by a given output path,
        #      introduced by the "-w" switch.
        #      Also, there is a parameter to select the English voice ("en"),
        #      introduced by the "-v" switch.
        #
        self.set_subprocess_arguments([
            u"/usr/bin/espeak",                     # path of espeak executable or just "espeak" if it is in your PATH
            u"-v",                                  # append "-v"
            self.CLI_PARAMETER_VOICE_CODE_STRING,   # it will be replaced by the actual voice code
            u"-w",                                  # append "-w"
            self.CLI_PARAMETER_WAVE_PATH,           # it will be replaced by the actual output file path
            self.CLI_PARAMETER_TEXT_STDIN           # text is read from stdin
        ])
        #
        # NOTE if your TTS engine only reads text from a file
        #      you can use the
        #      BaseTTSWrapper.CLI_PARAMETER_TEXT_PATH placeholder.
        #
        # NOTE if your TTS engine only writes audio data to stdout
        #      you can use the
        #      BaseTTSWrapper.CLI_PARAMETER_WAVE_STDOUT placeholder.
        #
        # NOTE if your TTS engine needs a more complex parameter
        #      for selecting the voice, e.g. Festival needs
        #      '-eval "(language_italian)"',
        #      you can implement a _voice_code_to_subprocess() function
        #      and use the
        #      BaseTTSWrapper.CLI_PARAMETER_VOICE_CODE_FUNCTION placeholder
        #      instead of the
        #      BaseTTSWrapper.CLI_PARAMETER_VOICE_CODE_STRING placeholder.
        #      See the aeneas/ttswrappers/festivalttswrapper.py file
        #      for an example.
        #
