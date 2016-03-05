#!/usr/bin/env python
# coding=utf-8

"""
A wrapper for the ``espeak`` TTS engine.
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.language import Language
from aeneas.logger import Logger
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

class ESPEAKWrapper(TTSWrapper):
    """
    A wrapper for the ``espeak`` TTS engine.

    It will perform one or more calls like ::

        $ espeak -v voice_code -w /tmp/output_file.wav < text

    This wrapper supports calling the TTS engine
    via ``subprocess`` or via Python C extension,
    and it is the default TTS engine.

    To specify the path of the TTS executable, use ::

        "tts=espeak|tts_path=/path/to/espeak"

    in the ``RuntimeConfiguration`` object.

    To run the ``cew`` Python C extension
    in a separate process via ``cewsubprocess``, use ::

        "cew_subprocess_enabled=True|cew_subprocess_path=/path/to/python"

    in the ``RuntimeConfiguration`` object.

    :param rconf: a runtime configuration. Default: ``None``, meaning that
                  default settings will be used.
    :type  rconf: :class:`aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = u"ESPEAKWrapper"

    OUTPUT_MONO_WAVE = True

    # all the languages listed in Language are supported by eSpeak
    # in the future this might change
    SUPPORTED_LANGUAGES = Language.ALLOWED_VALUES

    def __init__(self, rconf=None, logger=None):
        super(ESPEAKWrapper, self).__init__(
            has_subprocess_call=True,
            has_c_extension_call=True,
            has_python_call=False,
            rconf=rconf,
            logger=logger)
        self.set_subprocess_arguments([
            self.rconf[RuntimeConfiguration.TTS_PATH],
            u"-v",
            TTSWrapper.CLI_PARAMETER_VOICE_CODE_STRING,
            u"-w",
            TTSWrapper.CLI_PARAMETER_WAVE_PATH,
            TTSWrapper.CLI_PARAMETER_TEXT_STDIN
        ])

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def _language_to_voice_code(self, language):
        voice_code = language
        if language == Language.UK:
            voice_code = Language.RU
        self._log([u"Language to voice code: '%s' => '%s'", language, voice_code])
        return voice_code

    def _synthesize_multiple_c_extension(self, text_file, output_file_path, quit_after=None, backwards=False):
        """
        Synthesize multiple text fragments, using the cew extension.

        Return a tuple (anchors, total_time, num_chars).

        :rtype: (bool, (list, TimeValue, int))
        """
        self._log(u"Synthesizing using C extension...")

        # convert parameters from Python values to C values
        try:
            c_quit_after = float(quit_after)
        except TypeError:
            c_quit_after = 0.0
        c_backwards = 0
        if backwards:
            c_backwards = 1
        self._log([u"output_file_path: %s", output_file_path])
        self._log([u"c_quit_after:     %.3f", c_quit_after])
        self._log([u"c_backwards:      %d", c_backwards])
        self._log(u"Preparing u_text...")
        u_text = []
        fragments = text_file.fragments
        for fragment in fragments:
            f_lang = fragment.language
            f_text = fragment.filtered_text
            if f_lang is None:
                f_lang = self.default_language
            f_voice_code = self._language_to_voice_code(f_lang)
            if f_text is None:
                f_text = u""
            u_text.append((f_voice_code, f_text))
        self._log(u"Preparing u_text... done")

        # call C extension
        sr = None
        sf = None
        intervals = None
        if self.rconf[RuntimeConfiguration.CEW_SUBPROCESS_ENABLED]:
            self._log(u"Using cewsubprocess to call aeneas.cew")
            try:
                self._log(u"Importing aeneas.cewsubprocess...")
                from aeneas.cewsubprocess import CEWSubprocess
                self._log(u"Importing aeneas.cewsubprocess... done")
                self._log(u"Calling aeneas.cewsubprocess...")
                cewsub = CEWSubprocess(rconf=self.rconf, logger=self.logger)
                sr, sf, intervals = cewsub.synthesize_multiple(output_file_path, c_quit_after, c_backwards, u_text)
                self._log(u"Calling aeneas.cewsubprocess... done")
            except Exception as exc:
                self._log(u"Calling aeneas.cewsubprocess... failed")
                self._log(u"An unexpected exception occurred while running cewsubprocess:", Logger.WARNING)
                self._log([u"%s", exc], Logger.WARNING)
                # NOTE not critical, try calling aeneas.cew directly
                #return (False, None)

        if sr is None:
            self._log(u"Preparing c_text...")
            if gf.PY2:
                # Python 2 => pass byte strings
                c_text = [(gf.safe_bytes(t[0]), gf.safe_bytes(t[1])) for t in u_text]
            else:
                # Python 3 => pass Unicode strings
                c_text = [(gf.safe_unicode(t[0]), gf.safe_unicode(t[1])) for t in u_text]
            self._log(u"Preparing c_text... done")
            
            self._log(u"Calling aeneas.cew directly") 
            try:
                self._log(u"Importing aeneas.cew...")
                import aeneas.cew.cew
                self._log(u"Importing aeneas.cew... done")
                self._log(u"Calling aeneas.cew...")
                sr, sf, intervals = aeneas.cew.cew.synthesize_multiple(
                    output_file_path,
                    c_quit_after,
                    c_backwards,
                    c_text
                )
                self._log(u"Calling aeneas.cew... done")
            except Exception as exc:
                self._log(u"Calling aeneas.cew... failed")
                self._log(u"An unexpected exception occurred while running cew:", Logger.WARNING)
                self._log([u"%s", exc], Logger.WARNING)
                return (False, None)
        
        self._log([u"sr: %d", sr])
        self._log([u"sf: %d", sf])

        # create output
        anchors = []
        current_time = TimeValue("0.000")
        num_chars = 0
        if backwards:
            fragments = fragments[::-1]
        for i in range(sf):
            # get the correct fragment
            fragment = fragments[i]
            # store for later output
            anchors.append([
                TimeValue(intervals[i][0]),
                fragment.identifier,
                fragment.filtered_text
            ])
            # increase the character counter
            num_chars += fragment.characters
            # update current_time
            current_time = TimeValue(intervals[i][1])

        # return output
        # NOTE anchors do not make sense if backwards == True
        self._log([u"Returning %d time anchors", len(anchors)])
        self._log([u"Current time %.3f", current_time])
        self._log([u"Synthesized %d characters", num_chars])
        self._log(u"Synthesizing using C extension... done")
        return (True, (anchors, current_time, num_chars))

    def _synthesize_single_c_extension(self, text, voice_code, output_file_path):
        """
        Synthesize a single text fragment, using the cew extension.

        Return the duration of the synthesized text, in seconds.

        :rtype: (bool, (TimeValue, ))
        """
        self._log(u"Synthesizing using C extension...")

        end = None
        if self.rconf[RuntimeConfiguration.CEW_SUBPROCESS_ENABLED]:
            self._log(u"Using cewsubprocess to call aeneas.cew")
            try:
                self._log(u"Importing aeneas.cewsubprocess...")
                from aeneas.cewsubprocess import CEWSubprocess
                self._log(u"Importing aeneas.cewsubprocess... done")
                self._log(u"Calling aeneas.cewsubprocess...")
                cewsub = CEWSubprocess(rconf=self.rconf, logger=self.logger)
                end = cewsub.synthesize_single(output_file_path, voice_code, text)
                self._log(u"Calling aeneas.cewsubprocess... done")
            except Exception as exc:
                self._log(u"Calling aeneas.cewsubprocess... failed")
                self._log(u"An unexpected exception occurred while running cewsubprocess:", Logger.WARNING)
                self._log([u"%s", exc], Logger.WARNING)
                # NOTE not critical, try calling aeneas.cew directly
                #return (False, None)

        if end is None:
            self._log(u"Preparing c_text...")
            if gf.PY2:
                # Python 2 => pass byte strings
                c_text = gf.safe_bytes(text)
            else:
                # Python 3 => pass Unicode strings
                c_text = gf.safe_unicode(text)
            self._log(u"Preparing c_text... done")
            
            self._log(u"Calling aeneas.cew directly") 
            try:
                self._log(u"Importing aeneas.cew...")
                import aeneas.cew.cew
                self._log(u"Importing aeneas.cew... done")
                self._log(u"Calling aeneas.cew...")
                sr, begin, end = aeneas.cew.cew.synthesize_single(
                    output_file_path,
                    voice_code,
                    c_text
                )
                end = TimeValue(end)
                self._log(u"Calling aeneas.cew... done")
            except Exception as exc:
                self._log(u"Calling aeneas.cew... failed")
                self._log(u"An unexpected exception occurred while running cew:", Logger.WARNING)
                self._log([u"%s", exc], Logger.WARNING)
                return (False, None)

        self._log(u"Synthesizing using C extension... done")
        return (True, (end, ))



