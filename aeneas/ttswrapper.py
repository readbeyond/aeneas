#!/usr/bin/env python
# coding=utf-8

"""
An abstract wrapper for a TTS engine.
"""

from __future__ import absolute_import
from __future__ import print_function
import io
import subprocess

from aeneas.audiofile import AudioFile
from aeneas.audiofile import AudioFileUnsupportedFormatError
from aeneas.logger import Logger
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.timevalue import TimeValue
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

class TTSWrapper(object):
    """
    An abstract wrapper for a TTS engine.

    It will perform one or more calls like ::

        $ echo "text" | tts -v voice_code -w output_file.wav
        $ tts -eval "(voice_code)" -i text_file.txt -o output_file.wav

    The parameters, their order, and the switches
    can be configured in the concrete subclass
    for a specific TTS engine.

    You must make sure your TTS produces WAVE mono files
    (e.g., by calling it with suitable parameters).

    In case of multiple text fragments, the resulting WAVE files
    will be joined together in a single WAVE file.

    If call methods will be evaluated in the following order:

        1. direct Python call
        2. Python C extension
        3. via ``subprocess``

    :param bool has_subprocess_call: a subclass sets this to ``True`` to indicate
                                     that the TTS can be called via ``subprocess``
    :param bool has_c_extension_call: a subclass sets this to ``True`` to indicate
                                      that the TTS can be called via a Python C extension
    :param bool has_python_call: a subclass sets this to ``True`` to indicate
                                 that the TTS can be called via Python code
    :param rconf: a runtime configuration. Default: ``None``, meaning that
                  default settings will be used.
    :type  rconf: :class:`aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`

    :raise ValueError: if all the call arguments are ``False``
    """

    TAG = u"TTSWrapper"

    CLI_PARAMETER_TEXT_PATH = "TEXT_PATH"
    """
    Placeholder to specify the path to the UTF-8 encoded file
    containing the text to be synthesized,
    to be read by the TTS engine.
    """

    CLI_PARAMETER_TEXT_STDIN = "TEXT_STDIN"
    """
    Placeholder to specify that the TTS engine
    reads the text to be synthesized from stdin.
    """

    CLI_PARAMETER_VOICE_CODE_FUNCTION = "VOICE_CODE_FUNCTION"
    """
    Placeholder to specify a list of arguments
    for the TTS engine to select the TTS voice
    to be used for synthesizing the text.
    """

    CLI_PARAMETER_VOICE_CODE_STRING = "VOICE_CODE_STRING"
    """
    Placeholder for the voice code string.
    """

    CLI_PARAMETER_WAVE_PATH = "WAVE_PATH"
    """
    Placeholder to specify the path to the audio file
    to be synthesized by the TTS engine.
    """

    CLI_PARAMETER_WAVE_STDOUT = "WAVE_STDOUT"
    """
    Placeholder to specify that the TTS engine
    outputs the audio data to stdout.
    """

    LANGUAGE_TO_VOICE_CODE = {}
    """
    Map a language code to a voice code.
    Concrete subclasses must populate this class field,
    according to the language and voice codes
    supported by the TTS engine they wrap.
    """

    OUTPUT_MONO_WAVE = False
    """
    Set to ``True`` if the TTS outputs audio data
    in PCM16 mono WAVE (RIFF) format,
    which can be read without converting.
    """

    DEFAULT_LANGUAGE = None
    """
    The default language for this TTS engine.
    Concrete subclasses must populate this class field,
    according to the languages supported
    by the TTS engine they wrap.
    """

    def __init__(
            self,
            has_subprocess_call=True,
            has_c_extension_call=False,
            has_python_call=False,
            rconf=None,
            logger=None
    ):
        if (not has_subprocess_call) and (not has_c_extension_call) and (not has_python_call):
            raise ValueError("You must implement at least one call method: subprocess, C extension, or Python")
        self.has_subprocess_call = has_subprocess_call
        self.has_c_extension_call = has_c_extension_call
        self.has_python_call = has_python_call
        self.subprocess_arguments = []
        self.logger = logger if logger is not None else Logger()
        self.rconf = rconf if rconf is not None else RuntimeConfiguration()
        self._log([u"Has subprocess call?  %s", self.has_subprocess_call])
        self._log([u"Has C extension call? %s", self.has_c_extension_call])
        self._log([u"Has Python call?      %s", self.has_python_call])

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def _language_to_voice_code(self, language):
        """
        Translate a language value to a voice code.

        If you want to mock support for a language
        by using a voice for a similar language,
        please add it to the ``LANGUAGE_TO_VOICE_CODE`` dictionary.

        :param language: the requested language
        :type  language: :class:`aeneas.language.Language`
        :rtype: string
        """
        voice_code = self.rconf[RuntimeConfiguration.TTS_VOICE_CODE]
        if voice_code is None:
            try:
                voice_code = self.LANGUAGE_TO_VOICE_CODE[language]
            except KeyError:
                self._log([u"Language code '%s' not found in LANGUAGE_TO_VOICE_CODE", language], Logger.WARNING)
                self._log(u"Using the language code as the voice code", Logger.WARNING)
                voice_code = language
        else:
            self._log(u"TTS voice override in rconf")
        self._log([u"Language to voice code: '%s' => '%s'", language, voice_code])
        return voice_code

    def _voice_code_to_subprocess(self, voice_code):
        """
        Convert the ``voice_code`` to a list of parameters
        used when calling the TTS via subprocess.
        """
        return []

    def set_subprocess_arguments(self, subprocess_arguments):
        """
        Set the list of arguments that the wrapper will pass to ``subprocess``.

        Placeholders ``CLI_PARAMETER_*`` can be used, and they will be replaced
        by actual values in the ``_synthesize_multiple_subprocess()`` and
        ``_synthesize_single_subprocess()`` built-in functions.
        Literal parameters will be passed unchanged.

        The list should start with the path to the TTS engine.

        This function should be called in the constructor
        of concrete subclasses.

        :param list subprocess_arguments: the list of arguments to be passed to
                                          the TTS engine via subprocess
        """
        self.subprocess_arguments = subprocess_arguments
        self._log([u"Subprocess arguments: %s", subprocess_arguments])

    def synthesize_multiple(self, text_file, output_file_path, quit_after=None, backwards=False):
        """
        Synthesize the text contained in the given fragment list
        into a WAVE file.

        Return a tuple (anchors, total_time, num_chars).

        Concrete subclasses must implement at least one
        of the following private functions:

            1. _synthesize_multiple_python()
            2. _synthesize_multiple_c_extension()
            3. _synthesize_multiple_subprocess()

        :param text_file: the text file to be synthesized
        :type  text_file: :class:`aeneas.textfile.TextFile`
        :param string output_file_path: the path to the output audio file
        :param quit_after: stop synthesizing as soon as
                                 reaching this many seconds
        :type quit_after: :class:`aeneas.timevalue.TimeValue`
        :param bool backwards: if > 0, synthese from the end of the text file
        :rtype: tuple (anchors, total_time, num_chars)

        :raise TypeError: if ``text_file`` is ``None`` or
                          one of the text fragments is not a ``unicode`` object
        :raise ValueError: if ``rconf[RuntimeConfiguration.ALLOW_UNLISTED_LANGUAGES]`` is ``False`` and
                           a fragment has a language code not supported by the TTS engine, or
                           if ``text_file`` has no fragments
        :raise OSError: if output file cannot be written to ``output_file_path``
        :raise RuntimeError: if both the C extension and
                             the pure Python code did not succeed.
        """
        # check that text_file is not None
        if text_file is None:
            self._log(u"text_file is None", Logger.CRITICAL)
            raise TypeError("text_file is None")

        # check that text_file has at least one text fragment
        if len(text_file) < 1:
            self._log(u"text_file has no fragments", Logger.CRITICAL)
            raise ValueError("text_file has no fragments")

        # check that the lines in the text file all have
        # a supported language code and unicode type
        if not self.rconf[RuntimeConfiguration.ALLOW_UNLISTED_LANGUAGES]:
            for fragment in text_file.fragments:
                if fragment.language not in self.LANGUAGE_TO_VOICE_CODE:
                    self._log([u"Language '%s' is not supported by the selected TTS engine", fragment.language], Logger.CRITICAL)
                    raise ValueError("Language not supported by the selected TTS engine")
        for fragment in text_file.fragments:
            for line in fragment.lines:
                if not gf.is_unicode(line):
                    self._log(u"Text file must contain only unicode strings", Logger.CRITICAL)
                    raise TypeError("Text file must contain only unicode strings")

        # log parameters
        if quit_after is not None:
            self._log([u"Quit after reaching %.3f", quit_after])
        if backwards:
            self._log(u"Synthesizing backwards")

        # check that output_file_path can be written
        if not gf.file_can_be_written(output_file_path):
            self._log([u"Cannot write output file to '%s'", output_file_path], Logger.CRITICAL)
            raise OSError("Cannot write output file")

        # first, call Python function _synthesize_multiple_python() if available
        if self.has_python_call:
            self._log(u"Calling TTS engine via Python")
            try:
                computed, result = self._synthesize_multiple_python(text_file, output_file_path, quit_after, backwards)
                if computed:
                    self._log(u"The _synthesize_multiple_python() call was successful, returning anchors")
                    return result
                else:
                    self._log(u"The _synthesize_multiple_python() call failed")
            except Exception as exc:
                self._log(u"An unexpected exception occurred while calling _synthesize_multiple_python():", Logger.WARNING)
                self._log([u"%s", exc], Logger.WARNING)

        # call _synthesize_multiple_c_extension() or _synthesize_multiple_subprocess()
        self._log(u"Calling TTS engine via C extension or subprocess")
        c_extension_function = self._synthesize_multiple_c_extension if self.has_c_extension_call else None
        subprocess_function = self._synthesize_multiple_subprocess if self.has_subprocess_call else None
        return gf.run_c_extension_with_fallback(
            self._log,
            "cew",
            c_extension_function,
            subprocess_function,
            (text_file, output_file_path, quit_after, backwards),
            c_extension=self.rconf[RuntimeConfiguration.C_EXTENSIONS]
        )

    def _synthesize_multiple_python(self, text_file, output_file_path, quit_after=None, backwards=False):
        """
        Synthesize multiple fragments via a Python call.

        :rtype: tuple (result, (anchors, current_time, num_chars))
        """
        raise NotImplementedError("This function must be implemented in concrete subclasses")

    def _synthesize_multiple_c_extension(self, text_file, output_file_path, quit_after=None, backwards=False):
        """
        Synthesize multiple fragments via a Python C extension.

        :rtype: tuple (result, (anchors, current_time, num_chars))
        """
        raise NotImplementedError("This function must be implemented in concrete subclasses")

    def _synthesize_multiple_subprocess(self, text_file, output_file_path, quit_after=None, backwards=False):
        """
        Synthesize multiple fragments via ``subprocess``.

        :rtype: tuple (result, (anchors, current_time, num_chars))
        """
        def synthesize_and_clean(text, voice_code):
            """
            Synthesize a single fragment via subprocess,
            and immediately remove the temporary file.

            :rtype: tuple (duration, sample_rate, encoding, samples)
            """
            self._log(u"Synthesizing text...")
            handler, tmp_destination = gf.tmp_file(suffix=u".wav", root=self.rconf[RuntimeConfiguration.TMP_PATH])
            result, data = self._synthesize_single_subprocess(
                text=(text + u" "),
                voice_code=voice_code,
                output_file_path=tmp_destination
            )
            self._log([u"Removing temporary file '%s'", tmp_destination])
            gf.delete_file(handler, tmp_destination)
            self._log(u"Synthesizing text... done")
            return data

        self._log(u"Calling TTS engine via subprocess...")

        try:
            # get sample rate and encoding
            du_nu, sample_rate, encoding, da_nu = synthesize_and_clean(
                text=u"Dummy text to get sample_rate",
                voice_code=self._language_to_voice_code(self.DEFAULT_LANGUAGE)
            )

            # open output file
            output_file = AudioFile(rconf=self.rconf, logger=self.logger)
            output_file.audio_format = encoding
            output_file.audio_channels = 1
            output_file.audio_sample_rate = sample_rate

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
                self._log([u"Synthesizing fragment %d", num])
                duration, sr_nu, enc_nu, samples = synthesize_and_clean(
                    text=fragment.filtered_text,
                    voice_code=voice_code
                )
                # store for later output
                anchors.append([current_time, fragment.identifier, fragment.text])
                # increase the character counter
                num_chars += fragment.characters
                # concatenate new samples
                self._log([u"Fragment %d starts at: %.3f", num, current_time])
                if duration > 0:
                    self._log([u"Fragment %d duration: %.3f", num, duration])
                    current_time += duration
                    output_file.add_samples(samples, reverse=backwards)
                else:
                    self._log([u"Fragment %d has zero duration", num])
                # increment fragment counter
                num += 1
                # check if we must stop synthesizing because we have enough audio
                if (quit_after is not None) and (current_time > quit_after):
                    self._log([u"Quitting after reached duration %.3f", current_time])
                    break

            # minimize memory
            self._log(u"Minimizing memory...")
            output_file.minimize_memory()
            self._log(u"Minimizing memory... done")

            # if backwards, we need to reverse the audio samples again
            if backwards:
                self._log(u"Reversing audio samples...")
                output_file.reverse()
                self._log(u"Reversing audio samples... done")
            
            # write output file
            self._log([u"Writing audio file '%s'", output_file_path])
            output_file.write(file_path=output_file_path)
        except Exception as exc:
            self._log(u"Calling TTS engine via subprocess... failed")
            self._log(u"An unexpected exception occurred while calling TTS engine via subprocess:", Logger.WARNING)
            self._log([u"%s", exc], Logger.WARNING)
            return (False, None)

        # return output
        if backwards:
            self._log(u"Please note that anchor time values do not make sense since backwards=True", Logger.WARNING)
        self._log([u"Returning %d time anchors", len(anchors)])
        self._log([u"Current time %.3f", current_time])
        self._log([u"Synthesized %d characters", num_chars])
        self._log(u"Calling TTS engine via subprocess... done")
        return (True, (anchors, current_time, num_chars))

    def synthesize_single(self, text, language, output_file_path):
        """
        Create a mono WAVE audio file containing the synthesized text.

        The ``text`` must be a Unicode string encodable with UTF-8.

        Return the duration of the synthesized audio file, in seconds.

        Concrete subclasses can (but they are not required to) implement one
        of the following private functions:

            1. _synthesize_single_python()
            2. _synthesize_single_c_extension()
            3. _synthesize_single_subprocess()

        :param string text: the text to synthesize
        :param language: the language to use
        :type  language: :class:`aeneas.language.Language`
        :param string output_file_path: the path of the output audio file
        :rtype: :class:`aeneas.timevalue.TimeValue`

        :raise TypeError: if ``text`` is ``None`` or it is not a ``unicode`` object
        :raise ValueError: if ``rconf[RuntimeConfiguration.ALLOW_UNLISTED_LANGUAGES]`` is ``False`` and
                           ``language`` is not supported by the TTS engine
        :raise OSError: if output file cannot be written to ``output_file_path``
        :raise RuntimeError: if both the C extension and
                             the pure Python code did not succeed.
        """
        # check that text_file is not None
        if text is None:
            self._log(u"text is None", Logger.CRITICAL)
            raise TypeError("text is None")

        # check that text has unicode type
        if not gf.is_unicode(text):
            self._log(u"text must be a unicode string", Logger.CRITICAL)
            raise TypeError("text must be a unicode string")

        # check that output_file_path can be written
        if not gf.file_can_be_written(output_file_path):
            self._log([u"Cannot write output file to '%s'", output_file_path], Logger.CRITICAL)
            raise OSError("Cannot write output file")

        # check that the requested language is listed in language.py
        if (language not in self.LANGUAGE_TO_VOICE_CODE) and (not self.rconf[RuntimeConfiguration.ALLOW_UNLISTED_LANGUAGES]):
            self._log([u"Language '%s' is not supported by the selected TTS engine", language], Logger.CRITICAL)
            raise ValueError("Language not supported by the selected TTS engine")

        self._log([u"Synthesizing text: '%s'", text])
        self._log([u"Synthesizing language: '%s'", language])
        self._log([u"Synthesizing to file: '%s'", output_file_path])

        # return zero if text is the empty string
        if len(text) == 0:
            self._log(u"len(text) is zero: returning 0.000")
            return TimeValue("0.000")

        # language to voice code
        voice_code = self._language_to_voice_code(language)
        self._log([u"Using voice code: '%s'", voice_code])

        # first, call Python function _synthesize_single_python() if available
        if self.has_python_call:
            self._log(u"Calling TTS engine via Python")
            try:
                result = self._synthesize_single_python(text, voice_code, output_file_path)
                return result[0]
            except Exception as exc:
                self._log(u"An unexpected exception occurred while calling _synthesize_single_python():", Logger.WARNING)
                self._log([u"%s", exc], Logger.WARNING)

        # call _synthesize_single_c_extension() or _synthesize_single_subprocess()
        self._log(u"Calling TTS engine via C extension or subprocess")
        c_extension_function = self._synthesize_single_c_extension if self.has_c_extension_call else None
        subprocess_function = self._synthesize_single_subprocess if self.has_subprocess_call else None
        result = gf.run_c_extension_with_fallback(
            self._log,
            "cew",
            c_extension_function,
            subprocess_function,
            (text, voice_code, output_file_path),
            c_extension=self.rconf[RuntimeConfiguration.C_EXTENSIONS]
        )
        return result[0]

    def _synthesize_single_python(self, text, voice_code, output_file_path):
        """
        Synthesize a single text fragment via a Python call.

        :rtype: tuple (result, (duration, sample_rate, encoding, samples))
        """
        raise NotImplementedError("This function must be implemented in concrete subclasses")

    def _synthesize_single_c_extension(self, text, voice_code, output_file_path):
        """
        Synthesize a single text fragment via a Python C extension.

        :rtype: tuple (result, (duration, sample_rate, encoding, samples))
        """
        raise NotImplementedError("This function must be implemented in concrete subclasses")

    def _synthesize_single_subprocess(self, text, voice_code, output_file_path):
        """
        Synthesize a single text fragment via ``subprocess``.

        :rtype: tuple (result, (duration, sample_rate, encoding, samples))
        """
        self._log(u"Synthesizing using pure Python...")
        try:
            # if the TTS engine reads text from file,
            # write the text into a temporary file
            if self.CLI_PARAMETER_TEXT_PATH in self.subprocess_arguments:
                self._log(u"TTS engine reads text from file")
                tmp_text_file_handler, tmp_text_file_path = gf.tmp_file(suffix=u".txt", root=self.rconf[RuntimeConfiguration.TMP_PATH])
                self._log([u"Creating temporary text file '%s'...", tmp_text_file_path])
                with io.open(tmp_text_file_path, "w", encoding="utf-8") as tmp_text_file:
                    tmp_text_file.write(text)
                self._log([u"Creating temporary text file '%s'... done", tmp_text_file_path])
            else:
                self._log(u"TTS engine reads text from stdin")
                tmp_text_file_handler = None
                tmp_text_file_path = None

            # copy all relevant arguments
            self._log(u"Creating arguments list...")
            arguments = []
            for arg in self.subprocess_arguments:
                if arg == self.CLI_PARAMETER_VOICE_CODE_FUNCTION:
                    arguments.extend(self._voice_code_to_subprocess(voice_code))
                elif arg == self.CLI_PARAMETER_VOICE_CODE_STRING:
                    arguments.append(voice_code)
                elif arg == self.CLI_PARAMETER_TEXT_PATH:
                    arguments.append(tmp_text_file_path)
                elif arg == self.CLI_PARAMETER_WAVE_PATH:
                    arguments.append(output_file_path)
                elif arg == self.CLI_PARAMETER_TEXT_STDIN:
                    # placeholder, do not append
                    pass
                elif arg == self.CLI_PARAMETER_WAVE_STDOUT:
                    # placeholder, do not append
                    pass
                else:
                    arguments.append(arg)
            self._log(u"Creating arguments list... done")

            # actual call via subprocess
            self._log(u"Calling TTS engine...")
            self._log([u"Calling with arguments '%s'", arguments])
            self._log([u"Calling with text '%s'", text])
            proc = subprocess.Popen(
                arguments,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True)
            if self.CLI_PARAMETER_TEXT_STDIN in self.subprocess_arguments:
                self._log(u"Passing text via stdin...")
                if gf.PY2:
                    (stdoutdata, stderrdata) = proc.communicate(input=gf.safe_bytes(text))
                else:
                    (stdoutdata, stderrdata) = proc.communicate(input=text)
                self._log(u"Passing text via stdin... done")
            else:
                self._log(u"Passing text via file...")
                (stdoutdata, stderrdata) = proc.communicate()
                self._log(u"Passing text via file... done")
            proc.stdout.close()
            proc.stdin.close()
            proc.stderr.close()

            if self.CLI_PARAMETER_WAVE_STDOUT in self.subprocess_arguments:
                self._log(u"TTS engine wrote audio data to stdout")
                self._log([u"Writing audio data to file '%s'...", output_file_path])
                with io.open(output_file_path, "wb") as output_file:
                    output_file.write(stdoutdata)
                self._log([u"Writing audio data to file '%s'... done", output_file_path])
            else:
                self._log(u"TTS engine wrote audio data to file")

            if tmp_text_file_path is not None:
                self._log([u"Delete temporary text file '%s'", tmp_text_file_path])
                gf.delete_file(tmp_text_file_handler, tmp_text_file_path)

            self._log(u"Calling TTS ... done")
        except Exception as exc:
            self._log(u"Calling TTS ... failed")
            self._log(u"An unexpected exception occurred while calling TTS via subprocess:", Logger.WARNING)
            self._log([u"%s", exc], Logger.WARNING)
            return (False, None)

        # check the file can be read
        if not gf.file_can_be_read(output_file_path):
            self._log([u"Output file '%s' does not exist", output_file_path], Logger.CRITICAL)
            return (False, None)

        # return the duration of the output file
        try:
            # if we know the TTS outputs to PCM16 mono WAVE,
            # we can read samples directly from it,
            # without an intermediate conversion through ffmpeg
            audio_file = AudioFile(
                file_path=output_file_path,
                is_mono_wave=self.OUTPUT_MONO_WAVE,
                rconf=self.rconf,
                logger=self.logger
            )
            audio_file.read_samples_from_file()
            self._log([u"Duration of '%s': %f", output_file_path, audio_file.audio_length])
            self._log(u"Synthesizing using pure Python... done")
            return (True, (
                audio_file.audio_length,
                audio_file.audio_sample_rate,
                audio_file.audio_format,
                audio_file.audio_samples
            ))
        except (AudioFileUnsupportedFormatError, OSError) as exc:
            self._log(u"Error while trying reading the sythesized audio file", Logger.CRITICAL)
            return (False, None)



