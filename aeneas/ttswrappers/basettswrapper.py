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

* :class:`~aeneas.ttswrappers.basettswrapper.TTSCache`,
  a TTS cache;
* :class:`~aeneas.ttswrappers.basettswrapper.BaseTTSWrapper`,
  an abstract wrapper for a TTS engine.
"""

from __future__ import absolute_import
from __future__ import print_function
import io
import subprocess

from aeneas.audiofile import AudioFile
from aeneas.audiofile import AudioFileUnsupportedFormatError
from aeneas.exacttiming import TimeValue
from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration
import aeneas.globalfunctions as gf


class TTSCache(Loggable):
    """
    A TTS cache, that is,
    a dictionary whose keys are pairs
    ``(fragment_language, fragment_text)``
    and whose values are pairs
    ``(file_handler, file_path)``.

    An item in the cache means that the text of the key
    has been synthesized to the file
    located at the path of the corresponding value.

    Note that it is not enough to store
    the string of the text as the key,
    since the same text might be pronounced in a different language.

    Also note that the values also store the file handler,
    since we might want to close it explicitly
    before removing the file from disk.

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    TAG = u"TTSCache"

    def __init__(self, rconf=None, logger=None):
        super(TTSCache, self).__init__(rconf=rconf, logger=logger)
        self._initialize_cache()

    def _initialize_cache(self):
        self.cache = dict()
        self.log(u"Cache initialized")

    def __len__(self):
        return len(self.cache)

    def keys(self):
        """
        Return the sorted list of keys currently in the cache.

        :rtype: list of tuples ``(language, text)``
        """
        return sorted(list(self.cache.keys()))

    def is_cached(self, fragment_info):
        """
        Return ``True`` if the given ``(language, text)`` key
        is present in the cache, or ``False`` otherwise.

        :rtype: bool
        """
        return fragment_info in self.cache

    def add(self, fragment_info, file_info):
        """
        Add the given ``(key, value)`` pair to the cache.

        :param fragment_info: the text key
        :type  fragment_info: tuple of str ``(language, text)``
        :param file_info: the path value
        :type  file_info: tuple ``(handler, path)``
        :raises: ValueError if the key is already present in the cache
        """
        if self.is_cached(fragment_info):
            raise ValueError(u"Attempt to add text already cached")
        self.cache[fragment_info] = file_info

    def get(self, fragment_info):
        """
        Get the value associated with the given key.

        :param fragment_info: the text key
        :type  fragment_info: tuple of str ``(language, text)``
        :raises: KeyError if the key is not present in the cache
        """
        if not self.is_cached(fragment_info):
            raise KeyError(u"Attempt to get text not cached")
        return self.cache[fragment_info]

    def clear(self):
        """
        Clear the cache and remove all the files from disk.
        """
        self.log(u"Clearing cache...")
        for file_handler, file_info in self.cache.values():
            self.log([u"  Removing file '%s'", file_info])
            gf.delete_file(file_handler, file_info)
        self._initialize_cache()
        self.log(u"Clearing cache... done")


class BaseTTSWrapper(Loggable):
    """
    An abstract wrapper for a TTS engine.

    It calls the TTS executable or library, passing parameters
    like the text string and languages, and it produces
    a WAVE file on disk and a list of time anchors.

    In case of multiple text fragments, the resulting WAVE files
    will be joined together in a single WAVE file.

    The TTS parameters, their order, and the switches
    can be configured in the concrete subclass
    for a specific TTS engine.

    For example, it might perform one or more calls like ::

        $ echo "text" | tts -v voice_code -w output_file.wav
        or
        $ tts -eval "(voice_code)" -i text_file.txt -o output_file.wav

    The call methods will be attempted in the following order:

        1. direct Python call
        2. Python C extension
        3. TTS executable via ``subprocess``

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    :raises: NotImplementedError: if none of the call methods is available
    """

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

    CODE_TO_HUMAN = {}
    """
    Map from voice code to human-readable name.
    """

    CODE_TO_HUMAN_LIST = []
    """
    List of all language codes with their human-readable names.
    """

    OUTPUT_AUDIO_FORMAT = None
    """
    A tuple ``(codec, channels, rate)``
    specifying the format
    of the audio file generated by the TTS engine,
    for example ``("pcm_s16le", 1, 22050)``.
    If unknown, set it to ``None``:
    in this case, the audio file will be converted
    to PCM16 mono WAVE (RIFF) as needed.
    """

    DEFAULT_LANGUAGE = None
    """
    The default language for this TTS engine.
    Concrete subclasses must populate this class field,
    according to the languages supported
    by the TTS engine they wrap.
    """

    DEFAULT_TTS_PATH = None
    """
    The default path for this TTS engine,
    when called via ``subprocess``,
    otherwise set it to ``None``.
    """

    HAS_SUBPROCESS_CALL = False
    """
    If ``True``, the TTS wrapper can invoke the TTS engine
    via ``subprocess``.
    """

    HAS_C_EXTENSION_CALL = False
    """
    If ``True``, the TTS wrapper can invoke the TTS engine
    via a C extension call.
    """

    HAS_PYTHON_CALL = False
    """
    If ``True``, the TTS wrapper can invoke the TTS engine
    via a direct Python call.
    """

    C_EXTENSION_NAME = ""
    """
    If the TTS wrapper can invoke the TTS engine
    via a C extension call,
    set here the name of the corresponding Python C/C++ extension.
    """

    TAG = u"BaseTTSWrapper"

    def __init__(self, rconf=None, logger=None):
        if not (self.HAS_SUBPROCESS_CALL or self.HAS_C_EXTENSION_CALL or self.HAS_PYTHON_CALL):
            raise NotImplementedError(u"You must implement at least one call method: subprocess, C extension, or Python")
        super(BaseTTSWrapper, self).__init__(rconf=rconf, logger=logger)
        self.subprocess_arguments = []
        self.tts_path = self.rconf[RuntimeConfiguration.TTS_PATH]
        if self.tts_path is None:
            self.log(u"No tts_path specified in rconf, setting default TTS path")
            self.tts_path = self.DEFAULT_TTS_PATH
        self.use_cache = self.rconf[RuntimeConfiguration.TTS_CACHE]
        self.cache = TTSCache(rconf=rconf, logger=logger) if self.use_cache else None
        self.log([u"TTS path is             %s", self.tts_path])
        self.log([u"TTS cache?              %s", self.use_cache])
        self.log([u"Has Python      call?   %s", self.HAS_PYTHON_CALL])
        self.log([u"Has C extension call?   %s", self.HAS_C_EXTENSION_CALL])
        self.log([u"Has subprocess  call?   %s", self.HAS_SUBPROCESS_CALL])

    def _language_to_voice_code(self, language):
        """
        Translate a language value to a voice code.

        If you want to mock support for a language
        by using a voice for a similar language,
        please add it to the ``LANGUAGE_TO_VOICE_CODE`` dictionary.

        :param language: the requested language
        :type  language: :class:`~aeneas.language.Language`
        :rtype: string
        """
        voice_code = self.rconf[RuntimeConfiguration.TTS_VOICE_CODE]
        if voice_code is None:
            try:
                voice_code = self.LANGUAGE_TO_VOICE_CODE[language]
            except KeyError as exc:
                self.log_exc(u"Language code '%s' not found in LANGUAGE_TO_VOICE_CODE" % (language), exc, False, None)
                self.log_warn(u"Using the language code as the voice code")
                voice_code = language
        else:
            self.log(u"TTS voice override in rconf")
        self.log([u"Language to voice code: '%s' => '%s'", language, voice_code])
        return voice_code

    def _voice_code_to_subprocess(self, voice_code):
        """
        Convert the ``voice_code`` to a list of parameters
        used when calling the TTS via subprocess.
        """
        return []

    def clear_cache(self):
        """
        Clear the TTS cache, removing all cache files from disk.

        .. versionadded:: 1.6.0
        """
        if self.use_cache:
            self.log(u"Requested to clear TTS cache")
            self.cache.clear()

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
        # NOTE this is a method because we might need to access self.rconf,
        #      so we cannot specify the list of arguments as a class field
        self.subprocess_arguments = subprocess_arguments
        self.log([u"Subprocess arguments: %s", subprocess_arguments])

    def synthesize_multiple(self, text_file, output_file_path, quit_after=None, backwards=False):
        """
        Synthesize the text contained in the given fragment list
        into a WAVE file.

        Return a tuple (anchors, total_time, num_chars).

        Concrete subclasses must implement at least one
        of the following private functions:

            1. ``_synthesize_multiple_python()``
            2. ``_synthesize_multiple_c_extension()``
            3. ``_synthesize_multiple_subprocess()``

        :param text_file: the text file to be synthesized
        :type  text_file: :class:`~aeneas.textfile.TextFile`
        :param string output_file_path: the path to the output audio file
        :param quit_after: stop synthesizing as soon as
                                 reaching this many seconds
        :type quit_after: :class:`~aeneas.exacttiming.TimeValue`
        :param bool backwards: if > 0, synthesize from the end of the text file
        :rtype: tuple (anchors, total_time, num_chars)
        :raises: TypeError: if ``text_file`` is ``None`` or
                            one of the text fragments is not a Unicode string
        :raises: ValueError: if ``self.rconf[RuntimeConfiguration.ALLOW_UNLISTED_LANGUAGES]`` is ``False``
                             and a fragment has a language code not supported by the TTS engine, or
                             if ``text_file`` has no fragments or all its fragments are empty
        :raises: OSError: if output file cannot be written to ``output_file_path``
        :raises: RuntimeError: if both the C extension and
                               the pure Python code did not succeed.
        """
        if text_file is None:
            self.log_exc(u"text_file is None", None, True, TypeError)
        if len(text_file) < 1:
            self.log_exc(u"The text file has no fragments", None, True, ValueError)
        if text_file.chars == 0:
            self.log_exc(u"All fragments in the text file are empty", None, True, ValueError)
        if not self.rconf[RuntimeConfiguration.ALLOW_UNLISTED_LANGUAGES]:
            for fragment in text_file.fragments:
                if fragment.language not in self.LANGUAGE_TO_VOICE_CODE:
                    self.log_exc(u"Language '%s' is not supported by the selected TTS engine" % (fragment.language), None, True, ValueError)
        for fragment in text_file.fragments:
            for line in fragment.lines:
                if not gf.is_unicode(line):
                    self.log_exc(u"The text file contain a line which is not a Unicode string", None, True, TypeError)

        # log parameters
        if quit_after is not None:
            self.log([u"Quit after reaching %.3f", quit_after])
        if backwards:
            self.log(u"Synthesizing backwards")

        # check that output_file_path can be written
        if not gf.file_can_be_written(output_file_path):
            self.log_exc(u"Cannot write to output file '%s'" % (output_file_path), None, True, OSError)

        # first, call Python function _synthesize_multiple_python() if available
        if self.HAS_PYTHON_CALL:
            self.log(u"Calling TTS engine via Python")
            try:
                computed, result = self._synthesize_multiple_python(text_file, output_file_path, quit_after, backwards)
                if computed:
                    self.log(u"The _synthesize_multiple_python call was successful, returning anchors")
                    return result
                else:
                    self.log(u"The _synthesize_multiple_python call failed")
            except Exception as exc:
                self.log_exc(u"An unexpected error occurred while calling _synthesize_multiple_python", exc, False, None)

        # call _synthesize_multiple_c_extension() or _synthesize_multiple_subprocess()
        self.log(u"Calling TTS engine via C extension or subprocess")
        c_extension_function = self._synthesize_multiple_c_extension if self.HAS_C_EXTENSION_CALL else None
        subprocess_function = self._synthesize_multiple_subprocess if self.HAS_SUBPROCESS_CALL else None
        return gf.run_c_extension_with_fallback(
            self.log,
            self.C_EXTENSION_NAME,
            c_extension_function,
            subprocess_function,
            (text_file, output_file_path, quit_after, backwards),
            rconf=self.rconf
        )

    def _synthesize_multiple_python(self, text_file, output_file_path, quit_after=None, backwards=False):
        """
        Synthesize multiple fragments via a Python call.

        :rtype: tuple (result, (anchors, current_time, num_chars))
        """
        self.log(u"Synthesizing multiple via a Python call...")
        ret = self._synthesize_multiple_generic(
            helper_function=self._synthesize_single_python_helper,
            text_file=text_file,
            output_file_path=output_file_path,
            quit_after=quit_after,
            backwards=backwards
        )
        self.log(u"Synthesizing multiple via a Python call... done")
        return ret

    def _synthesize_single_python_helper(self, text, voice_code, output_file_path=None, return_audio_data=True):
        """
        This is an helper function to synthesize a single text fragment via a Python call.

        If ``output_file_path`` is ``None``,
        the audio data will not persist to file at the end of the method.

        If ``return_audio_data`` is ``True``,
        return the audio data at the end of the function call;
        if ``False``, just return ``(True, None)`` in case of success.

        :rtype: tuple (result, (duration, sample_rate, codec, data)) or (result, None)
        """
        raise NotImplementedError(u"This function must be implemented in concrete subclasses supporting Python call")

    def _synthesize_multiple_c_extension(self, text_file, output_file_path, quit_after=None, backwards=False):
        """
        Synthesize multiple fragments via a Python C extension.

        :rtype: tuple (result, (anchors, current_time, num_chars))
        """
        raise NotImplementedError(u"This function must be implemented in concrete subclasses supporting C extension call")

    def _synthesize_single_c_extension_helper(self, text, voice_code, output_file_path=None):
        """
        This is an helper function to synthesize a single text fragment via a Python C extension.

        If ``output_file_path`` is ``None``,
        the audio data will not persist to file at the end of the method.

        :rtype: tuple (result, (duration, sample_rate, codec, data))
        """
        raise NotImplementedError(u"This function might be implemented in concrete subclasses supporting C extension call")

    def _synthesize_multiple_subprocess(self, text_file, output_file_path, quit_after=None, backwards=False):
        """
        Synthesize multiple fragments via ``subprocess``.

        :rtype: tuple (result, (anchors, current_time, num_chars))
        """
        self.log(u"Synthesizing multiple via subprocess...")
        ret = self._synthesize_multiple_generic(
            helper_function=self._synthesize_single_subprocess_helper,
            text_file=text_file,
            output_file_path=output_file_path,
            quit_after=quit_after,
            backwards=backwards
        )
        self.log(u"Synthesizing multiple via subprocess... done")
        return ret

    def _synthesize_single_subprocess_helper(self, text, voice_code, output_file_path=None, return_audio_data=True):
        """
        This is an helper function to synthesize a single text fragment via ``subprocess``.

        If ``output_file_path`` is ``None``,
        the audio data will not persist to file at the end of the method.

        If ``return_audio_data`` is ``True``,
        return the audio data at the end of the function call;
        if ``False``, just return ``(True, None)`` in case of success.

        :rtype: tuple (result, (duration, sample_rate, codec, data)) or (result, None)
        """
        # return zero if text is the empty string
        if len(text) == 0:
            #
            # NOTE sample_rate, codec, data do not matter
            #      if the duration is 0.000 => set them to None
            #
            self.log(u"len(text) is zero: returning 0.000")
            return (True, (TimeValue("0.000"), None, None, None))

        # create a temporary output file if needed
        synt_tmp_file = (output_file_path is None)
        if synt_tmp_file:
            self.log(u"Synthesizer helper called with output_file_path=None => creating temporary output file")
            output_file_handler, output_file_path = gf.tmp_file(suffix=u".wav", root=self.rconf[RuntimeConfiguration.TMP_PATH])
            self.log([u"Temporary output file path is '%s'", output_file_path])

        try:
            # if the TTS engine reads text from file,
            # write the text into a temporary file
            if self.CLI_PARAMETER_TEXT_PATH in self.subprocess_arguments:
                self.log(u"TTS engine reads text from file")
                tmp_text_file_handler, tmp_text_file_path = gf.tmp_file(suffix=u".txt", root=self.rconf[RuntimeConfiguration.TMP_PATH])
                self.log([u"Creating temporary text file '%s'...", tmp_text_file_path])
                with io.open(tmp_text_file_path, "w", encoding="utf-8") as tmp_text_file:
                    tmp_text_file.write(text)
                self.log([u"Creating temporary text file '%s'... done", tmp_text_file_path])
            else:
                self.log(u"TTS engine reads text from stdin")
                tmp_text_file_handler = None
                tmp_text_file_path = None

            # copy all relevant arguments
            self.log(u"Creating arguments list...")
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
            self.log(u"Creating arguments list... done")

            # actual call via subprocess
            self.log(u"Calling TTS engine...")
            self.log([u"Calling with arguments '%s'", arguments])
            self.log([u"Calling with text '%s'", text])
            proc = subprocess.Popen(
                arguments,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            if self.CLI_PARAMETER_TEXT_STDIN in self.subprocess_arguments:
                self.log(u"Passing text via stdin...")
                if gf.PY2:
                    (stdoutdata, stderrdata) = proc.communicate(input=gf.safe_bytes(text))
                else:
                    (stdoutdata, stderrdata) = proc.communicate(input=text)
                self.log(u"Passing text via stdin... done")
            else:
                self.log(u"Passing text via file...")
                (stdoutdata, stderrdata) = proc.communicate()
                self.log(u"Passing text via file... done")
            proc.stdout.close()
            proc.stdin.close()
            proc.stderr.close()

            if self.CLI_PARAMETER_WAVE_STDOUT in self.subprocess_arguments:
                self.log(u"TTS engine wrote audio data to stdout")
                self.log([u"Writing audio data to file '%s'...", output_file_path])
                with io.open(output_file_path, "wb") as output_file:
                    output_file.write(stdoutdata)
                self.log([u"Writing audio data to file '%s'... done", output_file_path])
            else:
                self.log(u"TTS engine wrote audio data to file")

            if tmp_text_file_path is not None:
                self.log([u"Delete temporary text file '%s'", tmp_text_file_path])
                gf.delete_file(tmp_text_file_handler, tmp_text_file_path)

            self.log(u"Calling TTS ... done")
        except Exception as exc:
            self.log_exc(u"An unexpected error occurred while calling TTS engine via subprocess", exc, False, None)
            return (False, None)

        # check the file can be read
        if not gf.file_can_be_read(output_file_path):
            self.log_exc(u"Output file '%s' cannot be read" % (output_file_path), None, True, None)
            return (False, None)

        # read audio data
        ret = self._read_audio_data(output_file_path) if return_audio_data else (True, None)

        # if the output file was temporary, remove it
        if synt_tmp_file:
            self.log([u"Removing temporary output file path '%s'", output_file_path])
            gf.delete_file(output_file_handler, output_file_path)

        # return audio data or (True, None)
        return ret

    def _read_audio_data(self, file_path):
        """
        Read audio data from file.

        :rtype: tuple (True, (duration, sample_rate, codec, data)) or (False, None) on exception
        """
        try:
            self.log(u"Reading audio data...")
            # if we know the TTS outputs to PCM16 mono WAVE
            # with the correct sample rate,
            # we can read samples directly from it,
            # without an intermediate conversion through ffmpeg
            audio_file = AudioFile(
                file_path=file_path,
                file_format=self.OUTPUT_AUDIO_FORMAT,
                rconf=self.rconf,
                logger=self.logger
            )
            audio_file.read_samples_from_file()
            self.log([u"Duration of '%s': %f", file_path, audio_file.audio_length])
            self.log(u"Reading audio data... done")
            return (True, (
                audio_file.audio_length,
                audio_file.audio_sample_rate,
                audio_file.audio_format,
                audio_file.audio_samples
            ))
        except (AudioFileUnsupportedFormatError, OSError) as exc:
            self.log_exc(u"An unexpected error occurred while reading audio data", exc, True, None)
            return (False, None)

    def _synthesize_multiple_generic(self, helper_function, text_file, output_file_path, quit_after=None, backwards=False):
        """
        Synthesize multiple fragments, generic function.

        The ``helper_function`` is a function that takes parameters
        ``(text, voice_code, output_file_path)``
        and returns a tuple
        ``(result, (audio_length, audio_sample_rate, audio_format, audio_samples))``.

        :rtype: tuple (result, (anchors, current_time, num_chars))
        """
        self.log(u"Calling TTS engine using multiple generic function...")

        # get sample rate and codec
        self.log(u"Determining codec and sample rate...")
        if (self.OUTPUT_AUDIO_FORMAT is None) or (len(self.OUTPUT_AUDIO_FORMAT) != 3):
            self.log(u"Determining codec and sample rate with dummy text...")
            succeeded, data = helper_function(
                text=u"Dummy text to get sample_rate",
                voice_code=self._language_to_voice_code(self.DEFAULT_LANGUAGE),
                output_file_path=None
            )
            if not succeeded:
                self.log_crit(u"An unexpected error occurred in helper_function")
                return (False, None)
            du_nu, sample_rate, codec, da_nu = data
            self.log(u"Determining codec and sample rate with dummy text... done")
        else:
            self.log(u"Reading codec and sample rate from OUTPUT_AUDIO_FORMAT")
            codec, channels_nu, sample_rate = self.OUTPUT_AUDIO_FORMAT
        self.log(u"Determining codec and sample rate... done")
        self.log([u"  codec:       %s", codec])
        self.log([u"  sample rate: %d", sample_rate])

        # open output file
        output_file = AudioFile(rconf=self.rconf, logger=self.logger)
        output_file.audio_format = codec
        output_file.audio_channels = 1
        output_file.audio_sample_rate = sample_rate

        # create output
        anchors = []
        current_time = TimeValue("0.000")
        num_chars = 0
        fragments = text_file.fragments
        if backwards:
            fragments = fragments[::-1]
        loop_function = self._loop_use_cache if self.use_cache else self._loop_no_cache
        for num, fragment in enumerate(fragments):
            succeeded, data = loop_function(
                helper_function=helper_function,
                num=num,
                fragment=fragment
            )
            if not succeeded:
                self.log_crit(u"An unexpected error occurred in loop_function")
                return (False, None)
            duration, sr_nu, enc_nu, samples = data
            # store for later output
            anchors.append([current_time, fragment.identifier, fragment.text])
            # increase the character counter
            num_chars += fragment.characters
            # concatenate new samples
            self.log([u"Fragment %d starts at: %.3f", num, current_time])
            if duration > 0:
                self.log([u"Fragment %d duration: %.3f", num, duration])
                current_time += duration
                output_file.add_samples(samples, reverse=backwards)
            else:
                self.log([u"Fragment %d has zero duration", num])
            # check if we must stop synthesizing because we have enough audio
            if (quit_after is not None) and (current_time > quit_after):
                self.log([u"Quitting after reached duration %.3f", current_time])
                break

        # minimize memory
        self.log(u"Minimizing memory...")
        output_file.minimize_memory()
        self.log(u"Minimizing memory... done")

        # if backwards, we need to reverse the audio samples again
        if backwards:
            self.log(u"Reversing audio samples...")
            output_file.reverse()
            self.log(u"Reversing audio samples... done")

        # write output file
        self.log([u"Writing audio file '%s'", output_file_path])
        output_file.write(file_path=output_file_path)

        # return output
        if backwards:
            self.log_warn(u"Please note that anchor time values do not make sense since backwards=True")
        self.log([u"Returning %d time anchors", len(anchors)])
        self.log([u"Current time %.3f", current_time])
        self.log([u"Synthesized %d characters", num_chars])
        self.log(u"Calling TTS engine using multiple generic function... done")
        return (True, (anchors, current_time, num_chars))

    def _loop_no_cache(self, helper_function, num, fragment):
        """ Synthesize all fragments without using the cache """
        self.log([u"Examining fragment %d (no cache)...", num])
        # synthesize and get the duration of the output file
        voice_code = self._language_to_voice_code(fragment.language)
        self.log(u"Calling helper function")
        succeeded, data = helper_function(
            text=fragment.filtered_text,
            voice_code=voice_code,
            output_file_path=None,
            return_audio_data=True
        )
        # check output
        if not succeeded:
            self.log_crit(u"An unexpected error occurred in helper_function")
            return (False, None)
        self.log([u"Examining fragment %d (no cache)... done", num])
        return (True, data)

    def _loop_use_cache(self, helper_function, num, fragment):
        """ Synthesize all fragments using the cache """
        self.log([u"Examining fragment %d (cache)...", num])
        fragment_info = (fragment.language, fragment.filtered_text)
        if self.cache.is_cached(fragment_info):
            self.log(u"Fragment cached: retrieving audio data from cache")

            # read data from file, whose path is in the cache
            file_handler, file_path = self.cache.get(fragment_info)
            self.log([u"Reading cached fragment at '%s'...", file_path])
            succeeded, data = self._read_audio_data(file_path)
            if not succeeded:
                self.log_crit(u"An unexpected error occurred while reading cached audio file")
                return (False, None)
            self.log([u"Reading cached fragment at '%s'... done", file_path])
        else:
            self.log(u"Fragment not cached: synthesizing and caching")

            # creating destination file
            file_info = gf.tmp_file(suffix=u".cache.wav", root=self.rconf[RuntimeConfiguration.TMP_PATH])
            file_handler, file_path = file_info
            self.log([u"Synthesizing fragment to '%s'...", file_path])

            # synthesize and get the duration of the output file
            voice_code = self._language_to_voice_code(fragment.language)
            self.log(u"Calling helper function")
            succeeded, data = helper_function(
                text=fragment.filtered_text,
                voice_code=voice_code,
                output_file_path=file_path,
                return_audio_data=True
            )
            # check output
            if not succeeded:
                self.log_crit(u"An unexpected error occurred in helper_function")
                return (False, None)
            self.log([u"Synthesizing fragment to '%s'... done", file_path])
            duration, sr_nu, enc_nu, samples = data
            if duration > 0:
                self.log(u"Fragment has > 0 duration, adding it to cache")
                self.cache.add(fragment_info, file_info)
                self.log(u"Added fragment to cache")
            else:
                self.log(u"Fragment has zero duration, not adding it to cache")
            self.log([u"Closing file handler for cached output file path '%s'", file_path])
            gf.close_file_handler(file_handler)
        self.log([u"Examining fragment %d (cache)... done", num])
        return (True, data)
