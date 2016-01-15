#!/usr/bin/env python
# coding=utf-8

"""
Wrapper around ``espeak`` to synthesize text into a ``wav`` audio file.
"""

from __future__ import absolute_import
from __future__ import print_function
import subprocess

from aeneas.audiofile import AudioFileMonoWAVE
from aeneas.audiofile import AudioFileUnsupportedFormatError
from aeneas.language import Language
from aeneas.logger import Logger
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.4.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class ESPEAKWrapper(object):
    """
    Wrapper around ``espeak`` to synthesize text into a ``wav`` audio file.

    It will perform one or more calls like ::

        $ espeak -v language_code -w /tmp/output_file.wav < text

    In case of multiple text fragments, the resulting wav files
    will be joined together.

    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = u"ESPEAKWrapper"

    def __init__(self, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def _replace_language(self, language):
        """
        Mock support for a given language by
        synthesizing using a similar language.

        :param language: the requested language
        :type  language: :class:`aeneas.language.Language` enum
        :rtype: :class:`aeneas.language.Language` enum
        """
        if language == Language.UK:
            self._log([u"Replaced '%s' with '%s'", Language.UK, Language.RU])
            return Language.RU
        return language

    def synthesize_multiple(
            self,
            text_file,
            output_file_path,
            quit_after=None,
            backwards=False,
            force_pure_python=False,
            allow_unlisted_languages=False
    ):
        """
        Synthesize the text contained in the given fragment list
        into a ``wav`` file.

        :param text_file: the text file to be synthesized
        :type  text_file: :class:`aeneas.textfile.TextFile`
        :param output_file_path: the path to the output audio file
        :type  output_file_path: string (path)
        :param quit_after: stop synthesizing as soon as
                           reaching this many seconds
        :type  quit_after: float
        :param backwards: synthesizing from the end of the text file
        :type  backwards: bool
        :param force_pure_python: force using the pure Python version
        :type  force_pure_python: bool
        :param allow_unlisted_languages: if ``True``, do not emit an error
                                         if ``text_file`` contains fragments
                                         with language not listed in
                                         :class:`aeneas.language.Language`
        :type  allow_unlisted_languages: bool
        :rtype: tuple (anchors, total_time, num_chars)

        :raise TypeError: if ``text_file`` is ``None`` or
                          one of the text fragments is not a ``unicode`` object
        :raise ValueError: if ``allow_unlisted_languages`` is ``False`` and
                           a fragment has its language code not listed in
                           :class:`aeneas.language.Language`
        :raise OSError: if output file cannot be written to ``output_file_path``
        :raise RuntimeError: if both the C extension and
                             the pure Python code did not succeed.
        """
        # check that text_file is not None
        if text_file is None:
            self._log(u"text_file is None", Logger.CRITICAL)
            raise TypeError("text_file is None")

        # check that the lines in the text file all have
        # a supported language code and unicode type
        for fragment in text_file.fragments:
            if (fragment.language not in Language.ALLOWED_VALUES) and (not allow_unlisted_languages):
                self._log([u"Language '%s' is not allowed", fragment.language], Logger.CRITICAL)
                raise ValueError("Language code not allowed")
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

        return gf.run_c_extension_with_fallback(
            self._log,
            "cew",
            self._synthesize_multiple_c_extension,
            self._synthesize_multiple_pure_python,
            (text_file, output_file_path, quit_after, backwards),
            force_pure_python=force_pure_python
        )

    def _synthesize_multiple_c_extension(
            self,
            text_file,
            output_file_path,
            quit_after=None,
            backwards=False
    ):
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
        self._log(u"Preparing c_text...")
        c_text = []
        fragments = text_file.fragments
        for fragment in fragments:
            f_lang = fragment.language
            f_text = fragment.filtered_text
            if f_lang is None:
                f_lang = Language.EN
            f_lang = self._replace_language(f_lang)
            if f_text is None:
                f_text = u""
            if gf.PY2:
                # Python 2 => pass byte strings
                c_text.append((gf.safe_bytes(f_lang), gf.safe_bytes(f_text)))
            else:
                # Python 3 => pass Unicode strings
                c_text.append((gf.safe_unicode(f_lang), gf.safe_unicode(f_text)))
        self._log(u"Preparing c_text... done")

        # call C extension
        try:
            self._log(u"Importing aeneas.cew...")
            import aeneas.cew
            self._log(u"Importing aeneas.cew... done")
            self._log(u"Calling aeneas.cew...")
            sr, sf, intervals = aeneas.cew.cew_synthesize_multiple(
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
        current_time = 0.0
        num_chars = 0
        if backwards:
            fragments = fragments[::-1]
        for i in range(sf):
            # get the correct fragment
            fragment = fragments[i]
            # store for later output
            anchors.append([
                intervals[i][0],
                fragment.identifier,
                fragment.filtered_text
            ])
            # increase the character counter
            num_chars += fragment.characters
            # update current_time
            current_time = intervals[i][1]

        # return output
        # NOTE anchors do not make sense if backwards == True
        self._log([u"Returning %d time anchors", len(anchors)])
        self._log([u"Current time %.3f", current_time])
        self._log([u"Synthesized %d characters", num_chars])
        self._log(u"Synthesizing using C extension... done")
        return (True, (anchors, current_time, num_chars))

    def _synthesize_multiple_pure_python(
            self,
            text_file,
            output_file_path,
            quit_after=None,
            backwards=False
    ):
        def synthesize_and_clean(text, language):
            """
            Synthesize a single fragment, pure Python,
            and immediately remove the temporary file.
            """
            self._log(u"Synthesizing text...")
            handler, tmp_destination = gf.tmp_file(suffix=".wav")
            result, data = self._synthesize_single_pure_python(
                text=(text + u" "),
                language=language,
                output_file_path=tmp_destination
            )
            self._log([u"Removing temporary file '%s'", tmp_destination])
            gf.delete_file(handler, tmp_destination)
            self._log(u"Synthesizing text... done")
            return data

        self._log(u"Synthesizing using pure Python...")

        try:
            # get sample rate and encoding
            du_nu, sample_rate, encoding, da_nu = synthesize_and_clean(
                u"Dummy text to get sample_rate",
                Language.EN
            )

            # open output file
            output_file = AudioFileMonoWAVE(
                file_path=output_file_path,
                logger=self.logger
            )
            output_file.audio_format = encoding
            output_file.audio_sample_rate = sample_rate

            # create output
            anchors = []
            current_time = 0.0
            num = 0
            num_chars = 0
            fragments = text_file.fragments
            if backwards:
                fragments = fragments[::-1]
            for fragment in fragments:
                # replace language
                language = self._replace_language(fragment.language)
                # synthesize and get the duration of the output file
                self._log([u"Synthesizing fragment %d", num])
                duration, sr_nu, enc_nu, data = synthesize_and_clean(
                    text=fragment.filtered_text,
                    language=language
                )
                # store for later output
                anchors.append([current_time, fragment.identifier, fragment.text])
                # increase the character counter
                num_chars += fragment.characters
                # append/prepend data
                self._log([u"Fragment %d starts at: %f", num, current_time])
                if duration > 0:
                    self._log([u"Fragment %d duration: %f", num, duration])
                    current_time += duration
                    #
                    # NOTE since numpy.append cannot be in place,
                    # it seems that the only alternative to make
                    # this more efficient consists in pre-allocating
                    # the destination array,
                    # possibly truncating or extending it as needed
                    #
                    if backwards:
                        output_file.prepend_data(data)
                    else:
                        output_file.append_data(data)
                else:
                    self._log([u"Fragment %d has zero duration", num])

                # increment fragment counter
                num += 1

                # check if we must stop synthesizing because we have enough audio
                if (quit_after is not None) and (current_time > quit_after):
                    self._log([u"Quitting after reached duration %.3f", current_time])
                    break

            # write output file
            self._log([u"Writing audio file '%s'", output_file_path])
            output_file.write(file_path=output_file_path)
            self._log(u"Synthesizing using pure Python... done")
        except Exception as exc:
            self._log(u"Synthesizing using pure Python... failed")
            self._log(u"An unexpected exception occurred while running pure Python code:", Logger.WARNING)
            self._log([u"%s", exc], Logger.WARNING)
            return (False, None)

        # return output
        # NOTE anchors do not make sense if backwards == True
        self._log([u"Returning %d time anchors", len(anchors)])
        self._log([u"Current time %.3f", current_time])
        self._log([u"Synthesized %d characters", num_chars])
        self._log(u"Synthesizing using pure Python... done")
        return (True, (anchors, current_time, num_chars))

    def synthesize_single(
            self,
            text,
            language,
            output_file_path,
            force_pure_python=False,
            allow_unlisted_languages=False
    ):
        """
        Create a ``wav`` audio file containing the synthesized text.

        The ``text`` must be a unicode string encodable with UTF-8,
        otherwise ``espeak`` might fail.

        Return the duration of the synthesized audio file, in seconds.

        :param text: the text to synthesize
        :type  text: unicode
        :param language: the language to use
        :type  language: :class:`aeneas.language.Language` enum
        :param output_file_path: the path of the output audio file
        :type  output_file_path: string
        :param force_pure_python: force using the pure Python version
        :type  force_pure_python: bool
        :param allow_unlisted_languages: if ``True``, do not emit an error
                                         if ``text_file`` contains fragments
                                         with language not listed in
                                         :class:`aeneas.language.Language`
        :type  allow_unlisted_languages: bool
        :rtype: float

        :raise TypeError: if ``text`` is ``None`` or it is not a ``unicode`` object
        :raise ValueError: if ``allow_unlisted_languages`` is ``False`` and
                           a fragment has its language code not listed in
                           :class:`aeneas.language.Language`
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
        if (language not in Language.ALLOWED_VALUES) and (not allow_unlisted_languages):
            self._log([u"Language '%s' is not allowed", language], Logger.CRITICAL)
            raise ValueError("Language code not allowed")

        self._log([u"Synthesizing text: '%s'", text])
        self._log([u"Synthesizing language: '%s'", language])
        self._log([u"Synthesizing to file: '%s'", output_file_path])

        # return zero if text is the empty string
        if len(text) == 0:
            self._log(u"len(text) is zero: returning 0.0")
            return 0.0

        # replace language
        language = self._replace_language(language)
        self._log([u"Using language: '%s'", language])

        result = gf.run_c_extension_with_fallback(
            self._log,
            "cew",
            self._synthesize_single_c_extension,
            self._synthesize_single_pure_python,
            (text, language, output_file_path),
            force_pure_python=force_pure_python
        )
        return result[0]

    def _synthesize_single_c_extension(self, text, language, output_file_path):
        """
        Synthesize a single text fragment, using cew extension.

        Return the duration of the synthesized text, in seconds.

        :rtype: (bool, (float, ))
        """
        self._log(u"Synthesizing using C extension...")

        self._log(u"Preparing c_text...")
        if gf.PY2:
            # Python 2 => pass byte strings
            c_text = gf.safe_bytes(text)
        else:
            # Python 3 => pass Unicode strings
            c_text = text
        # NOTE language has been replaced already!
        self._log(u"Preparing c_text... done")

        try:
            self._log(u"Importing aeneas.cew...")
            import aeneas.cew
            self._log(u"Importing aeneas.cew... done")
            self._log(u"Calling aeneas.cew...")
            sr, begin, end = aeneas.cew.cew_synthesize_single(
                output_file_path,
                language,
                c_text
            )
            self._log(u"Calling aeneas.cew... done")
        except Exception as exc:
            self._log(u"Calling aeneas.cew... failed")
            self._log(u"An unexpected exception occurred while running cew:", Logger.WARNING)
            self._log([u"%s", exc], Logger.WARNING)
            return (False, None)

        self._log(u"Synthesizing using C extension... done")
        return (True, (end, ))

    def _synthesize_single_pure_python(self, text, language, output_file_path):
        """
        Synthesize a single text fragment, pure Python.

        :rtype: tuple (duration, sample_rate, encoding, data)
        """
        self._log(u"Synthesizing using pure Python...")

        # NOTE language has been replaced already!

        try:
            # call espeak via subprocess
            self._log(u"Calling espeak ...")
            arguments = [gc.ESPEAK_PATH, "-v", language, "-w", output_file_path]
            self._log([u"Calling with arguments '%s'", " ".join(arguments)])
            self._log([u"Calling with text '%s'", text])
            proc = subprocess.Popen(
                arguments,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True)
            if gf.PY2:
                proc.communicate(input=gf.safe_bytes(text))
            else:
                proc.communicate(input=text)
            proc.stdout.close()
            proc.stdin.close()
            proc.stderr.close()
            self._log(u"Calling espeak ... done")
        except Exception as exc:
            self._log(u"Calling espeak ... failed")
            self._log(u"An unexpected exception occurred while running pure Python code:", Logger.WARNING)
            self._log([u"%s", exc], Logger.WARNING)
            return (False, None)

        # check the file can be read
        if not gf.file_can_be_read(output_file_path):
            self._log([u"Output file '%s' does not exist", output_file_path], Logger.CRITICAL)
            return (False, None)

        # return the duration of the output file
        try:
            audio_file = AudioFileMonoWAVE(
                file_path=output_file_path,
                logger=self.logger
            )
            audio_file.load_data()
            duration = audio_file.audio_length
            sample_rate = audio_file.audio_sample_rate
            encoding = audio_file.audio_format
            data = audio_file.audio_data
            self._log([u"Duration of '%s': %f", output_file_path, duration])
            self._log(u"Synthesizing using pure Python... done")
            return (True, (duration, sample_rate, encoding, data))
        except (AudioFileUnsupportedFormatError, OSError) as exc:
            self._log(u"Error while trying reading the sythesized audio file", Logger.CRITICAL)
            return (False, None)



