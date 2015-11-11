#!/usr/bin/env python
# coding=utf-8

"""
Wrapper around ``espeak`` to synthesize text into a ``wav`` audio file.
"""

import subprocess
import tempfile

from aeneas.audiofile import AudioFileMonoWAV
from aeneas.audiofile import AudioFileUnsupportedFormatError
from aeneas.language import Language
from aeneas.logger import Logger
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.3.2"
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

    TAG = "ESPEAKWrapper"

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
        :type  language: string (from :class:`aeneas.language.Language` enumeration)
        :rtype: string (from :class:`aeneas.language.Language` enumeration)
        """
        if language == Language.UK:
            self._log(["Replaced '%s' with '%s'", Language.UK, Language.RU])
            return Language.RU
        return language

    def synthesize_multiple(
            self,
            text_file,
            output_file_path,
            quit_after=None,
            backwards=False,
            force_pure_python=False
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
        :rtype: tuple (anchors, total_time, num_chars)

        :raise TypeError: if ``text_file`` is ``None`` or one of the text fragments is not a ``unicode`` object
        :raise IOError: if output file cannot be written to ``output_file_path``
        """
        # check that text_file is not None
        if text_file is None:
            self._log("text_file is None", Logger.CRITICAL)
            raise TypeError("text_file is None")

        # check that the lines in the text file all have unicode type
        for fragment in text_file.fragments:
            for line in fragment.lines:
                if not isinstance(line, unicode):
                    self._log("Text file must contain only unicode strings", Logger.CRITICAL)
                    raise TypeError("Text file must contain only unicode strings")

        # log parameters
        if quit_after is not None:
            self._log(["Quit after reaching %.3f", quit_after])
        if backwards:
            self._log("Synthesizing backwards")

        # check that output_file_path can be written
        if not gf.file_can_be_written(output_file_path):
            self._log(["Cannot write output file to '%s'", output_file_path], Logger.CRITICAL)
            raise IOError("Cannot write output file")

        # force using pure Python code
        if force_pure_python:
            self._log("Force using pure Python code")
            return self._synthesize_multiple_pure_python(
                text_file,
                output_file_path,
                quit_after,
                backwards
            )

        # call C extension, if possible
        if gc.USE_C_EXTENSIONS:
            self._log("C extensions enabled in gc")
            if gf.can_run_c_extension("cew"):
                self._log("C extensions enabled in gc and cew can be loaded")
                try:
                    return self._synthesize_multiple_c_extension(
                        text_file,
                        output_file_path,
                        quit_after,
                        backwards
                    )
                except:
                    self._log(
                        "An error occurred running cew",
                         severity=Logger.WARNING
                    )
            else:
                self._log("C extensions enabled in gc, but cew cannot be loaded")
        else:
            self._log("C extensions disabled in gc")

        # fallback: run pure Python code
        self._log("Running the pure Python code")
        return self._synthesize_multiple_pure_python(
            text_file,
            output_file_path,
            quit_after,
            backwards
        )

    def _synthesize_multiple_c_extension(
            self,
            text_file,
            output_file_path,
            quit_after=None,
            backwards=False
    ):
        self._log("Synthesizing using C extension...")

        # convert parameters from Python values to C values
        try:
            c_quit_after = float(quit_after)
        except TypeError:
            c_quit_after = 0.0
        c_backwards = 0
        if backwards:
            c_backwards = 1
        self._log(["output_file_path: %s", output_file_path])
        self._log(["c_quit_after:     %.3f", c_quit_after])
        self._log(["c_backwards:      %d", c_backwards])
        self._log("Preparing c_text...")
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
            c_text.append((f_lang, f_text.encode("utf-8")))
        self._log("Preparing c_text... done")

        # call C extension
        self._log("Importing aeneas.cew...")
        import aeneas.cew
        self._log("Importing aeneas.cew... done")
        self._log("Calling aeneas.cew...")
        sr, sf, intervals = aeneas.cew.cew_synthesize_multiple(
            output_file_path,
            c_quit_after,
            c_backwards,
            c_text
        )
        self._log("Calling aeneas.cew... done")
        self._log(["sr: %d", sr])
        self._log(["sf: %d", sf])

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
        self._log(["Returning %d time anchors", len(anchors)])
        self._log(["Current time %.3f", current_time])
        self._log(["Synthesized %d characters", num_chars])
        self._log("Synthesizing using C extension... done")
        return (anchors, current_time, num_chars)

    def _synthesize_multiple_pure_python(
            self,
            text_file,
            output_file_path,
            quit_after=None,
            backwards=False
    ):
        self._log("Synthesizing using pure Python...")

        # get sample rate and encoding
        result = self._synthesize_fragment_pure_python("Dummy", Language.EN)
        (du_nu, sample_rate, encoding, da_nu) = result

        # open output file
        output_file = AudioFileMonoWAV(
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
            self._log(["Synthesizing fragment %d", num])
            result = self._synthesize_fragment_pure_python(
                text=fragment.filtered_text,
                language=language
            )
            (duration, sr_nu, enc_nu, data) = result
            # store for later output
            anchors.append([current_time, fragment.identifier, fragment.text])
            # increase the character counter
            num_chars += fragment.characters
            # append/prepend data 
            self._log(["Fragment %d starts at: %f", num, current_time])
            if duration > 0:
                self._log(["Fragment %d duration: %f", num, duration])
                current_time += duration
                #
                # NOTE since numpy.append cannot be in place,
                # it seems that the only alternative is pre-allocating
                # the destination array,
                # possibly truncating or extending it as needed
                #
                if backwards:
                    output_file.prepend_data(data)
                else:
                    output_file.append_data(data)
            else:
                self._log(["Fragment %d has zero duration", num])

            # increment fragment counter
            num += 1
            
            # check if we must stop synthesizing because we have enough audio
            if (quit_after is not None) and (current_time > quit_after):
                self._log(["Quitting after reached duration %.3f", current_time])
                break

        # write output file
        self._log(["Writing audio file '%s'", output_file_path])
        output_file.write(file_path=output_file_path)

        # return output
        # NOTE anchors do not make sense if backwards == True
        self._log(["Returning %d time anchors", len(anchors)])
        self._log(["Current time %.3f", current_time])
        self._log(["Synthesized %d characters", num_chars])
        self._log("Synthesizing using pure Python... done")
        return (anchors, current_time, num_chars)

    def synthesize_single(
            self,
            text,
            language,
            output_file_path,
            force_pure_python=False
    ):
        """
        Create a ``wav`` audio file containing the synthesized text.

        The ``text`` must be a unicode string encodable with UTF-8,
        otherwise ``espeak`` might fail.

        Return the duration of the synthesized audio file, in seconds.

        :param text: the text to synthesize
        :type  text: unicode
        :param language: the language to use
        :type  language: string (from :class:`aeneas.language.Language` enumeration)
        :param output_file_path: the path of the output audio file
        :type  output_file_path: string
        :param force_pure_python: force using the pure Python version
        :type  force_pure_python: bool
        :rtype: float

        :raise TypeError: if ``text`` is ``None`` or it is not a ``unicode`` object
        :raise IOError: if output file cannot be written to ``output_file_path``
        """
        # check that text_file is not None
        if text is None:
            self._log("text is None", Logger.CRITICAL)
            raise TypeError("text is None")

        # check that text has unicode type
        if not isinstance(text, unicode):
            self._log("text must be a unicode object", Logger.CRITICAL)
            raise TypeError("text must be a unicode object")

        # check that output_file_path can be written
        if not gf.file_can_be_written(output_file_path):
            self._log(["Cannot write output file to '%s'", output_file_path], Logger.CRITICAL)
            raise IOError("Cannot write output file")

        self._log(["Synthesizing text: '%s'", text])
        self._log(["Synthesizing language: '%s'", language])
        self._log(["Synthesizing to file: '%s'", output_file_path])

        # return zero if text is the empty string
        if len(text) == 0:
            self._log("len(text) is zero: returning 0.0")
            return 0.0

        # replace language
        language = self._replace_language(language)
        self._log(["Using language: '%s'", language])

        # return 0 if the requested language is not listed in language.py
        # NOTE disabling this check to allow testing new languages
        # TODO put it back, add an option in gc to allow unlisted languages
        #if language not in Language.ALLOWED_VALUES:
        #    self._log(["Language %s is not allowed", language])
        #    return 0

        # force using pure Python code
        if force_pure_python:
            self._log("Force using pure Python code")
            result = self._synthesize_single_pure_python(
                text,
                language,
                output_file_path
            )
            return result[0]

        # call C extension, if possible
        if gc.USE_C_EXTENSIONS:
            self._log("C extensions enabled in gc")
            if gf.can_run_c_extension("cew"):
                self._log("C extensions enabled in gc and cew can be loaded")
                try:
                    return self._synthesize_single_c_extension(
                        text,
                        language,
                        output_file_path
                    )
                except:
                    self._log(
                        "An error occurred running cew",
                        severity=Logger.WARNING
                    )
            else:
                self._log("C extensions enabled in gc, but cew cannot be loaded")
        else:
            self._log("C extensions disabled in gc")

        # fallback: run pure Python code
        self._log("Running the pure Python code")
        result = self._synthesize_single_pure_python(
            text,
            language,
            output_file_path
        )
        return result[0]

    def _synthesize_single_c_extension(self, text, language, output_file_path):
        """
        Synthesize a single text fragment, using cew extension.

        Return the duration of the synthesized text, in seconds.

        :rtype: float
        """
        self._log("Synthesizing using C extension...")

        self._log("Preparing c_text...")
        c_text = text.encode("utf-8")
        # NOTE language has been replaced already!
        self._log("Preparing c_text... done")
        self._log("Importing aeneas.cew...")
        import aeneas.cew
        self._log("Importing aeneas.cew... done")
        self._log("Calling aeneas.cew...")
        sr, begin, end = aeneas.cew.cew_synthesize_single(
            output_file_path,
            language,
            c_text
        )
        self._log("Calling aeneas.cew... done")

        self._log("Synthesizing using C extension... done")
        return end

    def _synthesize_single_pure_python(self, text, language, output_file_path):
        """
        Synthesize a single text fragment, pure Python.

        :rtype: tuple (duration, sample_rate, encoding, data)
        """
        self._log("Synthesizing using pure Python...")

        # NOTE language has been replaced already!

        # call espeak via subprocess
        arguments = []
        arguments += [gc.ESPEAK_PATH]
        arguments += ["-v", language]
        arguments += ["-w", output_file_path]
        self._log(["Calling with arguments '%s'", " ".join(arguments)])
        self._log(["Calling with text '%s'", text])
        proc = subprocess.Popen(
            arguments,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True)
        proc.communicate(input=text.encode("utf-8"))
        proc.stdout.close()
        proc.stdin.close()
        proc.stderr.close()
        self._log("Call completed")

        # check the file can be read
        if not gf.file_exists(output_file_path):
            self._log(["Output file '%s' does not exist", output_file_path], Logger.CRITICAL)
            raise IOError("Output file does not exist")

        # return the duration of the output file
        try:
            audio_file = AudioFileMonoWAV(
                file_path=output_file_path,
                logger=self.logger
            )
            audio_file.load_data()
            duration = audio_file.audio_length
            sample_rate = audio_file.audio_sample_rate
            encoding = audio_file.audio_format
            data = audio_file.audio_data
            self._log(["Duration of '%s': %f", output_file_path, duration])
            self._log("Synthesizing using pure Python... done")
            return (duration, sample_rate, encoding, data)
        except (AudioFileUnsupportedFormatError, IOError) as exc:
            self._log("Error while trying reading the output file")
            self._log(["Message: %s", exc])
            return (0, None, None, None)

    def _synthesize_fragment_pure_python(self, text, language):
        """
        Synthesize a single fragment, pure Python,
        and immediately remove the temporary file.
        """
        self._log("Synthesizing text...")
        handler, tmp_destination = tempfile.mkstemp(
            suffix=".wav",
            dir=gf.custom_tmp_dir()
        )
        result = self._synthesize_single_pure_python(
            # TODO check this out
            text=text + u" ",
            language=language,
            output_file_path=tmp_destination
        )
        self._log(["Removing temporary file '%s'", tmp_destination])
        gf.delete_file(handler, tmp_destination)
        self._log("Synthesizing text... done")
        return result



