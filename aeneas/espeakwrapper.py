#!/usr/bin/env python
# coding=utf-8

"""
Wrapper around ``espeak`` to synthesize text into a ``wav`` audio file.
"""

import os
import subprocess
from scikits.audiolab import wavread

import aeneas.globalconstants as gc
from aeneas.language import Language
from aeneas.logger import Logger

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl (www.readbeyond.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.0.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class ESPEAKWrapper(object):
    """
    Wrapper around ``espeak`` to synthesize text into a ``wav`` audio file.

    It will perform a call like ::

        $ espeak -v language_code -w /tmp/output_file.wav < text

    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = "ESPEAKWrapper"

    def __init__(self, logger=None):
        self.logger = logger
        if self.logger == None:
            self.logger = Logger()

    def _log(self, message, severity=Logger.DEBUG):
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
            self._log("Replaced '%s' with '%s'" % (Language.UK, Language.RU))
            return Language.RU
        return language

    def synthesize(self, text, language, output_file_path):
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
        :rtype: float
        """
        self._log("Synthesizing text: '%s'" % text)
        self._log("Synthesizing language: '%s'" % language)
        self._log("Synthesizing to file: '%s'" % output_file_path)

        # return 0 if no text is given
        if text == None or len(text) == 0:
            return 0

        # replace language
        language = self._replace_language(language)
        self._log("Using language: '%s'" % language)

        # call espeak
        arguments = []
        arguments += [gc.ESPEAK_PATH]
        arguments += ["-v", language]
        arguments += ["-w", output_file_path]
        self._log("Calling with arguments '%s'" % " ".join(arguments))
        self._log("Calling with text '%s'" % text)
        proc = subprocess.Popen(
            arguments,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True)
        proc.communicate(input=text.encode('utf-8'))
        proc.stdout.close()
        proc.stdin.close()
        proc.stderr.close()
        self._log("Call completed")

        # check if the output file exists
        if not os.path.exists(output_file_path):
            msg = "Output file '%s' cannot be read" % output_file_path
            self._log(msg, Logger.CRITICAL)
            raise OSError(msg)

        # return the duration of the output file
        self._log("Calling wavread to analyze file '%s'" % output_file_path)
        data, sample_frequency, encoding = wavread(output_file_path)
        duration = len(data) / float(sample_frequency)
        self._log("Duration of '%s': %f" % (output_file_path, duration))
        return duration



