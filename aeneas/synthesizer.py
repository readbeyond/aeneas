#!/usr/bin/env python
# coding=utf-8

"""
A class to synthesize text fragments into
a single ``wav`` file,
along with the corresponding time anchors.
"""

from __future__ import absolute_import
from __future__ import print_function
from aeneas.espeakwrapper import ESPEAKWrapper
from aeneas.logger import Logger
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.textfile import TextFile
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.4.1"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class Synthesizer(object):
    """
    A class to synthesize text fragments into
    a single ``wav`` file,
    along with the corresponding time anchors.

    :param rconf: a runtime configuration. Default: ``None``, meaning that
                  default settings will be used.
    :type  rconf: :class:`aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = u"Synthesizer"

    def __init__(self, rconf=None, logger=None):
        self.logger = logger or Logger()
        self.rconf = rconf or RuntimeConfiguration()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def synthesize(
            self,
            text_file,
            audio_file_path,
            quit_after=None,
            backwards=False
    ):
        """
        Synthesize the text contained in the given fragment list
        into a ``wav`` file.

        :param text_file: the text file to be synthesized
        :type  text_file: :class:`aeneas.textfile.TextFile`
        :param audio_file_path: the path to the output audio file
        :type  audio_file_path: string (path)
        :param quit_after: stop synthesizing as soon as
                           reaching this many seconds
        :type  quit_after: float
        :param backwards: if ``True``, synthesizing from the end of the text file
        :type  backwards: bool

        :raise TypeError: if ``text_file`` is ``None`` or not an instance of ``TextFile``
        :raise OSError: if ``audio_file_path`` cannot be written
        """
        if text_file is None:
            raise TypeError("text_file is None")
        if not isinstance(text_file, TextFile):
            raise TypeError("text_file is not an instance of TextFile")
        if not gf.file_can_be_written(audio_file_path):
            raise OSError("audio_file_path cannot be written")

        # at the moment only espeak TTS is supported
        self._log(u"Synthesizing using espeak...")
        espeak = ESPEAKWrapper(rconf=self.rconf, logger=self.logger)
        result = espeak.synthesize_multiple(
            text_file=text_file,
            output_file_path=audio_file_path,
            quit_after=quit_after,
            backwards=backwards
        )
        self._log(u"Synthesizing using espeak... done")

        if not gf.file_exists(audio_file_path):
            raise OSError("audio_file_path was not written")

        return result



