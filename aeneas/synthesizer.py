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
from aeneas.festivalwrapper import FESTIVALWrapper
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
__version__ = "1.5.0"
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

    :raises: OSError if a custom TTS engine is requested but it cannot be loaded
    """

    TAG = u"Synthesizer"

    def __init__(self, rconf=None, logger=None):
        self.logger = logger if logger is not None else Logger()
        self.rconf = rconf if rconf is not None else RuntimeConfiguration()
        self.tts_engine = None
        self._select_tts_engine()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def _select_tts_engine(self):
        """
        Select the TTS engine to be used by looking at the rconf object.
        """
        self._log(u"Selecting TTS engine...")
        if self.rconf[RuntimeConfiguration.TTS] == "custom":
            self._log(u"TTS engine: custom")
            tts_path = self.rconf[RuntimeConfiguration.TTS_PATH]
            if not gf.file_can_be_read(tts_path):
                raise OSError("tts_path cannot be read")
            try:
                import imp
                self._log([u"Loading CustomTTSWrapper module from '%s'...", tts_path])
                imp.load_source("CustomTTSWrapperModule", tts_path)
                self._log([u"Loading CustomTTSWrapper module from '%s'... done", tts_path])
                self._log(u"Importing CustomTTSWrapper...")
                from CustomTTSWrapperModule import CustomTTSWrapper
                self._log(u"Importing CustomTTSWrapper... done")
                self._log(u"Creating CustomTTSWrapper instance...")
                self.tts_engine = CustomTTSWrapper(rconf=self.rconf, logger=self.logger)
                self._log(u"Creating CustomTTSWrapper instance... done")
            except Exception as exc:
                raise OSError("Unable to load custom TTS wrapper")
        elif self.rconf[RuntimeConfiguration.TTS] == "festival":
            self._log(u"TTS engine: festival")
            self.tts_engine = FESTIVALWrapper(rconf=self.rconf, logger=self.logger)
        else:
            self._log(u"TTS engine: espeak")
            self.tts_engine = ESPEAKWrapper(rconf=self.rconf, logger=self.logger)
        self._log(u"Selecting TTS engine... done")

    def output_is_mono_wave(self):
        """
        Return ``True`` if the TTS engine
        outputs a PCM16 mono WAVE file.

        :rtype: bool
        """
        if self.tts_engine is not None:
            return self.tts_engine.OUTPUT_MONO_WAVE
        return False

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

        Return a tuple ``(anchors, total_time, num_chars)``.

        :param text_file: the text file to be synthesized
        :type  text_file: :class:`aeneas.textfile.TextFile`
        :param string audio_file_path: the path to the output audio file
        :param float quit_after: stop synthesizing as soon as
                                 reaching this many seconds
        :param bool backwards: if ``True``, synthesizing from the end of the text file
        :rtype: tuple

        :raise TypeError: if ``text_file`` is ``None`` or not an instance of ``TextFile``
        :raise OSError: if ``audio_file_path`` cannot be written
        :raise OSError: if ``tts=custom`` in the RuntimeConfiguration and ``tts_path`` cannot be read 
        """
        if text_file is None:
            raise TypeError("text_file is None")
        if not isinstance(text_file, TextFile):
            raise TypeError("text_file is not an instance of TextFile")
        if not gf.file_can_be_written(audio_file_path):
            raise OSError("audio_file_path cannot be written")
        if self.tts_engine is None:
            raise ValueError("Cannot select the TTS engine")

        # synthesize
        self._log(u"Synthesizing text...")
        result = self.tts_engine.synthesize_multiple(
            text_file=text_file,
            output_file_path=audio_file_path,
            quit_after=quit_after,
            backwards=backwards
        )
        self._log(u"Synthesizing text... done")

        # TODO remove this?
        # check that the output file has been written 
        if not gf.file_exists(audio_file_path):
            raise OSError("audio_file_path was not written")

        return result



