#!/usr/bin/env python
# coding=utf-8

"""
A class to synthesize text fragments into
a single ``wav`` file,
along with the corresponding time anchors.
"""

import numpy
import os
import tempfile
from scikits.audiolab import wavread
from scikits.audiolab import wavwrite

import aeneas.globalfunctions as gf
from aeneas.espeakwrapper import ESPEAKWrapper
from aeneas.logger import Logger

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.2.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class Synthesizer(object):
    """
    A class to synthesize text fragments into
    a single ``wav`` file,
    along with the corresponding time anchors.

    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = "Synthesizer"

    def __init__(self, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def synthesize(self, text_file, audio_file_path, quit_after=None, backwards=False):
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
        :param backwards: synthesizing from the end of the text file
        :type  backwards: bool
        """

        # time anchors
        anchors = []

        # initialize time
        current_time = 0.0

        # waves is used to concatenate all the fragments WAV files
        waves = numpy.array([])

        # espeak wrapper
        espeak = ESPEAKWrapper(logger=self.logger)

        if quit_after is not None:
            self._log(["Quit after reaching %.3f", quit_after])
        if backwards:
            self._log("Synthesizing backwards")

        # for each fragment, synthesize it and concatenate it
        num = 0
        num_chars = 0
        fragments = text_file.fragments
        if backwards:
            fragments = fragments[::-1]
        for fragment in fragments:

            # synthesize and get the duration of the output file
            self._log(["Synthesizing fragment %d", num])
            handler, tmp_destination = tempfile.mkstemp(
                suffix=".wav",
                dir=gf.custom_tmp_dir()
            )
            duration = espeak.synthesize(
                text=fragment.text,
                language=fragment.language,
                output_file_path=tmp_destination
            )

            # store for later output
            anchors.append([current_time, fragment.identifier, fragment.text])

            # increase the character counter
            num_chars += fragment.characters

            # concatenate to buffer
            self._log(["Fragment %d starts at: %f", num, current_time])
            if duration > 0:
                self._log(["Fragment %d duration: %f", num, duration])
                current_time += duration
                data, sample_frequency, encoding = wavread(tmp_destination)
                #
                # TODO this might result in memory swapping
                # if we have a large number of fragments
                # is there a better way?
                #
                # NOTE since append cannot be in place,
                # it seems that the only alternative is pre-allocating
                # the destination array,
                # possibly truncating or extending it as needed
                #
                if backwards:
                    waves = numpy.append(data, waves)
                else:
                    waves = numpy.append(waves, data)
            else:
                self._log(["Fragment %d has zero duration", num])

            # remove temporary file
            self._log(["Removing temporary file '%s'", tmp_destination])
            os.close(handler)
            os.remove(tmp_destination)
            num += 1

            if (quit_after is not None) and (current_time > quit_after):
                self._log(["Quitting after reached duration %.3f", current_time])
                break

        # output WAV file, concatenation of synthesized fragments
        self._log(["Writing audio file '%s'", audio_file_path])
        wavwrite(waves, audio_file_path, sample_frequency, encoding)

        # return the time anchors
        # TODO anchors do not make sense if backwards == True
        self._log(["Returning %d time anchors", len(anchors)])
        self._log(["Current time %.3f", current_time])
        self._log(["Synthesized %d characters", num_chars])
        return (anchors, current_time, num_chars)



