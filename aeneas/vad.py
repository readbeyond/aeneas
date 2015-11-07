#!/usr/bin/env python
# coding=utf-8

"""
This module contains the implementation
of a simple Voice Activity Detector (VAD),
based on the energy of the first MFCC component.

Given an audio file, it will compute
a list of non-overlapping
time intervals where speech has been detected,
and its complementary list,
that is a list of non-overlapping nonspeech time intervals.

.. versionadded:: 1.0.4
"""

import numpy
import os

from aeneas.audiofile import AudioFileMonoWAV
from aeneas.logger import Logger
import aeneas.globalconstants as gc

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

class VAD(object):
    """
    The VAD extractor.

    :param wave_mfcc: the MFCCs of the audio file
    :type  wave_mfcc: numpy 2D array
    :param wave_len: the duration of the audio file
    :type  wave_len: float
    :param frame_rate: the MFCC frame rate, in frames per second. Default:
                       :class:`aeneas.globalconstants.MFCC_FRAME_RATE`
    :type  frame_rate: int
    :param energy_threshold: the threshold for the VAD algorithm to decide
                             that a given frame contains speech. Note that
                             this is the log10 of the energy coefficient.
                             Default:
                             :class:`aeneas.globalconstants.VAD_LOG_ENERGY_THRESHOLD`
    :type  energy_threshold: float
    :param min_nonspeech_length: the minimum number of nonspeech frames
                                 the VAD algorithm must encounter
                                 to create a nonspeech interval. Default:
                                 :class:`aeneas.globalconstants.VAD_MIN_NONSPEECH_LENGTH`
    :type  min_nonspeech_length: int
    :param extend_after: extend a speech interval by this many frames after.
                         Default: :class:`aeneas.globalconstants.VAD_EXTEND_SPEECH_INTERVAL_AFTER`
    :type  extend_after: int
    :param extend_before: extend a speech interval by this many frames before.
                          Default: :class:`aeneas.globalconstants.VAD_EXTEND_SPEECH_INTERVAL_BEFORE`
    :type  extend_before: int
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = "VAD"

    def __init__(
            self,
            wave_mfcc,
            wave_len,
            frame_rate=gc.MFCC_FRAME_RATE,
            energy_threshold=gc.VAD_LOG_ENERGY_THRESHOLD,
            min_nonspeech_length=gc.VAD_MIN_NONSPEECH_LENGTH,
            extend_after=gc.VAD_EXTEND_SPEECH_INTERVAL_AFTER,
            extend_before=gc.VAD_EXTEND_SPEECH_INTERVAL_BEFORE,
            logger=None
        ):
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()
        self.wave_mfcc = wave_mfcc
        self.wave_len = wave_len
        self.frame_rate = frame_rate
        self.energy_threshold = energy_threshold
        self.min_nonspeech_length = min_nonspeech_length
        self.extend_after = extend_after
        self.extend_before = extend_before
        self.speech = None
        self.nonspeech = None

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    @property
    def speech(self):
        """
        Return the list of time intervals containing speech,
        as a list of lists, each being a pair of floats: ::

        [[s_1, e_1], [s_2, e_2], ..., [s_k, e_k]]

        where ``s_i`` is the time when the ``i``-th interval starts,
        and ``e_i`` is the time when it ends.

        :rtype: list of pairs of floats (see above)
        """
        return self.__speech

    @speech.setter
    def speech(self, speech):
        self.__speech = speech

    @property
    def nonspeech(self):
        """
        Return the list of time intervals not containing speech,
        as a list of lists, each being a pair of floats: ::

        [[s_1, e_1], [s_2, e_2], ..., [s_j, e_j]]

        where ``s_i`` is the time when the ``i``-th interval starts,
        and ``e_i`` is the time when it ends.

        :rtype: list of pairs of floats (see above)
        """
        return self.__nonspeech

    @nonspeech.setter
    def nonspeech(self, nonspeech):
        self.__nonspeech = nonspeech

    def compute_vad(self):
        """
        Compute the time intervals containing speech and nonspeech,
        and store them internally in the corresponding properties.
        """
        self._log("Computing VAD for wave")
        self.speech, self.nonspeech = self._compute_vad()

    def _compute_vad(self):
        labels = []
        energy_vector = self.wave_mfcc[0]
        energy_threshold = numpy.min(energy_vector) + self.energy_threshold
        current_time = 0
        time_step = 1.0 / self.frame_rate
        self._log(["Time step: %.3f", time_step])
        last_index = len(energy_vector) - 1
        self._log(["Last frame index: %d", last_index])

        # decide whether each frame has speech or not,
        # based only on its energy
        self._log("Assigning initial labels")
        for current_energy in energy_vector:
            start_time = current_time
            end_time = start_time + time_step
            has_speech = False
            if current_energy >= energy_threshold:
                has_speech = True
            labels.append([start_time, end_time, current_energy, has_speech])
            current_time = end_time

        # to start a new nonspeech interval, there must be
        # at least self.min_nonspeech_length nonspeech frames ahead
        # spotty False values immersed in True runs are changed to True
        self._log("Smoothing labels")
        in_nonspeech = True
        if len(labels) > self.min_nonspeech_length:
            for i in range(len(labels) - self.min_nonspeech_length):
                if ((not labels[i][3]) and
                        (self._nonspeech_ahead(labels, i, in_nonspeech))):
                    labels[i][3] = False
                    in_nonspeech = True
                else:
                    labels[i][3] = True
                    in_nonspeech = False
            # deal with the tail
            first_index_not_set = len(labels) - self.min_nonspeech_length
            speech_at_the_end = False
            for i in range(first_index_not_set, last_index + 1):
                speech_at_the_end = speech_at_the_end or labels[i][3]
            for i in range(first_index_not_set, last_index + 1):
                labels[i][3] = speech_at_the_end

        self._log("Extending speech intervals before and after")
        self._log(["Extend before: %d", self.extend_before])
        self._log(["Extend after: %d", self.extend_after])
        in_speech = False
        run_starts = []
        run_ends = []
        for i in range(len(labels)):
            if in_speech:
                if not labels[i][3]:
                    run_ends.append(i-1)
                    in_speech = False
            else:
                if labels[i][3]:
                    run_starts.append(i)
                    in_speech = True
        if in_speech:
            run_ends.append(last_index)
        adj_starts = [max(0, x - self.extend_before) for x in run_starts]
        adj_ends = [min(x + self.extend_after, last_index) for x in run_ends]
        speech_indices = zip(adj_starts, adj_ends)

        self._log("Generating speech and nonspeech list of intervals")
        speech = []
        nonspeech = []
        nonspeech_time = 0
        for speech_interval in speech_indices:
            start, end = speech_interval
            if nonspeech_time < start:
                nonspeech.append(
                    [labels[nonspeech_time][0], labels[start - 1][1]]
                )
            speech.append([labels[start][0], labels[end][1]])
            nonspeech_time = end + 1
        if nonspeech_time < last_index:
            nonspeech.append([labels[nonspeech_time][0], labels[last_index][1]])

        self._log("Returning speech and nonspeech list of intervals")
        return speech, nonspeech

    # TODO check if a numpy sliding window is faster
    def _nonspeech_ahead(self, array, current_index, in_nonspeech):
        if in_nonspeech:
            return not array[current_index][3]
        ahead = range(current_index, current_index + self.min_nonspeech_length)
        for index in ahead:
            if array[index][3]:
                return False
        return True



