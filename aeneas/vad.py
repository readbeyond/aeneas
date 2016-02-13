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

from __future__ import absolute_import
from __future__ import print_function
import numpy

from aeneas.logger import Logger
from aeneas.runtimeconfiguration import RuntimeConfiguration

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

class VAD(object):
    """
    The VAD extractor.

    :param wave_mfcc: the MFCCs of the audio file
    :type  wave_mfcc: numpy 2D array
    :param wave_len: the duration of the audio file
    :type  wave_len: float
    :param rconf: a runtime configuration. Default: ``None``, meaning that
                  default settings will be used.
    :type  rconf: :class:`aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = u"VAD"

    def __init__(self, wave_mfcc, wave_len, rconf=None, logger=None):
        self.logger = logger or Logger()
        self.rconf = rconf or RuntimeConfiguration()
        self.wave_mfcc = wave_mfcc
        self.wave_len = wave_len
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
        self._log(u"Computing VAD for wave")
        labels = []
        energy_vector = self.wave_mfcc[0]
        energy_threshold = numpy.min(energy_vector) + self.rconf["vad_log_energy_thr"]
        current_time = 0
        time_step = self.rconf["mfcc_win_shift"]
        self._log([u"Time step: %.3f", time_step])
        last_index = len(energy_vector) - 1
        self._log([u"Last frame index: %d", last_index])

        # decide whether each frame has speech or not,
        # based only on its energy
        self._log(u"Assigning initial labels")
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
        self._log(u"Smoothing labels")
        in_nonspeech = True
        min_nonspeech_length = int(self.rconf["vad_min_ns_len"] / self.rconf["mfcc_win_shift"])
        if len(labels) > min_nonspeech_length:
            for i in range(len(labels) - min_nonspeech_length):
                if (
                        (not labels[i][3]) and
                        (self._nonspeech_ahead(labels, i, in_nonspeech, min_nonspeech_length))
                ):
                    labels[i][3] = False
                    in_nonspeech = True
                else:
                    labels[i][3] = True
                    in_nonspeech = False
            # deal with the tail
            first_index_not_set = len(labels) - min_nonspeech_length
            speech_at_the_end = False
            for i in range(first_index_not_set, last_index + 1):
                speech_at_the_end = speech_at_the_end or labels[i][3]
            for i in range(first_index_not_set, last_index + 1):
                labels[i][3] = speech_at_the_end

        extend_before = int(self.rconf["vad_extend_s_before"] / self.rconf["mfcc_win_shift"])
        extend_after = int(self.rconf["vad_extend_s_after"] / self.rconf["mfcc_win_shift"])
        self._log(u"Extending speech intervals before and after")
        self._log([u"Extend before: %d", extend_before])
        self._log([u"Extend after: %d", extend_after])
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
        adj_starts = [max(0, x - extend_before) for x in run_starts]
        adj_ends = [min(x + extend_after, last_index) for x in run_ends]
        speech_indices = zip(adj_starts, adj_ends)

        self._log(u"Generating speech and nonspeech list of intervals")
        speech = []
        nonspeech = []
        nonspeech_time = 0
        for start, end in speech_indices:
            if nonspeech_time < start:
                nonspeech.append([labels[nonspeech_time][0], labels[start - 1][1]])
            speech.append([labels[start][0], labels[end][1]])
            nonspeech_time = end + 1
        if nonspeech_time < last_index:
            nonspeech.append([labels[nonspeech_time][0], labels[last_index][1]])

        self._log(u"Setting speech and nonspeech list of intervals")
        self.speech = speech
        self.nonspeech = nonspeech

    # TODO check if a numpy sliding window is faster
    def _nonspeech_ahead(self, array, current_index, in_nonspeech, min_nonspeech_length):
        if in_nonspeech:
            return not array[current_index][3]
        ahead = range(current_index, current_index + min_nonspeech_length)
        for index in ahead:
            if array[index][3]:
                return False
        return True



