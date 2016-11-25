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

* :class:`~aeneas.vad.VAD`,
  a simple voice activity detector
  based on the energy of the 0-th MFCC.

Given an energy vector representing an audio file,
it will return a boolean mask
with elements set to ``True`` where speech is,
and ``False`` where nonspeech occurs.

.. versionadded:: 1.0.4
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy

from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration


class VAD(Loggable):
    """
    The voice activity detector (VAD).

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    TAG = u"VAD"

    def run_vad(
        self,
        wave_energy,
        log_energy_threshold=None,
        min_nonspeech_length=None,
        extend_before=None,
        extend_after=None
    ):
        """
        Compute the time intervals containing speech and nonspeech,
        and return a boolean mask with speech frames set to ``True``,
        and nonspeech frames set to ``False``.

        The last four parameters might be ``None``:
        in this case, the corresponding RuntimeConfiguration values
        are applied.

        :param wave_energy: the energy vector of the audio file (0-th MFCC)
        :type  wave_energy: :class:`numpy.ndarray` (1D)
        :param float log_energy_threshold: the minimum log energy threshold to consider a frame as speech
        :param int min_nonspeech_length: the minimum length, in frames, of a nonspeech interval
        :param int extend_before: extend each speech interval by this number of frames to the left (before)
        :param int extend_after: extend each speech interval by this number of frames to the right (after)
        :rtype: :class:`numpy.ndarray` (1D)
        """
        self.log(u"Computing VAD for wave")
        mfcc_window_shift = self.rconf.mws
        self.log([u"MFCC window shift (s):         %.3f", mfcc_window_shift])
        if log_energy_threshold is None:
            log_energy_threshold = self.rconf[RuntimeConfiguration.VAD_LOG_ENERGY_THRESHOLD]
            self.log([u"Log energy threshold:          %.3f", log_energy_threshold])
        if min_nonspeech_length is None:
            min_nonspeech_length = int(self.rconf[RuntimeConfiguration.VAD_MIN_NONSPEECH_LENGTH] / mfcc_window_shift)
            self.log([u"Min nonspeech length (s):      %.3f", self.rconf[RuntimeConfiguration.VAD_MIN_NONSPEECH_LENGTH]])
        if extend_before is None:
            extend_before = int(self.rconf[RuntimeConfiguration.VAD_EXTEND_SPEECH_INTERVAL_BEFORE] / mfcc_window_shift)
            self.log([u"Extend speech before (s):      %.3f", self.rconf[RuntimeConfiguration.VAD_EXTEND_SPEECH_INTERVAL_BEFORE]])
        if extend_after is None:
            extend_after = int(self.rconf[RuntimeConfiguration.VAD_EXTEND_SPEECH_INTERVAL_AFTER] / mfcc_window_shift)
            self.log([u"Extend speech after (s):       %.3f", self.rconf[RuntimeConfiguration.VAD_EXTEND_SPEECH_INTERVAL_AFTER]])
        energy_length = len(wave_energy)
        energy_threshold = numpy.min(wave_energy) + log_energy_threshold
        self.log([u"Min nonspeech length (frames): %d", min_nonspeech_length])
        self.log([u"Extend speech before (frames): %d", extend_before])
        self.log([u"Extend speech after (frames):  %d", extend_after])
        self.log([u"Energy vector length (frames): %d", energy_length])
        self.log([u"Energy threshold (log):        %.3f", energy_threshold])

        # using windows to be sure we have at least
        # min_nonspeech_length consecutive frames with nonspeech
        self.log(u"Determining initial labels...")
        mask = wave_energy >= energy_threshold
        windows = self._rolling_window(mask, min_nonspeech_length)
        nonspeech_runs = self._compute_runs((numpy.where(numpy.sum(windows, axis=1) == 0))[0])
        self.log(u"Determining initial labels... done")

        # initially, everything is marked as speech
        # we remove the nonspeech intervals as needed,
        # possibly extending the adjacent speech interval
        # if requested by the user
        self.log(u"Determining final labels...")
        mask = numpy.ones(energy_length, dtype="bool")
        for ns in nonspeech_runs:
            start = ns[0]
            if (extend_after > 0) and (start > 0):
                start += extend_after
            stop = ns[-1] + min_nonspeech_length
            if (extend_before > 0) and (stop < energy_length - 1):
                stop -= extend_before
            mask[start:stop] = 0
        self.log(u"Determining final labels... done")
        return mask

    @classmethod
    def _compute_runs(self, array):
        """
        Compute runs as a list of arrays,
        each containing the indices of a contiguous run.

        :param array: the data array
        :type  array: numpy 1D array
        :rtype: list of numpy 1D arrays
        """
        if len(array) < 1:
            return []
        return numpy.split(array, numpy.where(numpy.diff(array) != 1)[0] + 1)

    @classmethod
    def _rolling_window(self, array, size):
        """
        Compute rolling windows of width ``size`` of the given array.

        Return a numpy 2D stride array,
        where rows are the windows, each of ``size`` elements.

        :param array: the data array
        :type  array: numpy 1D array (n)
        :param int size: the width of each window
        :rtype: numpy 2D stride array (n // size, size)
        """
        shape = array.shape[:-1] + (array.shape[-1] - size + 1, size)
        strides = array.strides + (array.strides[-1],)
        return numpy.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
