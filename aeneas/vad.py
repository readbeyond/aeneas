#!/usr/bin/env python
# coding=utf-8

"""
This module contains the implementation
of a simple Voice Activity Detector (VAD),
based on the energy of the 0-th MFCC component.

Given an energy vector representing an audio file,
it will return a boolean mask with elements set to ``True`` where speech is,
and ``False`` where nonspeech occurs.

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
__version__ = "1.5.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class VAD(object):
    """
    The VAD extractor.

    :param rconf: a runtime configuration. Default: ``None``, meaning that
                  default settings will be used.
    :type  rconf: :class:`aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = u"VAD"

    def __init__(self, rconf=None, logger=None):
        self.logger = logger if logger is not None else Logger()
        self.rconf = rconf if rconf is not None else RuntimeConfiguration()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def run_vad(self, wave_energy):
        """
        Compute the time intervals containing speech and nonspeech,
        and return a boolean mask with speech frames set to ``True``,
        and nonspeech frames set to ``False``.

        :param wave_energy: the energy vector of the audio file (0-th MFCC)
        :type  wave_energy: numpy 1D array
        """
        self._log(u"Computing VAD for wave")
        mfcc_window_shift = self.rconf["mfcc_win_shift"]
        log_energy_threshold = self.rconf["vad_log_energy_thr"]
        min_nonspeech_length = int(self.rconf["vad_min_ns_len"] / mfcc_window_shift)
        extend_before = int(self.rconf["vad_extend_s_before"] / mfcc_window_shift)
        extend_after = int(self.rconf["vad_extend_s_after"] / mfcc_window_shift)
        energy_length = len(wave_energy)
        energy_threshold = numpy.min(wave_energy) + log_energy_threshold
        self._log([u"MFCC window shift (s):         %.3f", mfcc_window_shift])
        self._log([u"Log energy threshold:          %.3f", log_energy_threshold])
        self._log([u"Min nonspeech length (s):      %.3f", self.rconf["vad_min_ns_len"]])
        self._log([u"Min nonspeech length (frames): %d", min_nonspeech_length])
        self._log([u"Extend speech before (s):      %.3f", self.rconf["vad_extend_s_before"]])
        self._log([u"Extend speech before (frames): %d", extend_before])
        self._log([u"Extend speech after (s):       %.3f", self.rconf["vad_extend_s_after"]])
        self._log([u"Extend speech after (frames):  %d", extend_after])
        self._log([u"Energy vector length (frames): %d", energy_length])
        self._log([u"Energy threshold (log):        %.3f", energy_threshold])

        # using windows to be sure we have at least
        # min_nonspeech_length consecutive frames with nonspeech
        self._log(u"Determining initial labels...")
        mask = wave_energy >= energy_threshold
        windows = self._rolling_window(mask, min_nonspeech_length)
        nonspeech_runs = self._compute_runs((numpy.where(numpy.sum(windows, axis=1) == 0))[0])
        self._log(u"Determining initial labels... done")

        # initially, everything is marked as speech
        # we remove the nonspeech intervals as needed,
        # possibly extending the adjacent speech interval
        # if requested by the user
        self._log(u"Determining final labels...")
        mask = numpy.ones(energy_length, dtype="bool")
        for ns in nonspeech_runs:
            start = ns[0]
            if (extend_after > 0) and (start > 0):
                start += extend_after
            stop = ns[-1] + min_nonspeech_length
            if (extend_before > 0) and (stop < energy_length - 1):
                stop -= extend_before
            mask[start:stop] = 0
        self._log(u"Determining final labels... done")
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



