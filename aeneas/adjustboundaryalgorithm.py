#!/usr/bin/env python
# coding=utf-8

"""
Enumeration of the available algorithms to adjust
the boundary point between two fragments.

.. versionadded:: 1.0.4
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy

from aeneas.audiofilemfcc import AudioFileMFCC
from aeneas.logger import Logger
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.textfile import TextFile
from aeneas.timevalue import Decimal
from aeneas.timevalue import TimeValue

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

class AdjustBoundaryAlgorithm(object):
    """
    Enumeration of the available algorithms to adjust
    the boundary point between two consecutive fragments.

    :param algorithm: the aba algorithm to be used
    :type  algorithm: :class:`aeneas.adjustboundaryalgorithm.AdjustBoundaryAlgorithm`
    :param list parameters: a list of parameters for the aba algorithm
    :param boundary_indices: the current boundary indices,
                             with respect to the audio file full MFCCs
    :type  boundary_indices: numpy 1D array
    :param real_wave_mfcc: the audio file MFCCs
    :type  real_wave_mfcc: :class:`aeneas.audiofilemfcc.AudioFileMFCC`
    :param text_file: the text file containing the text fragments associated
    :type  text_file: :class:`aeneas.textfile.TextFile`
    :param rconf: a runtime configuration. Default: ``None``, meaning that
                  default settings will be used.
    :type  rconf: :class:`aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`

    :raises ValueError: if `algorithm` value is not allowed
    :raises TypeError: if one of `boundary_indices`, `real_wave_mfcc`,
                       or `text_file` is `None` or it has a wrong type
    """

    AFTERCURRENT = "aftercurrent"
    """
    Set the boundary at ``value`` seconds
    after the end of the current fragment.

    Example (value ``0.200`` seconds):
    
    .. image:: _static/aftercurrent.200.png
       :scale: 100%
       :align: center
       :alt: Comparison between AUTO labels and AFTERCURRENT labels with 0.200 seconds offset
    """

    AUTO = "auto"
    """
    Auto (no adjustment).
    
    Example:

    .. image:: _static/auto.png
       :scale: 100%
       :align: center
       :alt: The AUTO method does not change the time intervals
    """

    BEFORENEXT = "beforenext"
    """
    Set the boundary at ``value`` seconds
    before the beginning of the next fragment.

    Example (value ``0.200`` seconds):

    .. image:: _static/beforenext.200.png
       :scale: 100%
       :align: center
       :alt: Comparison between AUTO labels and BEFORENEXT labels with 0.200 seconds offset
    """

    OFFSET = "offset"
    """
    Offset the current boundaries by ``value`` seconds.

    Example (value ``-0.200`` seconds):
    
    .. image:: _static/offset.m200.png
       :scale: 100%
       :align: center
       :alt: Comparison between AUTO labels and OFFSET labels with value -0.200

    Example (value ``0.200`` seconds):

    .. image:: _static/offset.200.png
       :scale: 100%
       :align: center
       :alt: Comparison between AUTO labels and OFFSET labels with value 0.200

    .. versionadded:: 1.1.0
    """

    PERCENT = "percent"
    """
    Set the boundary at ``value`` percent of
    the nonspeech interval between the current and the next fragment.

    Example (value ``25`` %):

    .. image:: _static/percent.25.png
       :scale: 100%
       :align: center
       :alt: Comparison between AUTO labels and PERCENT labels with value 25 %

    Example (value ``50`` %):

    .. image:: _static/percent.50.png
       :scale: 100%
       :align: center
       :alt: Comparison between AUTO labels and PERCENT labels with value 50 %

    Example (value ``75`` %):

    .. image:: _static/percent.75.png
       :scale: 100%
       :align: center
       :alt: Comparison between AUTO labels and PERCENT labels with value 75 %
    
    """

    RATE = "rate"
    """
    Adjust boundaries trying to respect the
    ``value`` characters/second constraint.

    Example (value ``XX``): TBW
    """

    RATEAGGRESSIVE = "rateaggressive"
    """
    Adjust boundaries trying to respect the
    ``value`` characters/second constraint (aggressive mode).

    Example (value ``XX``): TBW

    .. versionadded:: 1.1.0
    """

    ALLOWED_VALUES = [
        AFTERCURRENT,
        AUTO,
        BEFORENEXT,
        OFFSET,
        PERCENT,
        RATE,
        RATEAGGRESSIVE
    ]
    """ List of all the allowed values """

    TAG = u"AdjustBoundaryAlgorithm"

    def __init__(
            self,
            algorithm,
            parameters,
            boundary_indices,
            real_wave_mfcc,
            text_file,
            rconf=None,
            logger=None
        ):
        if algorithm not in self.ALLOWED_VALUES:
            raise ValueError("Algorithm value not allowed")
        if boundary_indices is None:
            raise TypeError("boundary_indices is None")
        if (real_wave_mfcc is None) or (not isinstance(real_wave_mfcc, AudioFileMFCC)):
            raise TypeError("real_wave_mfcc is None or not an AudioFileMFCC object")
        if (text_file is None) or (not isinstance(text_file, TextFile)):
            raise TypeError("text_file is None or not a TextFile object")
        self.algorithm = algorithm
        self.parameters = parameters
        self.real_wave_mfcc = real_wave_mfcc
        self.boundary_indices = boundary_indices
        self.text_file = text_file
        self.logger = logger if logger is not None else Logger()
        self.rconf = rconf if rconf is not None else RuntimeConfiguration()
        self.intervals = []

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def to_time_map(self):
        """
        Adjust the boundaries of the text map
        using the algorithm and parameters specified
        in the constructor, and return a list
        of time intervals.

        :rtype: list of intervals
        """
        if self.algorithm == self.AUTO:
            self._adjust_auto()
        elif self.algorithm == self.AFTERCURRENT:
            self._adjust_aftercurrent()
        elif self.algorithm == self.BEFORENEXT:
            self._adjust_beforenext()
        elif self.algorithm == self.OFFSET:
            self._adjust_offset()
        elif self.algorithm == self.PERCENT:
            self._adjust_percent()
        elif self.algorithm == self.RATE:
            self._adjust_rate(False)
        elif self.algorithm == self.RATEAGGRESSIVE:
            self._adjust_rate(True)
        else:
            self._adjust_auto()
        return self.intervals

    def _adjust_auto(self):
        """
        AUTO (do not modify)
        """
        self._log(u"Called _adjust_auto")
        self._apply_offset(TimeValue("0.000"))

    def _adjust_offset(self):
        """
        OFFSET
        """
        self._log(u"Called _adjust_offset")
        # NOTE self.parameters[0] is TimeValue
        self._apply_offset(self.parameters[0])

    def _adjust_percent(self):
        """
        PERCENT
        """
        def new_time(begin, end, current):
            """ Compute new time """
            # NOTE self.parameters[0] is an int
            percent = max(min(Decimal(self.parameters[0]) / 100, 100), 0)
            return (begin + (end + 1 - begin) * percent) * self.rconf.mws
        self._log(u"Called _adjust_percent")
        self._adjust_on_nonspeech(new_time)

    def _adjust_aftercurrent(self):
        """
        AFTERCURRENT
        """
        def new_time(begin, end, current):
            """ Compute new time """
            mws = self.rconf.mws
            # NOTE self.parameters[0] is TimeValue
            delay = max(self.parameters[0], TimeValue("0.000"))
            tentative = begin * mws + delay
            if tentative > (end + 1) * mws:
                return current * mws
            return tentative
        self._log(u"Called _adjust_aftercurrent")
        self._adjust_on_nonspeech(new_time)

    def _adjust_beforenext(self):
        """
        BEFORENEXT
        """
        def new_time(begin, end, current):
            """ Compute new time """
            mws = self.rconf.mws
            # NOTE self.parameters[0] is TimeValue
            delay = max(self.parameters[0], TimeValue("0.000"))
            tentative = (end + 1) * mws - delay
            if tentative < begin * mws:
                return current * mws
            return tentative
        self._log(u"Called _adjust_beforenext")
        self._adjust_on_nonspeech(new_time)

    def _adjust_rate(self, aggressive=False):
        self._log(u"Called _adjust_rate")
        # if only one fragment, return unchanged
        if len(self.text_file) <= 1:
            self._log(u"Only one fragment, returning")
            self._apply_offset(TimeValue("0.000"))
            return

        # compute fragments too fast
        mws = self.rconf.mws
        # NOTE self.parameters[0] is float 
        max_rate = self.parameters[0]
        times = self.boundary_indices * mws
        durations = numpy.diff(times)
        lengths = numpy.array([f.chars for f in self.text_file.fragments])
        # compute rates, dealing with division by zero
        with numpy.errstate(divide="ignore", invalid="ignore"):
            rates = numpy.divide(lengths, durations)
            rates[rates == numpy.inf] = 0
            rates = numpy.nan_to_num(rates)
        faster = numpy.where(rates > max_rate)[0]

        # if no fragment is faster, return unchanged
        if len(faster) == 0:
            self._log([u"No fragment faster than max rate %.3f", max_rate])
            self._apply_offset(TimeValue("0.000"))
            return

        # try fixing faster fragments
        for index in faster:
            self._log([u"Fragment %d has rate %.3f", index, rates[index]])
            fixed = False

            # first, try moving begin time back
            if index > 0:
                self._log(u"  Trying to move begin time back...")
                lacking = lengths[index] / max_rate - durations[index]
                self._log([u"  Overflow current fragment: %.3f", lacking])
                slack = durations[index - 1] - lengths[index - 1] / max_rate
                self._log([u"  Slack previous fragment:   %.3f", slack])
                if slack >= lacking:
                    self._log([u"  Moving begin time:         %.3f => %.3f", times[index], times[index] - lacking])
                    self._log(u"  Complete fix (slack >= lacking)")
                    times[index] -= lacking
                    durations[index - 1] -= lacking
                    durations[index] += lacking
                    rates[index - 1] = lengths[index - 1] / durations[index - 1]
                    rates[index] = lengths[index] / durations[index]
                    fixed = True
                elif slack > 0:
                    self._log([u"  Moving begin time:         %.3f => %.3f", times[index], times[index] - slack])
                    self._log(u"  Partial fix (slack < lacking but slack > 0)")
                    times[index] -= slack
                    durations[index - 1] -= slack
                    durations[index] += slack
                    rates[index - 1] = lengths[index - 1] / durations[index - 1]
                    rates[index] = lengths[index] / durations[index]
                else:
                    self._log(u"  Cannot move begin time back (slack <= 0)")

            # if aggressive and not completely fixed, try moving end time forward
            if (aggressive) and (not fixed) and (index < len(self.text_file) - 1):
                self._log(u"  Trying to move end time forward...")
                lacking = lengths[index] / max_rate - durations[index]
                self._log([u"  Overflow current fragment: %.3f", lacking])
                slack = durations[index + 1] - lengths[index + 1] / max_rate
                self._log([u"  Slack next fragment:       %.3f", slack])
                if slack >= lacking:
                    self._log([u"  Moving end time:           %.3f => %.3f", times[index + 1], times[index + 1] + lacking])
                    self._log(u"  Complete fix (slack >= lacking)")
                    times[index + 1] += lacking
                    durations[index] += lacking
                    durations[index + 1] -= lacking
                    rates[index] = lengths[index] / durations[index]
                    rates[index + 1] = lengths[index + 1] / durations[index + 1]
                    fixed = True
                elif slack > 0:
                    self._log([u"  Moving end time:           %.3f => %.3f", times[index + 1], times[index + 1] + slack])
                    self._log(u"  Partial fix (slack < lacking but slack > 0)")
                    times[index + 1] += slack
                    durations[index] += slack
                    durations[index + 1] -= slack
                    rates[index] = lengths[index] / durations[index]
                    rates[index + 1] = lengths[index + 1] / durations[index + 1]
                else:
                    self._log(u"  Cannot move end time forward (slack <= 0)")

            # if not completely fixed, log warning
            if not fixed:
                self._log([u"Fragment %d is faster and could not be fixed", index], Logger.WARNING)

        # create intervals and return
        self._times_to_intervals(times)

    def _times_to_intervals(self, times):
        """
        Transform a list of time values into a list of intervals.

        For example: [0,1,2,3,4] => [[0,1], [1,2], [2,3], [3,4]]

        :param times: the time values
        :type  times: list of :class:`aeneas.timevalue.TimeValue`
        """
        self._log(u"Converting times to intervals...")
        intervals = [[times[i], times[i+1]] for i in range(len(times) - 1)]
        self._log(u"Converting times to intervals... done")
        self._log(u"Adding head and tail...")
        self.intervals = [[TimeValue("0.000"), intervals[0][0]]] + intervals + [[intervals[-1][1], self.real_wave_mfcc.audio_length]]
        self._log(u"Adding head and tail... done")

    def _apply_offset(self, offset):
        """
        Apply the given offset (negative, zero, or positive)
        to all times.

        :param offset: the offset, in seconds
        :type  offset: :class:`aeneas.timevalue.TimeValue`
        """
        times = (self.boundary_indices * self.rconf.mws) + offset
        if numpy.min(times) < TimeValue("0.000"):
            self._log(u"After applying offset some boundary times are negative", Logger.WARNING)
        if numpy.max(times) > self.real_wave_mfcc.audio_length:
            self._log(u"After applying offset some boundary times are beyond audio file duration", Logger.WARNING)
        times = numpy.clip(times, TimeValue("0.000"), self.real_wave_mfcc.audio_length)
        self._times_to_intervals(times)

    def _adjust_on_nonspeech(self, adjust_function):
        """
        Apply the adjust function to each boundary point
        falling inside (extrema included) of a nonspeech interval.

        The adjust function is not applied to a boundary index
        if there are two or more boundary indices falling
        inside the same nonspeech interval.

        The adjust function is not applied to the last boundary index
        to avoid anticipating the end of the audio file.

        The adjust function takes three arguments: the begin and end
        indices of the nonspeech interval, and the current boundary index.
        """
        self._log(u"Called _adjust_on_nonspeech")
        mws = self.rconf.mws
        nonspeech_intervals = self.real_wave_mfcc.intervals(speech=False, time=False)
        #
        # first iteration
        # nonspeech_counter[i] is the number of boundary indices
        # falling in the i-th nonspeech interval
        #
        self._log(u"  First iteration...")
        nonspeech_counter = numpy.zeros(len(nonspeech_intervals), dtype=int)
        i = 0 # index of current boundary_index
        j = 0 # index of current nonspeech_interval
        while i < len(self.boundary_indices):
            # current boundary index
            cbi = self.boundary_indices[i]
            # current nonspeech interval
            # with the property that it ends at an index >= cbi - 1
            while (j < len(nonspeech_intervals)) and (nonspeech_intervals[j][1] < cbi - 1):
                j += 1
            if j >= len(nonspeech_intervals):
                break
            cni = nonspeech_intervals[j]
            self._log([u"FI Current boundary index:     %d %.3f", cbi, cbi * mws])
            self._log([u"FI Current nonspeech interval: %d %d", cni[0], cni[1]])
            if (cbi - 1 >= cni[0]) and (cbi - 1 <= cni[1]):
                self._log(u"FI  Current boundary index is inside nonspeech")
                nonspeech_counter[j] += 1
            i += 1
        self._log(u"  First iteration... done")
        #
        # second iteration
        # we adjust the time value only for those boundary indices that
        # 1. fall within a nonspeech interval and,
        # 2. each is the only boundary index falling in that nonspeech interval
        # all the other boundary indices are returned unchanged
        #
        self._log(u"  Second iteration...")
        times = numpy.zeros(len(self.boundary_indices), dtype=TimeValue)
        i = 0
        j = 0
        while i < len(self.boundary_indices):
            # current boundary index
            cbi = self.boundary_indices[i]
            # current nonspeech interval
            # with the property that it ends at an index >= cbi - 1
            while (j < len(nonspeech_intervals)) and (nonspeech_intervals[j][1] < cbi - 1):
                j += 1
            if j >= len(nonspeech_intervals):
                break
            cni = nonspeech_intervals[j]
            self._log([u"SI Current boundary index:     %d %.3f", cbi, cbi * mws])
            self._log([u"SI Current nonspeech interval: %d %d", cni[0], cni[1]])
            if (
                    (cbi - 1 >= cni[0]) and
                    (cbi - 1 <= cni[1]) and
                    (nonspeech_counter[j] == 1) and (i < len(self.boundary_indices) - 1)
                ):
                # falling inside and unique and not last => adjust
                times[i] = adjust_function(cni[0], cni[1], cbi)
                self._log([u"SI  Adjusted cbi %d : %.3f => %.3f", cbi, cbi * mws, times[i]])
            else:
                # not falling inside or not unique or last => do not adjust
                times[i] = cbi * mws
                self._log([u"SI  Not adjusted cbi %d : %.3f => %.3f", cbi, times[i], times[i]])
            i += 1
        while i < len(self.boundary_indices):
            # complete with remaining indices
            cbi = self.boundary_indices[i]
            times[i] = cbi * mws
            self._log([u"Not adjusting %d %.3f", cbi, times[i]])
            i += 1
        self._log(u"  Second iteration... done")
        self._times_to_intervals(times)



