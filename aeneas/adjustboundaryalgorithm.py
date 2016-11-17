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

* :class:`~aeneas.adjustboundaryalgorithm.AdjustBoundaryAlgorithm`
  implementing functions to adjust
  the boundary point between two consecutive fragments.

.. warning:: This module is likely to be refactored in a future version
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy

from aeneas.audiofilemfcc import AudioFileMFCC
from aeneas.exacttiming import Decimal
from aeneas.exacttiming import TimeInterval
from aeneas.exacttiming import TimeIntervalList
from aeneas.exacttiming import TimeValue
from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.syncmap import SyncMapFragment
from aeneas.textfile import TextFile
from aeneas.textfile import TextFragment
from aeneas.tree import Tree


class AdjustBoundaryAlgorithm(Loggable):
    """
    Enumeration and implementation of the available algorithms
    to adjust the boundary point between two consecutive fragments.

    :param dict aba_parameters: a dictionary containing the algorithm and its parameters,
                                as produced by ``aba_parameters()`` in ``TaskConfiguration``
    :param boundary_indices: the current boundary indices,
                             with respect to the audio file full MFCCs
    :type  boundary_indices: :class:`numpy.ndarray` (1D)
    :param real_wave_mfcc: the audio file MFCCs
    :type  real_wave_mfcc: :class:`~aeneas.audiofilemfcc.AudioFileMFCC`
    :param text_file: the text file containing the text fragments associated
    :type  text_file: :class:`~aeneas.textfile.TextFile`
    :param sync_root: the root of the sync map tree to which new nodes should be appended
    :type  sync_root: :class:`~aeneas.tree.Tree`
    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    :raises: TypeError: if one of ``boundary_indices``, ``real_wave_mfcc``,
                        or ``text_file`` is ``None`` or it has a wrong type
    """

    AFTERCURRENT = "aftercurrent"
    """
    Set the boundary at ``value`` seconds
    after the end of the current fragment,
    if the current boundary falls inside
    a nonspeech interval.
    If not, no adjustment is made.

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
    before the beginning of the next fragment,
    if the current boundary falls inside
    a nonspeech interval.
    If not, no adjustment is made.

    Example (value ``0.200`` seconds):

    .. image:: _static/beforenext.200.png
       :scale: 100%
       :align: center
       :alt: Comparison between AUTO labels and BEFORENEXT labels with 0.200 seconds offset
    """

    OFFSET = "offset"
    """
    Offset the current boundaries by ``value`` seconds.
    The ``value`` can be negative or positive.

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
    the nonspeech interval between the current and the next fragment,
    if the current boundary falls inside
    a nonspeech interval.
    The ``value`` must be an integer in ``[0, 100]``.
    If not, no adjustment is made.

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
    The ``value`` must be positive.
    First, the rates of all fragments are computed,
    using the current boundaries.
    For those fragments exceeding ``value`` characters/second,
    the algorithm will try to move the end boundary forward,
    so that its time interval increases (and hence its rate decreases).
    Clearly, it is possible that not all fragments
    can be adjusted this way: for example,
    if you have three consecutive fragments exceeding ``value``,
    the middle one cannot be stretched.

    Example (value ``13.0``, note how ``f000003`` is modified):

    .. image:: _static/rate.13.png
       :scale: 100%
       :align: center
       :alt: Comparison between AUTO labels and RATE labels with value 13.0

    """

    RATEAGGRESSIVE = "rateaggressive"
    """
    Adjust boundaries trying to respect the
    ``value`` characters/second constraint, in aggressive mode.
    The ``value`` must be positive.
    First, the rates of all fragments are computed,
    using the current boundaries.
    For those fragments exceeding ``value`` characters/second,
    the algorithm will try to move the end boundary forward,
    so that its time interval increases (and hence its rate decreases).
    If moving the end boundary is not possible,
    or it is not enough to keep the rate below ``value``,
    the algorithm will try to move the begin boundary back;
    this is the difference with the less aggressive
    :data:`~aeneas.adjustboundaryalgorithm.AdjustBoundaryAlgorithm.RATE`
    algorithm.
    Clearly, it is possible that not all fragments
    can be adjusted this way: for example,
    if you have three consecutive fragments exceeding ``value``,
    the middle one cannot be stretched.

    Example (value ``13.0``, note how ``f000003`` is modified):

    .. image:: _static/rateaggressive.13.png
       :scale: 100%
       :align: center
       :alt: Comparison between AUTO labels and RATEAGGRESSIVE labels with value 13.0

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
            aba_parameters,
            boundary_indices,
            real_wave_mfcc,
            text_file,
            sync_root,
            rconf=None,
            logger=None
    ):
        if boundary_indices is None:
            raise TypeError(u"boundary_indices is None")
        if (real_wave_mfcc is None) or (not isinstance(real_wave_mfcc, AudioFileMFCC)):
            raise TypeError(u"real_wave_mfcc is None or not an AudioFileMFCC object")
        if (text_file is None) or (not isinstance(text_file, TextFile)):
            raise TypeError(u"text_file is None or not a TextFile object")
        if (sync_root is None) or (not isinstance(sync_root, Tree)):
            raise TypeError(u"sync_root is None or not a Tree object")
        super(AdjustBoundaryAlgorithm, self).__init__(rconf=rconf, logger=logger)
        self.aba_parameters = aba_parameters
        self.real_wave_mfcc = real_wave_mfcc
        self.boundary_indices = boundary_indices
        self.text_file = text_file
        self.sync_root = sync_root
        self.intervals = None

    def adjust(self):
        """
        Adjust the boundaries of the text map
        using the algorithm and parameters
        specified in the constructor,
        and return a list of time intervals.

        :rtype: list of :class:`~aeneas.exacttiming.TimeInterval`
        """
        # convert boundary indices to time intervals
        self._boundary_indices_to_intervals()

        # check no fragment has zero length, if requested
        self._check_no_zero()

        # add silence intervals, if any
        self._process_long_silences()

        algorithm = self.aba_parameters["algorithm"][0]
        ALGORITHM_MAP = {
            self.AFTERCURRENT: self._adjust_aftercurrent,
            self.AUTO: self._adjust_auto,
            self.BEFORENEXT: self._adjust_beforenext,
            self.OFFSET: self._adjust_offset,
            self.PERCENT: self._adjust_percent,
            self.RATE: self._adjust_rate,
            self.RATEAGGRESSIVE: self._adjust_rate_aggressive,
        }
        if algorithm in ALGORITHM_MAP:
            ALGORITHM_MAP[algorithm]()
        else:
            self._adjust_auto()

        # ensure the HEAD interval starts at 0.000
        self.intervals[0].begin = TimeValue("0.000")
        # ensure the TAIL interval ends at audio length
        self.intervals[-1].end = self.real_wave_mfcc.audio_length

        # append new nodes to sync_root
        self._intervals_to_tree()

    def _intervals_to_tree(self):
        self.log(u"Converting intervals to tree...")
        fragments = (
            [TextFragment(identifier=u"HEAD", language=None, lines=[u""])] +
            self.text_file.fragments +
            [TextFragment(identifier=u"TAIL", language=None, lines=[u""])]
        )
        for interval, fragment in zip(self.intervals, fragments):
            sm_frag = SyncMapFragment(
                text_fragment=fragment,
                begin=interval.begin,
                end=interval.end,
                fragment_type=interval.interval_type
            )
            self.sync_root.add_child(Tree(value=sm_frag))
        self.log(u"Converting intervals to tree... done")

    def _boundary_indices_to_intervals(self):
        """
        Transform a list of time values into a list of intervals,
        and store it internally.

        For example: [0,1,2,3,4] => [(0,1), (1,2), (2,3), (3,4)]

        :param times: the time values
        :type  times: list of :class:`~aeneas.exacttiming.TimeIntervals`
        """
        self.log(u"Converting boundary indices to intervals...")
        self.intervals = TimeIntervalList(
            begin=TimeValue("0.000"),
            end=self.real_wave_mfcc.audio_length
        )
        times = self.boundary_indices * self.rconf.mws
        self.intervals.add(TimeInterval(
            begin=TimeValue("0.000"),
            end=times[0],
            interval_type=TimeInterval.HEAD
        ))
        for i in range(len(times) - 1):
            self.intervals.add(TimeInterval(
                begin=times[i],
                end=times[i + 1],
                interval_type=TimeInterval.REGULAR
            ))
        self.intervals.add(TimeInterval(
            begin=times[len(times) - 1],
            end=self.real_wave_mfcc.audio_length,
            interval_type=TimeInterval.TAIL
        ))
        self.log(u"Converting boundary indices to intervals... done")

    def _adjust_auto(self):
        """
        AUTO (do not modify)
        """
        self.log(u"Called _adjust_auto")
        self.log(u"Nothing to do, return unchanged")

    def _adjust_offset(self):
        """
        OFFSET
        """
        self.log(u"Called _adjust_offset")
        # NOTE parameter is a TimeValue
        offset = self.aba_parameters["algorithm"][1][0]
        self._apply_offset(offset)

    def _adjust_percent(self):
        """
        PERCENT
        """
        def new_time(begin, end, current):
            """ Compute new time """
            # NOTE parameter is an int
            percent = self.aba_parameters["algorithm"][1][0]
            percent = max(min(Decimal(percent) / 100, 100), 0)
            return (begin + (end + 1 - begin) * percent) * self.rconf.mws
        self.log(u"Called _adjust_percent")
        self._adjust_on_nonspeech(new_time)

    def _adjust_aftercurrent(self):
        """
        AFTERCURRENT
        """
        def new_time(begin, end, current):
            """ Compute new time """
            mws = self.rconf.mws
            # NOTE parameter is a TimeValue
            delay = self.aba_parameters["algorithm"][1][0]
            delay = max(delay, TimeValue("0.000"))
            tentative = begin * mws + delay
            if tentative > (end + 1) * mws:
                return current * mws
            return tentative
        self.log(u"Called _adjust_aftercurrent")
        self._adjust_on_nonspeech(new_time)

    def _adjust_beforenext(self):
        """
        BEFORENEXT
        """
        def new_time(begin, end, current):
            """ Compute new time """
            mws = self.rconf.mws
            # NOTE parameter is a TimeValue
            delay = self.aba_parameters["algorithm"][1][0]
            delay = max(delay, TimeValue("0.000"))
            tentative = (end + 1) * mws - delay
            if tentative < begin * mws:
                return current * mws
            return tentative
        self.log(u"Called _adjust_beforenext")
        self._adjust_on_nonspeech(new_time)

    def _adjust_rate(self):
        self.log(u"Called _adjust_rate")
        self._apply_rate(aggressive=False)

    def _adjust_rate_aggressive(self):
        self.log(u"Called _adjust_rate_aggressive")
        self._apply_rate(aggressive=True)

    # #####################################################
    # HELPERS
    # #####################################################

    def _apply_offset(self, offset):
        """
        Apply the given offset (negative, zero, or positive)
        to all time intervals.

        :param offset: the offset, in seconds
        :type  offset: :class:`~aeneas.exacttiming.TimeValue`
        """
        if not isinstance(offset, TimeValue):
            self.log_exc(u"offset is not an instance of TimeValue", None, True, TypeError)
        self.log([u"Applying offset %s", self.parameters[0]])
        self.intervals.offset(offset)

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
        self.log(u"Called _adjust_on_nonspeech")
        # TODO

    def _apply_rate(self, aggressive=False):
        self.log(u"Called _apply_rate")
        # if only one fragment, return unchanged
        if len(self.text_file) <= 1:
            self.log(u"Only one fragment, returning")
            return
        # TODO

    def _check_no_zero(self):
        self.log(u"Called _check_no_zero")
        if self.aba_parameters["nozero"][0]:
            self.log(u"Check requested: checking and fixing")
            offset = self.aba_parameters["nozero"][1]
            self.log([u"Offset is %.3f", offset])
            # ignore HEAD and TAIL
            max_index = len(self.intervals) - 1
            self.intervals.fix_zero_length_intervals(
                offset=offset,
                min_index=1,
                max_index=max_index
            )
            self.intervals[max_index].begin = self.intervals[max_index - 1].end
        else:
            self.log(u"Check not requested: returning")

    def _process_long_silences(self):
        self.log(u"Called _process_long_silences")
        sil_min, sil_string = self.aba_parameters["silence"]
        if sil_min is not None:
            self.log(u"Processing long silences requested: fixing")
            # TODO
        else:
            self.log(u"Processing long silences not requested: returning")
