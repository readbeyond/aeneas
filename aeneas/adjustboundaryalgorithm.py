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

from aeneas.audiofilemfcc import AudioFileMFCC
from aeneas.exacttiming import Decimal
from aeneas.exacttiming import TimeInterval
from aeneas.exacttiming import TimeValue
from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.syncmap import SyncMapFragment
from aeneas.syncmap import SyncMapFragmentList
from aeneas.textfile import TextFile
from aeneas.textfile import TextFragment
from aeneas.tree import Tree


class AdjustBoundaryAlgorithm(Loggable):
    """
    Enumeration and implementation of the available algorithms
    to adjust the boundary point between two consecutive fragments.

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

    def __init__(self, rconf=None, logger=None):
        super(AdjustBoundaryAlgorithm, self).__init__(rconf=rconf, logger=logger)
        self.smflist = None
        self.mws = self.rconf.mws

    def adjust(
        self,
        aba_parameters,
        boundary_indices,
        real_wave_mfcc,
        text_file,
    ):
        """
        Adjust the boundaries of the text map
        using the algorithm and parameters
        specified in the constructor,
        storing the sync map fragment list internally.

        :param dict aba_parameters: a dictionary containing the algorithm and its parameters,
                                    as produced by ``aba_parameters()`` in ``TaskConfiguration``
        :param boundary_indices: the current boundary indices,
                                 with respect to the audio file full MFCCs
        :type  boundary_indices: :class:`numpy.ndarray` (1D)
        :param real_wave_mfcc: the audio file MFCCs
        :type  real_wave_mfcc: :class:`~aeneas.audiofilemfcc.AudioFileMFCC`
        :param text_file: the text file containing the text fragments associated
        :type  text_file: :class:`~aeneas.textfile.TextFile`

        :rtype: list of :class:`~aeneas.syncmap.SyncMapFragmentList`
        """
        if boundary_indices is None:
            raise TypeError(u"boundary_indices is None")
        if (real_wave_mfcc is None) or (not isinstance(real_wave_mfcc, AudioFileMFCC)):
            raise TypeError(u"real_wave_mfcc is None or not an AudioFileMFCC object")
        if (text_file is None) or (not isinstance(text_file, TextFile)):
            raise TypeError(u"text_file is None or not a TextFile object")

        # save mws
        self.mws = real_wave_mfcc.rconf.mws

        # convert boundary indices to time intervals
        self.intervals_to_fragment_list(
            text_file=text_file,
            end=real_wave_mfcc.audio_length,
            boundary_indices=boundary_indices,
        )

        # check no fragment has zero length, if requested
        check, offset = aba_parameters["nozero"]
        self._check_no_zero(check=check, offset=offset)

        # add silence intervals, if any
        ns_min, ns_string = aba_parameters["nonspeech"]
        self._process_long_nonspeech(ns_min, ns_string)

        # adjust using the right algorithm
        algorithm, algo_parameters = aba_parameters["algorithm"]
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
            ALGORITHM_MAP[algorithm](real_wave_mfcc, algo_parameters)
        else:
            self._adjust_auto(real_wave_mfcc, algo_parameters)

        # ensure the HEAD interval starts at 0.000
        self.smflist[0].begin = TimeValue("0.000")
        # ensure the TAIL interval ends at audio length
        self.smflist[-1].end = real_wave_mfcc.audio_length

        return self.smflist

    def intervals_to_fragment_list(self, text_file, end, boundary_indices=None, time_values=None):
        """
        Transform a list of boundary indices or time values
        into a sync map fragment list and store it internally.

        For example:

            b_i=[1, 2, 3], end=4.567 => [(0.000, 1*mws), (1*mws, 2*mws), (2*mws, 3*mws), (3*mws, 4.567)]
            time_values=[0.000, 1.000, 2.000, 3.456] => [(0.000, 1.000), (1.000, 2.000), (2.000, 3.456)]

        :param text_file: the text file containing the text fragments associated
        :type  text_file: :class:`~aeneas.textfile.TextFile`
        :param time_values: the time values
        :type  time_values: list of :class:`~aeneas.exacttiming.TimeValue`
        """
        if (boundary_indices is None) and (time_values is None):
            self.log_exc(u"Both boundary_indices and times are None", None, True, TypeError)
        if boundary_indices is not None:
            self.log(u"Converting boundary indices to fragment list...")
            times = [TimeValue("0.000")] + list(boundary_indices * self.mws) + [end]
        else:
            self.log(u"Converting time intervals to fragment list...")
            times = time_values

        self.smflist = SyncMapFragmentList(begin=TimeValue("0.000"), end=end)
        # HEAD
        self.smflist.add(SyncMapFragment(
            text_fragment=TextFragment(identifier=u"HEAD"),
            begin=times[0],
            end=times[1],
            fragment_type=SyncMapFragment.HEAD
        ), sort=False)
        # REGULAR
        for i in range(1, len(times) - 2):
            self.smflist.add(SyncMapFragment(
                text_fragment=text_file.fragments[i - 1],
                begin=times[i],
                end=times[i + 1],
                fragment_type=SyncMapFragment.REGULAR
            ), sort=False)
        # TAIL
        self.smflist.add(SyncMapFragment(
            text_fragment=TextFragment(identifier=u"TAIL"),
            begin=times[len(times) - 2],
            end=end,
            fragment_type=SyncMapFragment.TAIL
        ), sort=False)
        self.log(u"Converting to fragment list... done")
        self.log(u"Sorting fragment list...")
        self.smflist.sort()
        self.log(u"Sorting fragment list... done")
        return self.smflist

    def append_fragment_list_to_sync_root(self, sync_root):
        """
        Append the sync map fragment list
        to the given node from a sync map tree.

        :param sync_root: the root of the sync map tree to which the new nodes should be appended
        :type  sync_root: :class:`~aeneas.tree.Tree`
        """
        if (sync_root is None) or (not isinstance(sync_root, Tree)):
            raise TypeError(u"sync_root is None or not a Tree object")

        self.log(u"Appending fragment list to sync root...")
        for fragment in self.smflist:
            sync_root.add_child(Tree(value=fragment))
        self.log(u"Appending fragment list to sync root... done")

    # #####################################################
    # NO ZERO AND LONG NONSPEECH FUNCTIONS
    # #####################################################

    def _check_no_zero(self, check, offset):
        self.log(u"Called _check_no_zero")
        if check:
            self.log(u"Check requested: checking and fixing")
            self.log([u"Offset is %.3f", offset])
            # ignore HEAD and TAIL
            max_index = len(self.smflist) - 1
            self.smflist.fix_zero_length_intervals(
                offset=offset,
                min_index=1,
                max_index=max_index
            )
            self.smflist[max_index].begin = self.smflist[max_index - 1].end
        else:
            self.log(u"Check not requested: returning")

    def _process_long_nonspeech(self, ns_min, ns_string):
        self.log(u"Called _process_long_nonspeech")
        if ns_min is not None:
            self.log(u"Processing long nonspeech intervals requested: fixing")
            # TODO
        else:
            self.log(u"Processing long nonspeech intervals not requested: returning")

    # #####################################################
    # ADJUST FUNCTIONS
    # #####################################################

    def _adjust_auto(self, real_wave_mfcc, algo_parameters):
        """
        AUTO (do not modify)
        """
        self.log(u"Called _adjust_auto")
        self.log(u"Nothing to do, return unchanged")

    def _adjust_offset(self, real_wave_mfcc, algo_parameters):
        """
        OFFSET
        """
        self.log(u"Called _adjust_offset")
        self._apply_offset(offset=algo_parameters[0])

    def _adjust_percent(self, real_wave_mfcc, algo_parameters):
        """
        PERCENT
        """
        def new_time(nsi):
            """
            The new boundary time value is ``percent``
            of the nonspeech interval ``nsi``.
            """
            percent = Decimal(algo_parameters[0])
            return nsi.percent_value(percent)
        self.log(u"Called _adjust_percent")
        self._adjust_on_nonspeech(real_wave_mfcc, new_time)

    def _adjust_aftercurrent(self, real_wave_mfcc, algo_parameters):
        """
        AFTERCURRENT
        """
        def new_time(nsi):
            """
            The new boundary time value is ``delay`` after
            the begin of the nonspeech interval ``nsi``.
            If ``nsi`` has length less than ``delay``,
            set the new boundary time to the end of ``nsi``.
            """
            delay = max(algo_parameters[0], TimeValue("0.000"))
            return min(nsi.begin + delay, nsi.end)
        self.log(u"Called _adjust_aftercurrent")
        self._adjust_on_nonspeech(real_wave_mfcc, new_time)

    def _adjust_beforenext(self, real_wave_mfcc, algo_parameters):
        """
        BEFORENEXT
        """
        def new_time(nsi):
            """
            The new boundary time value is ``delay`` before
            the end of the nonspeech interval ``nsi``.
            If ``nsi`` has length less than ``delay``,
            set the new boundary time to the begin of ``nsi``.
            """
            delay = max(algo_parameters[0], TimeValue("0.000"))
            return max(nsi.end - delay, nsi.begin)
        self.log(u"Called _adjust_beforenext")
        self._adjust_on_nonspeech(real_wave_mfcc, new_time)

    def _adjust_rate(self, real_wave_mfcc, algo_parameters):
        self.log(u"Called _adjust_rate")
        self._apply_rate(real_wave_mfcc, max_rate=algo_parameters[0], aggressive=False)

    def _adjust_rate_aggressive(self, real_wave_mfcc, algo_parameters):
        self.log(u"Called _adjust_rate_aggressive")
        self._apply_rate(real_wave_mfcc, max_rate=algo_parameters[0], aggressive=True)

    # #####################################################
    # HELPER FUNCTIONS
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
        self.log([u"Applying offset %s", offset])
        self.smflist.offset(offset)

    def _adjust_on_nonspeech(self, real_wave_mfcc, adjust_function):
        """
        Apply the adjust function to each boundary point
        falling inside (extrema included) of a nonspeech interval.

        The adjust function is not applied to a boundary index
        if there are two or more boundary indices falling
        inside the same nonspeech interval.

        The adjust function is not applied to the last boundary index
        to avoid anticipating the end of the audio file.

        The ``adjust function`` takes
        the nonspeech interval as its only argument.
        """
        self.log(u"Called _adjust_on_nonspeech")
        self.log(u"  Getting nonspeech intervals...")
        nonspeech_intervals = real_wave_mfcc.intervals(speech=False, time=True)
        self.log(u"  Getting nonspeech intervals... done")

        self.log(u"  First pass: associate each fragment end point to an nsi, if possible")
        tolerance = self.rconf[RuntimeConfiguration.ABA_NONSPEECH_TOLERANCE]
        nsi_index = 0
        frag_index = 0
        nsi_counter = [(n, []) for n in nonspeech_intervals]
        while (nsi_index < len(nonspeech_intervals)) and (frag_index < len(self.smflist)):
            nsi = nonspeech_intervals[nsi_index]
            nsi_shadow = nsi.shadow(tolerance)
            frag = self.smflist[frag_index]
            self.log([u"    nsi:        %s", nsi])
            self.log([u"    nsi shadow: %s", nsi_shadow])
            if frag.fragment_type == SyncMapFragment.REGULAR:
                frag_end = frag.end
                self.log(u"      REGULAR => examining it")
                self.log([u"        %.3f vs %s", frag_end, nsi_shadow])
                if nsi_shadow.contains(frag_end):
                    #
                    #      *************** nsi shadow
                    #      | *********** | nsi
                    # *****|***X         | frag (X=frag_end)
                    #
                    self.log(u"        Contained => register and go to next fragment")
                    nsi_counter[nsi_index][1].append(frag_index)
                    frag_index += 1
                elif nsi_shadow.begin > frag_end:
                    #
                    #      *************** nsi shadow
                    #      | *********** | nsi
                    # **X  |             | frag (X=frag_end)
                    #
                    self.log(u"        Before => go to next fragment")
                    frag_index += 1
                else:
                    #
                    #       ***************    nsi shadow
                    #       | *********** |    nsi
                    #       |        *****|**X frag (X=frag_end)
                    #
                    self.log(u"        After => go to next nsi")
                    nsi_index += 1
            else:
                self.log(u"      Not REGULAR => go to next fragment")
                frag_index += 1
        self.log(u"  First pass: done")

        self.log(u"  Second pass: move end point on good nsi")
        for nsi, frags in nsi_counter:
            self.log([u"    Examining nsi %s", nsi])
            if len(frags) == 1:
                frag_index = frags[0]
                new_value = adjust_function(nsi)
                self.log([u"      Only one index:   %d", frag_index])
                self.log([u"      New value:        %.3f", new_value])
                self.log([u"      Current interval: %s", self.smflist[frag_index].interval])
                self.smflist.move_end(index=frags[0], value=new_value)
                self.log([u"      New interval:     %s", self.smflist[frag_index].interval])
            else:
                self.log(u"      Skip it")
        self.log(u"  Second pass: done")

    def _apply_rate(self, real_wave_mfcc, max_rate, aggressive=False):
        self.log(u"Called _apply_rate")
        # if only one fragment, return unchanged
        if len(self.smflist) <= 3:
            self.log(u"Only one text fragment, returning")
            return
        # TODO
