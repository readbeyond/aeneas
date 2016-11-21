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

from __future__ import absolute_import
from __future__ import print_function
import bisect

from aeneas.exacttiming import TimeInterval
from aeneas.exacttiming import TimeValue
from aeneas.logger import Loggable
from aeneas.syncmap.fragment import SyncMapFragment
from aeneas.textfile import TextFragment
import aeneas.globalconstants as gc


class SyncMapFragmentList(Loggable):
    """
    A type representing a list of sync map fragments,
    with some constraints:

    * the begin and end time of each fragment should be within the list begin and end times;
    * two time fragments can only overlap at the boundary;
    * the list is kept sorted.

    This class has some convenience methods for
    clipping, offsetting, moving fragment boundaries,
    and fixing fragments with zero length.

    .. versionadded:: 1.7.0

    :param begin: the begin time
    :type  begin: :class:`~aeneas.exacttiming.TimeValue`
    :param end: the end time
    :type  end: :class:`~aeneas.exacttiming.TimeValue`
    :raises TypeError: if ``begin`` or ``end`` are not instances of :class:`~aeneas.exacttiming.TimeValue`
    :raises ValueError: if ``begin`` is negative or if ``begin`` is bigger than ``end``

    .. versionadded:: 1.7.0
    """

    ALLOWED_POSITIONS = [
        TimeInterval.RELATIVE_POSITION_PP_L,
        TimeInterval.RELATIVE_POSITION_PP_C,
        TimeInterval.RELATIVE_POSITION_PP_G,
        TimeInterval.RELATIVE_POSITION_PI_LL,
        TimeInterval.RELATIVE_POSITION_PI_LC,
        TimeInterval.RELATIVE_POSITION_PI_CG,
        TimeInterval.RELATIVE_POSITION_PI_GG,
        TimeInterval.RELATIVE_POSITION_IP_L,
        TimeInterval.RELATIVE_POSITION_IP_B,
        TimeInterval.RELATIVE_POSITION_IP_E,
        TimeInterval.RELATIVE_POSITION_IP_G,
        TimeInterval.RELATIVE_POSITION_II_LL,
        TimeInterval.RELATIVE_POSITION_II_LB,
        TimeInterval.RELATIVE_POSITION_II_EG,
        TimeInterval.RELATIVE_POSITION_II_GG,
    ]
    """ Allowed positions for any pair of time intervals in the list """

    TAG = u"SyncMapFragmentList"

    def __init__(self, begin, end, rconf=None, logger=None):
        if not isinstance(begin, TimeValue):
            raise TypeError(u"begin is not an instance of TimeValue")
        if not isinstance(end, TimeValue):
            raise TypeError(u"end is not an instance of TimeValue")
        if begin < 0:
            raise ValueError(u"begin is negative")
        if begin > end:
            raise ValueError(u"begin is bigger than end")
        super(SyncMapFragmentList, self).__init__(rconf=rconf, logger=logger)
        self.begin = begin
        self.end = end
        self.__sorted = True
        self.__fragments = []

    def __len__(self):
        return len(self.__fragments)

    def __getitem__(self, index):
        return self.__fragments[index]

    def __setitem__(self, index, value):
        self.__fragments[index] = value

    def _check_boundaries(self, fragment):
        """
        Check that the interval of the given fragment
        is within the boundaries of the list.
        Raises an error if not OK.
        """
        if not isinstance(fragment, SyncMapFragment):
            raise TypeError(u"fragment is not an instance of SyncMapFragment")
        interval = fragment.interval
        if not isinstance(interval, TimeInterval):
            raise TypeError(u"interval is not an instance of TimeInterval")
        if (self.begin is not None) and (interval.begin < self.begin):
            raise ValueError(u"interval.begin is before self.begin")
        if (self.end is not None) and (interval.end > self.end):
            raise ValueError(u"interval.end is after self.end")

    def _check_overlap(self, fragment):
        """
        Check that the interval of the given fragment does not overlap
        any existing interval in the list (except at its boundaries).
        Raises an error if not OK.
        """
        #
        # NOTE bisect does not work if there is a configuration like:
        #
        #      *********** <- existing interval
        #           ***    <- query interval
        #
        # TODO one should probably check this by doing bisect
        #      over the begin and end lists separately
        #
        for existing_fragment in self.fragments:
            if existing_fragment.interval.relative_position_of(fragment.interval) not in self.ALLOWED_POSITIONS:
                self.log_exc(u"interval overlaps another already present interval", None, True, ValueError)

    def _check_min_max_indices(self, min_index=None, max_index=None):
        """
        Ensure the given start/end fragment indices make sense:
        if one of them is ``None`` (i.e., not specified),
        then set it to ``0`` or ``len(self)``.
        """
        min_index = min_index or 0
        max_index = max_index or len(self)
        if min_index < 0:
            self.log_exc(u"min_index is negative", None, True, ValueError)
        if max_index > len(self):
            self.log_exc(u"max_index is bigger than the number of intervals in the list", None, True, ValueError)
        return min_index, max_index

    @property
    def is_guaranteed_sorted(self):
        """
        Return ``True`` if the list is sorted,
        and ``False`` if it might not be sorted
        (for example, because an ``add(..., sort=False)`` operation
        was performed).

        :rtype: bool
        """
        return self.__sorted

    @property
    def fragments(self):
        """
        Iterates through the fragments in the list
        (which are sorted).

        :rtype: generator of :class:`~aeneas.syncmap.SyncMapFragment`
        """
        for fragment in self.__fragments:
            yield fragment

    def sort(self):
        """
        Sort the fragments in the list.

        :raises ValueError: if there is a fragment which violates
                            the list constraints
        """
        if self.is_guaranteed_sorted:
            self.log(u"Already sorted, returning")
            return
        self.log(u"Sorting...")
        self.__fragments = sorted(self.__fragments)
        self.log(u"Sorting... done")
        self.log(u"Checking relative positions...")
        for i in range(len(self) - 1):
            if self[i].interval.relative_position_of(self[i + 1].interval) not in self.ALLOWED_POSITIONS:
                self.log(u"Found overlapping fragments:")
                self.log([u"  Index %d => %s", i, self[i].interval])
                self.log([u"  Index %d => %s", i + 1, self[i + 1].interval])
                self.log_exc(u"The list contains two fragments overlapping in a forbidden way", None, True, ValueError)
        self.log(u"Checking relative positions... done")
        self.__sorted = True

    def has_zero_length_fragments(self, min_index=None, max_index=None):
        """
        Return ``True`` if the list has at least one interval
        with zero length withing ``min_index`` and ``max_index``.
        If the latter are not specified, check all intervals.

        :param int min_index: examine fragments with index greater than or equal to this index (i.e., included)
        :param int max_index: examine fragments with index smaller than this index (i.e., excluded)
        :raises ValueError: if ``min_index`` is negative or ``max_index``
                            is bigger than the current number of fragments
        :rtype: bool
        """
        min_index, max_index = self._check_min_max_indices(min_index, max_index)
        zero = [i for i in range(min_index, max_index) if self[i].interval.has_zero_length]
        self.log([u"Fragments with zero length: %s", zero])
        return (len(zero) > 0)

    def has_adjacent_fragments_only(self, min_index=None, max_index=None):
        """
        Return ``True`` if the list contains only adjacent fragments,
        that is, if it does not have gaps.

        :param int min_index: examine fragments with index greater than or equal to this index (i.e., included)
        :param int max_index: examine fragments with index smaller than this index (i.e., excluded)
        :raises ValueError: if ``min_index`` is negative or ``max_index``
                            is bigger than the current number of fragments
        :rtype: bool
        """
        min_index, max_index = self._check_min_max_indices(min_index, max_index)
        for i in range(min_index, max_index - 1):
            if not self[i].interval.is_adjacent_before(self[i + 1].interval):
                self.log(u"Found non adjacent fragments")
                self.log([u"  Index %d => %s", i, self[i].interval])
                self.log([u"  Index %d => %s", i + 1, self[i + 1].interval])
                return False
        return True

    def add(self, fragment, sort=True):
        """
        Add the given fragment to the list (and keep the latter sorted).

        An error is raised if the fragment cannot be added,
        for example if its interval violates the list constraints.

        :param fragment: the fragment to be added
        :type  fragment: :class:`~aeneas.syncmap.SyncMapFragment`
        :param bool sort: if ``True`` ensure that after the insertion the list is kept sorted
        :raises TypeError: if ``interval`` is not an instance of ``TimeInterval``
        :raises ValueError: if ``interval`` does not respect the boundaries of the list
                            or if it overlaps an existing interval,
                            or if ``sort=True`` but the list is not guaranteed sorted
        """
        self._check_boundaries(fragment)
        if sort:
            if not self.is_guaranteed_sorted:
                self.log_exc(u"Unable to add with sort=True if the list is not guaranteed sorted", None, True, ValueError)
            self._check_overlap(fragment)
            bisect.insort(self.__fragments, fragment)
            # self.log(u"Inserted and kept sorted flag true")
        else:
            self.__fragments.append(fragment)
            self.__sorted = False
            # self.log(u"Appended at the end and invalidated sorted flag")

    def offset(self, offset):
        """
        Move all the intervals in the list by the given ``offset``.

        :param offset: the shift to be applied
        :type  offset: :class:`~aeneas.exacttiming.TimeValue`
        :raises TypeError: if ``offset`` is not an instance of ``TimeValue``
        """
        self.log(u"Applying offset to all fragments...")
        self.log([u"  Offset %.3f", offset])
        for fragment in self.fragments:
            fragment.interval.offset(
                offset=offset,
                allow_negative=False,
                min_begin_value=self.begin,
                max_end_value=self.end
            )
        self.log(u"Applying offset to all fragments... done")

    def move_transition_point(self, fragment_index, value):
        """
        Change the transition point between fragment ``fragment_index``
        and the next fragment to the time value ``value``.

        This method fails silently
        (without changing the fragment list)
        if at least one of the following conditions holds:

        * ``fragment_index`` is negative
        * ``fragment_index`` is the last or the second-to-last
        * ``value`` is after the current end of the next fragment
        * the current fragment and the next one are not adjacent and both proper intervals (not zero length)

        The above conditions ensure that the move makes sense
        and that it keeps the list satisfying the constraints.

        :param int fragment_index: the fragment index whose end should be moved
        :param value: the new transition point
        :type  value: :class:`~aeneas.exacttiming.TimeValue`
        """
        self.log(u"Called move_transition_point with")
        self.log([u"  fragment_index %d", fragment_index])
        self.log([u"  value          %.3f", value])
        if (fragment_index < 0) or (fragment_index > (len(self) - 3)):
            self.log(u"Bad fragment_index, returning")
            return
        current_interval = self[fragment_index].interval
        next_interval = self[fragment_index + 1].interval
        if value > next_interval.end:
            self.log(u"Bad value, returning")
            return
        if not current_interval.is_non_zero_before_non_zero(next_interval):
            self.log(u"Bad interval configuration, returning")
            return
        current_interval.end = value
        next_interval.begin = value
        self.log(u"Moved transition point")

    def fragments_ending_inside_nonspeech_intervals(
        self,
        nonspeech_intervals,
        tolerance
    ):
        """
        Determine a list of pairs (nonspeech interval, fragment index),
        such that the nonspeech interval contains exactly one fragment
        ending inside it (within the given tolerance) and
        adjacent to the next fragment.

        :param nonspeech_intervals: the list of nonspeech intervals to be examined
        :type  nonspeech_intervals: list of :class:`~aeneas.exacttiming.TimeInterval`
        :param tolerance: the tolerance to be applied when checking if the end point
                          falls within a given nonspeech interval
        :type  tolerance: :class:`~aeneas.exacttiming.TimeValue`
        :rtype: list of (:class:`~aeneas.exacttiming.TimeInterval`, int)
        """
        self.log(u"Called fragments_ending_inside_nonspeech_intervals")
        nsi_index = 0
        frag_index = 0
        nsi_counter = [(n, []) for n in nonspeech_intervals]
        # NOTE the last fragment is not eligible to be returned
        while (nsi_index < len(nonspeech_intervals)) and (frag_index < len(self) - 1):
            nsi = nonspeech_intervals[nsi_index]
            nsi_shadow = nsi.shadow(tolerance)
            frag = self[frag_index]
            self.log([u"  nsi        %s", nsi])
            self.log([u"  nsi_shadow %s", nsi_shadow])
            self.log([u"  frag       %s", frag.interval])
            if frag.fragment_type in [SyncMapFragment.REGULAR, SyncMapFragment.NONSPEECH]:
                self.log(u"    Fragment is REGULAR or NONSPEECH => inspecting it")
                if nsi_shadow.contains(frag.end):
                    #
                    #      *************** nsi shadow
                    #      | *********** | nsi
                    # *****|***X         | frag (X=frag.end)
                    #
                    nsi_counter[nsi_index][1].append(frag_index)
                    frag_index += 1
                    self.log(u"    nsi_shadow contains frag end => save it and go to next fragment")
                elif nsi_shadow.begin > frag.end:
                    #
                    #      *************** nsi shadow
                    #      | *********** | nsi
                    # **X  |             | frag (X=frag.end)
                    #
                    frag_index += 1
                    self.log(u"    nsi_shadow begins after frag end => skip to next fragment")
                else:
                    #
                    #       ***************    nsi shadow
                    #       | *********** |    nsi
                    #       |        *****|**X frag (X=frag.end)
                    #
                    nsi_index += 1
                    self.log(u"    nsi_shadow ends before frag end => skip to next nsi")
            else:
                self.log(u"    Fragment is HEAD or TAIL => skipping it")
                frag_index += 1
            self.log(u"")
        tbr = [(n, c[0]) for (n, c) in nsi_counter if len(c) == 1]
        self.log([u"Returning: %s", tbr])
        return tbr

    def inject_long_nonspeech_fragments(self, pairs, replacement_string):
        """
        Inject nonspeech fragments corresponding to the given intervals
        in this fragment list.

        It is assumed that ``pairs`` are consistent, e.g. they are produced
        by ``fragments_ending_inside_nonspeech_intervals``.

        :param list pairs: list of ``(TimeInterval, int)`` pairs,
                           each identifying a nonspeech interval and
                           the corresponding fragment index ending inside it
        :param string replacement_string: the string to be applied to the nonspeech intervals
        """
        self.log(u"Called inject_long_nonspeech_fragments")
        # set the appropriate type and text
        if replacement_string in [None, gc.PPV_TASK_ADJUST_BOUNDARY_NONSPEECH_REMOVE]:
            self.log(u"  Remove long nonspeech")
            fragment_type = SyncMapFragment.NONSPEECH
            lines = []
        else:
            self.log([u"  Replace long nonspeech with '%s'", replacement_string])
            fragment_type = SyncMapFragment.REGULAR
            lines = [replacement_string]
        # first, make room for the nonspeech intervals
        self.log(u"  First pass: making room...")
        for nsi, index in pairs:
            self[index].interval.end = nsi.begin
            self[index + 1].interval.begin = nsi.end
        self.log(u"  First pass: making room... done")
        self.log(u"  Second pass: append nonspeech intervals...")
        for i, (nsi, index) in enumerate(pairs, 1):
            identifier = u"n%06d" % i
            self.add(SyncMapFragment(
                text_fragment=TextFragment(
                    identifier=identifier,
                    language=None,
                    lines=lines,
                    filtered_lines=lines
                ),
                interval=nsi,
                fragment_type=fragment_type
            ), sort=False)
        self.log(u"  Second pass: append nonspeech intervals... done")
        self.log(u"  Third pass: sorting...")
        self.sort()
        self.log(u"  Third pass: sorting... done")

    def fix_zero_length_fragments(self, duration=TimeValue("0.001"), min_index=None, max_index=None):
        """
        Fix fragments with zero length,
        enlarging them to have length ``duration``,
        reclaiming the difference from the next fragment(s),
        or moving the next fragment(s) forward.

        This function assumes the fragments to be adjacent.

        :param duration: set the zero length fragments to have this duration
        :type  duration: :class:`~aeneas.exacttiming.TimeValue`
        :param int min_index: examine fragments with index greater than or equal to this index (i.e., included)
        :param int max_index: examine fragments with index smaller than this index (i.e., excluded)
        :raises ValueError: if ``min_index`` is negative or ``max_index``
                            is bigger than the current number of fragments
        """
        self.log(u"Called fix_zero_length_fragments")
        self.log([u"  Duration %.3f", duration])
        min_index, max_index = self._check_min_max_indices(min_index, max_index)
        if not self.has_adjacent_fragments_only(min_index, max_index):
            self.log_warn(u"There are non adjacent fragments: aborting")
            return
        i = min_index
        while i < max_index:
            if self[i].interval.has_zero_length:
                self.log([u"  Fragment %d (%s) has zero length => ENLARGE", i, self[i].interval])
                moves = [(i, "ENLARGE", duration)]
                slack = duration
                j = i + 1
                self.log([u"  Entered while with j == %d", j])
                while (j < max_index) and (self[j].interval.length < slack):
                    if self[j].interval.has_zero_length:
                        self.log([u"  Fragment %d (%s) has zero length => ENLARGE", j, self[j].interval])
                        moves.append((j, "ENLARGE", duration))
                        slack += duration
                    else:
                        self.log([u"  Fragment %d (%s) has non zero length => MOVE", j, self[j].interval])
                        moves.append((j, "MOVE", None))
                    j += 1
                self.log([u"  Exited while with j == %d", j])
                fixable = False
                if (j == max_index) and (self[j - 1].interval.end + slack <= self.end):
                    self.log(u"  Fixable by moving back")
                    current_time = self[j - 1].interval.end + slack
                    fixable = True
                elif j < max_index:
                    self.log(u"  Fixable by shrinking")
                    self[j].interval.shrink(slack)
                    current_time = self[j].interval.begin
                    fixable = True
                if fixable:
                    for index, move_type, move_amount in moves[::-1]:
                        self[index].interval.move_end_at(current_time)
                        if move_type == "ENLARGE":
                            self[index].interval.enlarge(move_amount)
                        current_time = self[index].interval.begin
                else:
                    self.log([u"Unable to fix fragment %d (%s)", i, self[i].interval])
                i = j - 1
            i += 1
