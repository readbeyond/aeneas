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
from aeneas.syncmap.fragment import SyncMapFragment
from aeneas.textfile import TextFragment
import aeneas.globalconstants as gc


class SyncMapFragmentList(object):
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
    :raises TypeError: if ``begin`` or ``end`` are not ``Non`` and not instances of :class:`~aeneas.exacttiming.TimeValue`
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

    def __init__(self, begin=TimeValue("0.000"), end=None):
        if (begin is not None) and (not isinstance(begin, TimeValue)):
            raise TypeError(u"begin is not an instance of TimeValue")
        if (end is not None) and (not isinstance(end, TimeValue)):
            raise TypeError(u"end is not an instance of TimeValue")
        if (begin is not None):
            if begin < 0:
                raise ValueError(u"begin is negative")
            if (end is not None) and (begin > end):
                raise ValueError(u"begin is bigger than end")
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
        # TODO bisect does not work if there is a configuration like:
        #
        #   *********** <- existing interval
        #        ***    <- query interval
        #
        # one should probably do this by bisect
        # over the begin and end lists separately
        #
        for existing_fragment in self.fragments:
            if existing_fragment.interval.relative_position_of(fragment.interval) not in self.ALLOWED_POSITIONS:
                raise ValueError(u"interval overlaps another already present interval")

    def sort(self):
        """
        Sort the intervals, if they are not sorted already.

        :raises ValueError: if ``interval`` does not respect the boundaries of the list
                            or if it overlaps an existing interval
        """
        if not self.is_guaranteed_sorted:
            self.__fragments = sorted(self.__fragments)
            for i in range(len(self) - 1):
                if self[i].interval.relative_position_of(self[i + 1].interval) not in self.ALLOWED_POSITIONS:
                    raise ValueError(u"The list contains two time intervals overlapping in a forbidden way")
            self.__sorted = True

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
                raise ValueError(u"Unable to add with sort=True if the list is not guaranteed sorted")
            # insert sorted, on the right if there is a tie
            self._check_overlap(fragment)
            bisect.insort(self.__fragments, fragment)
        else:
            # just append at the end
            self.__fragments.append(fragment)
            self.__sorted = False

    def move_transition_point(self, fragment_index, value):
        """
        Change the transition point between fragment ``fragment_index``
        and the next fragment to the time value ``value``.

        This method fails silently without changing the fragment list
        if at least one of the following conditions holds:

        * ``fragment_index`` is negative
        * ``fragment_index`` is the last or the second-to-last
        * ``value`` is after the current end of the next fragment
        * the current fragment and the next one are not adjacent

        The above conditions ensures that the move makes sense
        and keeps the list sorted.

        :param int fragment_index: the fragment index whose end should be moved
        :param value: the new transition point
        :type  value: :class:`~aeneas.exacttiming.TimeValue`
        """
        if (fragment_index < 0) or (fragment_index > (len(self) - 3)):
            # fails silently
            return
        current_fragment = self[fragment_index]
        next_fragment = self[fragment_index + 1]
        if (value > next_fragment.end) or (not current_fragment.interval.is_adjacent_before(next_fragment.interval)):
            # fails silently
            return
        current_fragment.interval.end = value
        next_fragment.interval.begin = value

    @property
    def fragments(self):
        """
        Iterates through the fragments in the list
        (which are sorted).

        :rtype: generator of :class:`~aeneas.syncmap.SyncMapFragment`
        """
        for fragment in self.__fragments:
            yield fragment

    def offset(self, offset):
        """
        Move all the intervals in the list by the given ``offset``.

        :param offset: the shift to be applied
        :type  offset: :class:`~aeneas.exacttiming.TimeValue`
        :raises TypeError: if ``offset`` is not an instance of ``TimeValue``
        """
        for fragment in self.fragments:
            fragment.interval.offset(
                offset=offset,
                allow_negative=False,
                min_begin_value=self.begin,
                max_end_value=self.end
            )

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
        nsi_index = 0
        frag_index = 0
        nsi_counter = [(n, []) for n in nonspeech_intervals]
        while (nsi_index < len(nonspeech_intervals)) and (frag_index < len(self) - 1):
            nsi = nonspeech_intervals[nsi_index]
            nsi_shadow = nsi.shadow(tolerance)
            frag = self[frag_index]
            if frag.fragment_type == SyncMapFragment.REGULAR:
                if nsi_shadow.contains(frag.end):
                    #
                    #      *************** nsi shadow
                    #      | *********** | nsi
                    # *****|***X         | frag (X=frag.end)
                    #
                    nsi_counter[nsi_index][1].append(frag_index)
                    frag_index += 1
                elif nsi_shadow.begin > frag.end:
                    #
                    #      *************** nsi shadow
                    #      | *********** | nsi
                    # **X  |             | frag (X=frag.end)
                    #
                    frag_index += 1
                else:
                    #
                    #       ***************    nsi shadow
                    #       | *********** |    nsi
                    #       |        *****|**X frag (X=frag.end)
                    #
                    nsi_index += 1
            else:
                frag_index += 1
        return [(n, c[0]) for (n, c) in nsi_counter if len(c) == 1]

    def inject_long_nonspeech_fragments(self, pairs, replacement_string):
        """
        Inject nonspeech fragments corresponding to the given intervals
        in this fragment list.

        :param list pairs: list of ``(TimeInterval, int)`` pairs,
                           each identifying a nonspeech interval and
                           the corresponding fragment index ending inside it
        :param string replacement_string: the string to be applied to the nonspeech intervals
        """
        # set the appropriate type and text
        if replacement_string in [None, gc.PPV_TASK_ADJUST_BOUNDARY_NONSPEECH_REMOVE]:
            fragment_type = SyncMapFragment.NONSPEECH
            lines = []
        else:
            fragment_type = SyncMapFragment.REGULAR
            lines = [replacement_string]
        # first, make room for the nonspeech intervals
        for nsi, index in pairs:
            self[index].interval.end = nsi.begin
            self[index + 1].interval.begin = nsi.end
        # then, append the nonspeech intervals
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
        # sort the list
        self.sort()

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
        """
        min_index = min_index or 0
        max_index = max_index or len(self)
        i = min_index
        while i < max_index:
            if self[i].interval.has_zero_length:
                moves = [(i, "ENLARGE", duration)]
                slack = duration
                j = i + 1
                while (j < max_index) and (self[j].interval.length < slack):
                    if self[j].interval.has_zero_length:
                        moves.append((j, "ENLARGE", duration))
                        slack += duration
                    else:
                        moves.append((j, "MOVE", None))
                    j += 1
                fixable = False
                if (j == max_index) and (self[j - 1].interval.end + slack <= self.end):
                    current_time = self[j - 1].interval.end + slack
                    fixable = True
                elif j < max_index:
                    self[j].interval.shrink(slack)
                    current_time = self[j].interval.begin
                    fixable = True
                if fixable:
                    for index, move_type, move_amount in moves[::-1]:
                        self[index].interval.move_end_at(current_time)
                        if move_type == "ENLARGE":
                            self[index].interval.enlarge(move_amount)
                        current_time = self[index].interval.begin
                # else:
                #     # unable to fix
                #     pass
                i = j - 1
            i += 1
