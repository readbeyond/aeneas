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
from copy import deepcopy
from bisect import insort

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

    def _is_valid_index(self, index):
        """
        Return ``True`` if and only if the given ``index``
        is valid.
        """
        if isinstance(index, int):
            return (index >= 0) and (index < len(self))
        if isinstance(index, list):
            valid = True
            for i in index:
                valid = valid or self._is_valid_index(i)
            return valid
        return False

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

    def clone(self):
        """
        Return a deep copy of this configuration object.

        :rtype: :class:`~aeneas.syncmap.fragmentlist.SyncMapFragmentList`
        """
        return deepcopy(self)

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

    @property
    def regular_fragments(self):
        """
        Iterates through the regular fragments in the list
        (which are sorted).

        :rtype: generator of (int, :class:`~aeneas.syncmap.SyncMapFragment`)
        """
        for i, fragment in enumerate(self.__fragments):
            if fragment.fragment_type == SyncMapFragment.REGULAR:
                yield (i, fragment)

    @property
    def nonspeech_fragments(self):
        """
        Iterates through the nonspeech fragments in the list
        (which are sorted).

        :rtype: generator of (int, :class:`~aeneas.syncmap.SyncMapFragment`)
        """
        for i, fragment in enumerate(self.__fragments):
            if fragment.fragment_type == SyncMapFragment.NONSPEECH:
                yield (i, fragment)

    def remove(self, indices):
        """
        Remove the fragments corresponding to the given list of indices.

        :param indices: the list of indices to be removed
        :type  indices: list of int
        :raises ValueError: if one of the indices is not valid
        """
        if not self._is_valid_index(indices):
            self.log_exc(u"The given list of indices is not valid", None, True, ValueError)
        new_fragments = []
        sorted_indices = sorted(indices)
        i = 0
        j = 0
        while (i < len(self)) and (j < len(sorted_indices)):
            if i != sorted_indices[j]:
                new_fragments.append(self[i])
            else:
                j += 1
            i += 1
        while i < len(self):
            new_fragments.append(self[i])
            i += 1
        self.__fragments = new_fragments

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
            current_interval = self[i].interval
            next_interval = self[i + 1].interval
            if current_interval.relative_position_of(next_interval) not in self.ALLOWED_POSITIONS:
                self.log(u"Found overlapping fragments:")
                self.log([u"  Index %d => %s", i, current_interval])
                self.log([u"  Index %d => %s", i + 1, next_interval])
                self.log_exc(u"The list contains two fragments overlapping in a forbidden way", None, True, ValueError)
        self.log(u"Checking relative positions... done")
        self.__sorted = True

    def remove_nonspeech_fragments(self, zero_length_only=False):
        """
        Remove ``NONSPEECH`` fragments from the list.

        If ``zero_length_only`` is ``True``, remove only
        those fragments with zero length,
        and make all the others ``REGULAR``.

        :param bool zero_length_only: remove only zero length NONSPEECH fragments
        """
        self.log(u"Removing nonspeech fragments...")
        nonspeech = list(self.nonspeech_fragments)
        if zero_length_only:
            nonspeech = [(i, f) for i, f in nonspeech if f.has_zero_length]
        nonspeech_indices = [i for i, f in nonspeech]
        self.remove(nonspeech_indices)
        if zero_length_only:
            for i, f in list(self.nonspeech_fragments):
                f.fragment_type = SyncMapFragment.REGULAR
        self.log(u"Removing nonspeech fragments... done")

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
        zero = [i for i in range(min_index, max_index) if self[i].has_zero_length]
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
            current_interval = self[i].interval
            next_interval = self[i + 1].interval
            if not current_interval.is_adjacent_before(next_interval):
                self.log(u"Found non adjacent fragments")
                self.log([u"  Index %d => %s", i, current_interval])
                self.log([u"  Index %d => %s", i + 1, next_interval])
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
            insort(self.__fragments, fragment)
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
        self.log([u"  List begin: %.3f", self.begin])
        self.log([u"  List end:   %.3f", self.end])
        nsi_index = 0
        frag_index = 0
        nsi_counter = [(n, []) for n in nonspeech_intervals]
        # NOTE the last fragment is not eligible to be returned
        while (nsi_index < len(nonspeech_intervals)) and (frag_index < len(self) - 1):
            nsi = nonspeech_intervals[nsi_index]
            if nsi.end > self.end:
                self.log(u"    nsi ends after self.end => breaking")
                break
            nsi_shadow = nsi.shadow(tolerance)
            frag = self[frag_index]
            self.log([u"  nsi        %s", nsi])
            self.log([u"  nsi_shadow %s", nsi_shadow])
            self.log([u"  frag       %s", frag.interval])
            if not frag.is_head_or_tail:
                self.log(u"    Fragment is not HEAD or TAIL => inspecting it")
                if nsi_shadow.contains(frag.end):
                    if nsi_shadow.contains(frag.begin):
                        #
                        #      *************** nsi shadow
                        #      | *********** | nsi
                        #      |   ***X      | frag (X=frag.end)
                        #
                        # NOTE this case might happen as the following:
                        #
                        #      *************** nsi shadow
                        #      |     ***     | nsi
                        #      | **X         | frag (X=frag.end)
                        #
                        #      so we must invalidate the nsi if this happens
                        #
                        nsi_counter[nsi_index] = (None, []) 
                        nsi_index += 1
                        frag_index += 1
                        self.log(u"    nsi_shadow entirely contains frag => invalidate nsi, and skip to next fragment, nsi")
                    else:
                        #
                        #      *************** nsi shadow
                        #      | *********** | nsi
                        # *****|***X         | frag (X=frag.end)
                        #
                        nsi_counter[nsi_index][1].append(frag_index)
                        frag_index += 1
                        self.log(u"    nsi_shadow contains frag end only => save it and go to next fragment")
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
        # set the appropriate fragment text
        if replacement_string in [None, gc.PPV_TASK_ADJUST_BOUNDARY_NONSPEECH_REMOVE]:
            self.log(u"  Remove long nonspeech")
            lines = []
        else:
            self.log([u"  Replace long nonspeech with '%s'", replacement_string])
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
                fragment_type=SyncMapFragment.NONSPEECH
            ), sort=False)
        self.log(u"  Second pass: append nonspeech intervals... done")
        self.log(u"  Third pass: sorting...")
        self.sort()
        self.log(u"  Third pass: sorting... done")

    def fix_zero_length_fragments(self, duration=TimeValue("0.001"), min_index=None, max_index=None, ensure_adjacent=True):
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
        if len(self) < 1:
            self.log(u"The list has no fragments: returning")
            return
        if not self.has_adjacent_fragments_only(min_index, max_index):
            self.log_warn(u"There are non adjacent fragments: aborting")
            return
        original_first_begin = None
        if (
                (ensure_adjacent) and
                (min_index > 0) and
                (self[min_index - 1].interval.is_adjacent_before(self[min_index].interval))
        ):
            original_first_begin = self[min_index].begin
            self.log([u"Original first was adjacent with previous, starting at %.3f", original_first_begin])
        original_last_end = None
        if (
                (ensure_adjacent) and
                (len(self) > 1) and
                (max_index < len(self)) and
                (self[max_index - 1].interval.is_adjacent_before(self[max_index].interval))
        ):
            original_last_end = self[max_index - 1].end
            self.log([u"Original last was adjacent with next, ending at %.3f", original_last_end])
        i = min_index
        while i < max_index:
            if self[i].has_zero_length:
                self.log([u"  Fragment %d (%s) has zero length => ENLARGE", i, self[i].interval])
                moves = [(i, "ENLARGE", duration)]
                slack = duration
                j = i + 1
                self.log([u"  Entered while with j == %d", j])
                while (j < max_index) and (self[j].interval.length < slack):
                    if self[j].has_zero_length:
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
        if original_first_begin is not None:
            if self[min_index].begin != self[min_index - 1].end:
                self.log(u"First fragment begin moved, restoring adjacency")
                self.log([u"  Original was %.3f", original_first_begin])
                self.log([u"  New      is  %.3f", self[min_index - 1].end])
                self[min_index].begin = self[min_index - 1].end
        if original_last_end is not None:
            if self[max_index].begin != self[max_index - 1].end:
                self.log(u"Last fragment end moved, restoring adjacency")
                self.log([u"  Original was %.3f", original_last_end])
                self.log([u"  New      is  %.3f", self[max_index].begin])
                self[max_index].begin = self[max_index - 1].end

    def fix_fragment_rate(self, fragment_index, max_rate, aggressive=False):
        def fix_pair(current_index, donor_index):
            self.log(u"Called fix_pair")
            if (
                (current_index < 0) or
                (current_index >= len(self)) or
                (donor_index < 0) or
                (donor_index >= len(self)) or
                (abs(current_index - donor_index) > 1)
            ):
                self.log(u"Invalid index, returning False")
                return False
            donor_is_previous = donor_index < current_index
            current_fragment = self[current_index]
            donor_fragment = self[donor_index]
            if (current_fragment.rate is not None) and (current_fragment.rate <= max_rate):
                self.log(u"Current fragment rate is already <= max_rate, returning True")
                return True
            if donor_is_previous:
                if not donor_fragment.interval.is_non_zero_before_non_zero(current_fragment.interval):
                    self.log(u"donor fragment is not adjacent before current fragment, returning False")
                    return False
            else:
                if not current_fragment.interval.is_non_zero_before_non_zero(donor_fragment.interval):
                    self.log(u"current fragment is not adjacent before donor fragment, returning False")
                    return False

            self.log(u"Current and donor fragments are adjacent and not zero length")
            current_lack = current_fragment.rate_lack(max_rate)
            donor_slack = donor_fragment.rate_slack(max_rate)
            self.log([u"Current lack %.3f", current_lack])
            self.log([u"Donor  slack %.3f", donor_slack])
            if donor_slack <= 0:
                self.log(u"Donor has no slack, returning False")
                return False
            self.log(u"Donor has some slack")
            effective_slack = min(current_lack, donor_slack)
            if donor_is_previous:
                self.move_transition_point(donor_index, donor_fragment.end - effective_slack)
            else:
                self.move_transition_point(current_index, current_fragment.end + effective_slack)
            if effective_slack == current_lack:
                self.log(u"Current lack can be fully stolen from donor")
                return True
            else:
                self.log(u"Current lack can be partially stolen from donor")
                return False

        # try fixing rate stealing slack from the previous fragment
        if fix_pair(fragment_index, fragment_index - 1):
            return True
        # if aggressive, try fixing rate stealing slack from the next fragment
        if aggressive:
            return fix_pair(fragment_index, fragment_index + 1)
        # cannot be fixed
        return False
