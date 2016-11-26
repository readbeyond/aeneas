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

from aeneas.exacttiming import Decimal
from aeneas.exacttiming import TimeInterval
from aeneas.exacttiming import TimeValue
import aeneas.globalfunctions as gf


class SyncMapFragment(object):
    """
    A sync map fragment, that is,
    a text fragment and an associated time interval.

    :param text_fragment: the text fragment
    :type  text_fragment: :class:`~aeneas.textfile.TextFragment`
    :param begin: the begin time of the audio interval
    :type  begin: :class:`~aeneas.exacttiming.TimeValue`
    :param end: the end time of the audio interval
    :type  end: :class:`~aeneas.exacttiming.TimeValue`
    :param float confidence: the confidence of the audio timing
    """

    TAG = u"SyncMapFragment"

    REGULAR = 0
    """ Regular fragment """

    HEAD = 1
    """ Head fragment """

    TAIL = 2
    """ Tail fragment """

    NONSPEECH = 3
    """ Nonspeech fragment (not head nor tail) """

    NOT_REGULAR_TYPES = [HEAD, TAIL, NONSPEECH]
    """ Types of fragment different than ``REGULAR`` """

    def __init__(
            self,
            text_fragment=None,
            interval=None,
            begin=None,
            end=None,
            fragment_type=REGULAR,
            confidence=1.0
    ):
        self.text_fragment = text_fragment
        if interval is not None:
            self.interval = interval
        elif (begin is not None) and (end is not None):
            self.interval = TimeInterval(begin, end)
        else:
            self.interval = None
        self.fragment_type = fragment_type
        self.confidence = confidence

    def __unicode__(self):
        return u"%s %d %.3f %.3f" % (
            self.text_fragment.identifier,
            self.fragment_type,
            self.begin,
            self.end
        )

    def __str__(self):
        return gf.safe_str(self.__unicode__())

    def __eq__(self, other):
        if not isinstance(other, SyncMapFragment):
            return False
        return self.interval == other.interval

    def __ne__(self, other):
        return not (self == other)

    def __gt__(self, other):
        if not isinstance(other, SyncMapFragment):
            return False
        return self.interval > other.interval

    def __lt__(self, other):
        if not isinstance(other, SyncMapFragment):
            return False
        return self.interval < other.interval

    def __ge__(self, other):
        return (self > other) or (self == other)

    def __le__(self, other):
        return (self < other) or (self == other)

    @property
    def text_fragment(self):
        """
        The text fragment associated with this sync map fragment.

        :rtype: :class:`~aeneas.textfile.TextFragment`
        """
        return self.__text_fragment

    @text_fragment.setter
    def text_fragment(self, text_fragment):
        self.__text_fragment = text_fragment

    @property
    def interval(self):
        """
        The time interval corresponding to this fragment.

        :rtype: :class:`~aeneas.exacttiming.TimeInterval`
        """
        return self.__interval

    @interval.setter
    def interval(self, interval):
        self.__interval = interval

    @property
    def fragment_type(self):
        """
        The type of fragment.

        Possible values are:

        * :data:`~aeneas.syncmap.fragment.SyncMapFragment.REGULAR`
        * :data:`~aeneas.syncmap.fragment.SyncMapFragment.HEAD`
        * :data:`~aeneas.syncmap.fragment.SyncMapFragment.TAIL`
        * :data:`~aeneas.syncmap.fragment.SyncMapFragment.NONSPEECH`

        :rtype: int
        """
        return self.__fragment_type

    @fragment_type.setter
    def fragment_type(self, fragment_type):
        self.__fragment_type = fragment_type

    @property
    def is_head_or_tail(self):
        """
        Return ``True`` if the fragment
        is HEAD or TAIL.

        :rtype: bool

        .. versionadded:: 1.7.0
        """
        return self.fragment_type in [self.HEAD, self.TAIL]

    @property
    def is_regular(self):
        """
        Return ``True`` if the fragment
        is REGULAR.

        :rtype: bool

        .. versionadded:: 1.7.0
        """
        return self.fragment_type == self.REGULAR

    @property
    def confidence(self):
        """
        The confidence of the audio timing, from ``0.0`` to ``1.0``.

        Currently this value is not used, and it is always ``1.0``.

        :rtype: float
        """
        return self.__confidence

    @confidence.setter
    def confidence(self, confidence):
        self.__confidence = confidence

    @property
    def pretty_print(self):
        """
        Pretty print representation of this fragment,
        as ``(identifier, begin, end, text)``.

        :rtype: string

        .. versionadded:: 1.7.0
        """
        return u"%s\t%.3f\t%.3f\t%s" % (
            (self.identifier or u""),
            (self.begin if self.begin is not None else TimeValue("-2.000")),
            (self.end if self.end is not None else TimeValue("-1.000")),
            (self.text or u"")
        )

    @property
    def identifier(self):
        """
        The identifier of this sync map fragment.

        :rtype: string

        .. versionadded:: 1.7.0
        """
        if self.text_fragment is None:
            return None
        return self.text_fragment.identifier

    @property
    def text(self):
        """
        The text of this sync map fragment.

        :rtype: string

        .. versionadded:: 1.7.0
        """
        if self.text_fragment is None:
            return None
        return self.text_fragment.text

    @property
    def begin(self):
        """
        The begin time of this sync map fragment.

        :rtype: :class:`~aeneas.exacttiming.TimeValue`
        """
        if self.interval is None:
            return None
        return self.interval.begin

    @begin.setter
    def begin(self, begin):
        if self.interval is None:
            raise TypeError(u"Attempting to set begin when interval is None")
        if not isinstance(begin, TimeValue):
            raise TypeError(u"The given begin value is not an instance of TimeValue")
        self.interval.begin = begin

    @property
    def end(self):
        """
        The end time of this sync map fragment.

        :rtype: :class:`~aeneas.exacttiming.TimeValue`
        """
        if self.interval is None:
            return None
        return self.interval.end

    @end.setter
    def end(self, end):
        if self.interval is None:
            raise TypeError(u"Attempting to set end when interval is None")
        if not isinstance(end, TimeValue):
            raise TypeError(u"The given end value is not an instance of TimeValue")
        self.interval.end = end

    @property
    def length(self):
        """
        The audio duration of this sync map fragment,
        as end time minus begin time.

        :rtype: :class:`~aeneas.exacttiming.TimeValue`
        """
        if self.interval is None:
            return TimeValue("0.000")
        return self.interval.length

    @property
    def has_zero_length(self):
        """
        Returns ``True`` if this sync map fragment has zero length,
        that is, if its begin and end values coincide.

        :rtype: bool

        .. versionadded:: 1.7.0
        """
        return self.length == TimeValue("0.000")

    @property
    def chars(self):
        """
        Return the number of characters of the text fragment,
        not including the line separators.

        :rtype: int

        .. versionadded:: 1.2.0
        """
        if self.text_fragment is None:
            return 0
        return self.text_fragment.chars

    @property
    def rate(self):
        """
        The rate, in characters/second, of this fragment.

        If the fragment is not ``REGULAR`` or its duration is zero,
        return ``None``.

        :rtype: ``None`` or :class:`~aeneas.exacttiming.Decimal`

        .. versionadded:: 1.2.0
        """
        if (
            (self.fragment_type != self.REGULAR) or
            (self.has_zero_length)
        ):
            return None
        return Decimal(self.chars / self.length)

    def rate_lack(self, max_rate):
        """
        The time interval that this fragment lacks
        to respect the given max rate.

        A positive value means that the current fragment
        is faster than the max rate (bad).
        A negative or zero value means that the current fragment
        has rate slower or equal to the max rate (good).

        Always return ``0.000`` for fragments that are not ``REGULAR``.

        :param max_rate: the maximum rate (characters/second)
        :type  max_rate: :class:`~aeneas.exacttiming.Decimal`
        :rtype: :class:`~aeneas.exacttiming.TimeValue`

        .. versionadded:: 1.7.0
        """
        if self.fragment_type == self.REGULAR:
            return self.chars / max_rate - self.length
        return TimeValue("0.000")

    def rate_slack(self, max_rate):
        """
        The maximum time interval that can be stolen to this fragment
        while keeping it respecting the given max rate.

        For ``REGULAR`` fragments this value is
        the opposite of the ``rate_lack``.
        For ``NONSPEECH`` fragments this value is equal to
        the length of the fragment.
        For ``HEAD`` and ``TAIL`` fragments this value is ``0.000``,
        meaning that they cannot be stolen.

        :param max_rate: the maximum rate (characters/second)
        :type  max_rate: :class:`~aeneas.exacttiming.Decimal`
        :rtype: :class:`~aeneas.exacttiming.TimeValue`

        .. versionadded:: 1.7.0
        """
        if self.fragment_type == self.REGULAR:
            return -self.rate_lack(max_rate)
        elif self.fragment_type == self.NONSPEECH:
            return self.length
        else:
            return TimeValue("0.000")
