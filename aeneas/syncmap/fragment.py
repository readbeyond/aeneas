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

    SILENCE = 3
    """ (Long) Silence fragment """

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
        * :data:`~aeneas.syncmap.fragment.SyncMapFragment.SILENCE`

        :rtype: int
        """
        return self.__fragment_type

    @fragment_type.setter
    def fragment_type(self, fragment_type):
        self.__fragment_type = fragment_type

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
    def audio_duration(self):
        """
        The audio duration of this sync map fragment,
        as end time minus begin time.

        :rtype: :class:`~aeneas.exacttiming.TimeValue`
        """
        if self.interval is None:
            return TimeValue("0.000")
        return self.interval.length

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

        :rtype: None or Decimal

        .. versionadded:: 1.2.0
        """
        if self.audio_duration == TimeValue("0.000"):
            return None
        return Decimal(self.chars / self.audio_duration)
