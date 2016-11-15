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
from aeneas.exacttiming import TimePoint
import aeneas.globalfunctions as gf


class SyncMapFragment(object):
    """
    A sync map fragment, that is,
    a text fragment and an associated time interval.

    :param text_fragment: the text fragment
    :type  text_fragment: :class:`~aeneas.textfile.TextFragment`
    :param begin: the begin time of the audio interval
    :type  begin: :class:`~aeneas.exacttiming.TimePoint`
    :param end: the end time of the audio interval
    :type  end: :class:`~aeneas.exacttiming.TimePoint`
    :param float confidence: the confidence of the audio timing
    """

    TAG = u"SyncMapFragment"

    FRAGMENT_TYPE_REGULAR = 0
    """ Regular fragment """

    FRAGMENT_TYPE_HEAD = 1
    """ Head fragment """

    FRAGMENT_TYPE_TAIL = 2
    """ Tail fragment """

    FRAGMENT_TYPE_SILENCE = 3
    """ (Long) Silence fragment """

    def __init__(
            self,
            text_fragment=None,
            begin=None,
            end=None,
            fragment_type=FRAGMENT_TYPE_REGULAR,
            confidence=1.0
    ):
        self.text_fragment = text_fragment
        if (begin is not None) and (end is not None):
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

        * :data:`~aeneas.syncmap.fragment.SyncMapFragment.FRAGMENT_TYPE_REGULAR`
        * :data:`~aeneas.syncmap.fragment.SyncMapFragment.FRAGMENT_TYPE_HEAD`
        * :data:`~aeneas.syncmap.fragment.SyncMapFragment.FRAGMENT_TYPE_TAIL`
        * :data:`~aeneas.syncmap.fragment.SyncMapFragment.FRAGMENT_TYPE_SILENCE`

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

        :rtype: :class:`~aeneas.exacttiming.TimePoint`
        """
        if self.interval is None:
            return None
        return self.interval.begin

    @begin.setter
    def begin(self, begin):
        if self.interval is None:
            raise TypeError(u"Attempting to set begin when interval is None")
        if not isinstance(begin, TimePoint):
            raise TypeError(u"The given begin value is not an instance of TimePoint")
        self.interval.begin = begin

    @property
    def end(self):
        """
        The end time of this sync map fragment.

        :rtype: :class:`~aeneas.exacttiming.TimePoint`
        """
        if self.interval is None:
            return None
        return self.interval.end

    @end.setter
    def end(self, end):
        if self.interval is None:
            raise TypeError(u"Attempting to set end when interval is None")
        if not isinstance(end, TimePoint):
            raise TypeError(u"The given end value is not an instance of TimePoint")
        self.interval.end = end

    @property
    def audio_duration(self):
        """
        The audio duration of this sync map fragment,
        as end time minus begin time.

        :rtype: :class:`~aeneas.exacttiming.TimePoint`
        """
        if self.interval is None:
            return TimePoint("0.000")
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
        if self.audio_duration == TimePoint("0.000"):
            return None
        return Decimal(self.chars / self.audio_duration)
