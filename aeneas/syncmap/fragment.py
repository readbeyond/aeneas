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
TBW
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.timevalue import Decimal
from aeneas.timevalue import TimeValue
import aeneas.globalfunctions as gf


class SyncMapFragment(object):
    """
    A sync map fragment, that is,
    a text fragment and an associated time interval ``[begin, end]``.

    :param text_fragment: the text fragment
    :type  text_fragment: :class:`~aeneas.textfile.TextFragment`
    :param begin: the begin time of the audio interval
    :type  begin: :class:`~aeneas.timevalue.TimeValue`
    :param end: the end time of the audio interval
    :type  end: :class:`~aeneas.timevalue.TimeValue`
    :param float confidence: the confidence of the audio timing
    """

    TAG = u"SyncMapFragment"

    def __init__(
            self,
            text_fragment=None,
            begin=None,
            end=None,
            confidence=1.0
    ):
        self.text_fragment = text_fragment
        self.begin = begin
        self.end = end
        self.confidence = confidence

    def __unicode__(self):
        return u"%s %.3f %.3f" % (
            self.text_fragment.identifier,
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
    def begin(self):
        """
        The begin time of this sync map fragment.

        :rtype: :class:`~aeneas.timevalue.TimeValue`
        """
        return self.__begin

    @begin.setter
    def begin(self, begin):
        self.__begin = begin

    @property
    def end(self):
        """
        The end time of this sync map fragment.

        :rtype: :class:`~aeneas.timevalue.TimeValue`
        """
        return self.__end

    @end.setter
    def end(self, end):
        self.__end = end

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
    def audio_duration(self):
        """
        The audio duration of this sync map fragment,
        as end time minus begin time.

        :rtype: :class:`~aeneas.timevalue.TimeValue`
        """
        if (self.begin is None) or (self.end is None):
            return TimeValue("0.000")
        return self.end - self.begin

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
