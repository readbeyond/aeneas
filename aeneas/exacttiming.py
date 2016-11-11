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

* :class:`~aeneas.exacttiming.TimePoint`,
  a numeric type to represent time values with arbitrary precision.

.. versionadded:: 1.5.0
"""

from __future__ import absolute_import
from __future__ import print_function
from decimal import Decimal
from decimal import InvalidOperation
import sys


PY2 = (sys.version_info[0] == 2)


class TimePoint(Decimal):
    """
    A numeric type to represent time values with arbitrary precision.
    """

    def __repr__(self):
        return super(TimePoint, self).__repr__().replace("Decimal", "TimePoint")

    # NOTE overriding so that the result
    #      is still an instance of TimePoint

    def __add__(self, other, context=None):
        if PY2:
            return TimePoint(Decimal.__add__(self, other, context))
        return TimePoint(Decimal.__add__(self, other))

    def __div__(self, other, context=None):
        if PY2:
            return TimePoint(Decimal.__div__(self, other, context))
        return TimePoint(Decimal.__div__(self, other))

    def __floordiv__(self, other, context=None):
        if PY2:
            return TimePoint(Decimal.__floordiv__(self, other, context))
        return TimePoint(Decimal.__floordiv__(self, other))

    def __mod__(self, other, context=None):
        if PY2:
            return TimePoint(Decimal.__mod__(self, other, context))
        return TimePoint(Decimal.__mod__(self, other))

    def __mul__(self, other, context=None):
        if PY2:
            return TimePoint(Decimal.__mul__(self, other, context))
        return TimePoint(Decimal.__mul__(self, other))

    def __radd__(self, other, context=None):
        if PY2:
            return TimePoint(Decimal.__radd__(self, other, context))
        return TimePoint(Decimal.__radd__(self, other))

    def __rdiv__(self, other, context=None):
        if PY2:
            return TimePoint(Decimal.__rdiv__(self, other, context))
        return TimePoint(Decimal.__rdiv__(self, other))

    def __rfloordiv__(self, other, context=None):
        if PY2:
            return TimePoint(Decimal.__rfloordiv__(self, other, context))
        return TimePoint(Decimal.__rfloordiv__(self, other))

    def __rmod__(self, other, context=None):
        if PY2:
            return TimePoint(Decimal.__rmod__(self, other, context))
        return TimePoint(Decimal.__rmod__(self, other))

    def __rmul__(self, other, context=None):
        if PY2:
            return TimePoint(Decimal.__rmul__(self, other, context))
        return TimePoint(Decimal.__rmul__(self, other))

    def __rsub__(self, other, context=None):
        if PY2:
            return TimePoint(Decimal.__rsub__(self, other, context))
        return TimePoint(Decimal.__rsub__(self, other))

    def __rtruediv__(self, other, context=None):
        if PY2:
            return TimePoint(Decimal.__rtruediv__(self, other, context))
        return TimePoint(Decimal.__rtruediv__(self, other))

    def __sub__(self, other, context=None):
        if PY2:
            return TimePoint(Decimal.__sub__(self, other, context))
        return TimePoint(Decimal.__sub__(self, other))

    def __truediv__(self, other, context=None):
        if PY2:
            return TimePoint(Decimal.__truediv__(self, other, context))
        return TimePoint(Decimal.__truediv__(self, other))
