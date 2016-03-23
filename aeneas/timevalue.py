#!/usr/bin/env python
# coding=utf-8

"""
This module contains the following classes:

* :class:`~aeneas.timevalue.TimeValue`,
  a numeric type to represent time values with arbitrary precision.

.. versionadded:: 1.5.0
"""

from __future__ import absolute_import
from __future__ import print_function
from decimal import Decimal
from decimal import InvalidOperation
import sys

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.5.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

PY2 = (sys.version_info[0] == 2)

class TimeValue(Decimal):
    """
    A numeric type to represent time values with arbitrary precision.
    """

    def __repr__(self):
        return super(TimeValue, self).__repr__().replace("Decimal", "TimeValue")

    # NOTE overriding so that the result
    #      is still an instance of TimeValue

    def __add__(self, other, context=None):
        if PY2:
            return TimeValue(Decimal.__add__(self, other, context))
        return TimeValue(Decimal.__add__(self, other))

    def __div__(self, other, context=None):
        if PY2:
            return TimeValue(Decimal.__div__(self, other, context))
        return TimeValue(Decimal.__div__(self, other))

    def __floordiv__(self, other, context=None):
        if PY2:
            return TimeValue(Decimal.__floordiv__(self, other, context))
        return TimeValue(Decimal.__floordiv__(self, other))

    def __mod__(self, other, context=None):
        if PY2:
            return TimeValue(Decimal.__mod__(self, other, context))
        return TimeValue(Decimal.__mod__(self, other))

    def __mul__(self, other, context=None):
        if PY2:
            return TimeValue(Decimal.__mul__(self, other, context))
        return TimeValue(Decimal.__mul__(self, other))

    def __radd__(self, other, context=None):
        if PY2:
            return TimeValue(Decimal.__radd__(self, other, context))
        return TimeValue(Decimal.__radd__(self, other))

    def __rdiv__(self, other, context=None):
        if PY2:
            return TimeValue(Decimal.__rdiv__(self, other, context))
        return TimeValue(Decimal.__rdiv__(self, other))

    def __rfloordiv__(self, other, context=None):
        if PY2:
            return TimeValue(Decimal.__rfloordiv__(self, other, context))
        return TimeValue(Decimal.__rfloordiv__(self, other))

    def __rmod__(self, other, context=None):
        if PY2:
            return TimeValue(Decimal.__rmod__(self, other, context))
        return TimeValue(Decimal.__rmod__(self, other))

    def __rmul__(self, other, context=None):
        if PY2:
            return TimeValue(Decimal.__rmul__(self, other, context))
        return TimeValue(Decimal.__rmul__(self, other))

    def __rsub__(self, other, context=None):
        if PY2:
            return TimeValue(Decimal.__rsub__(self, other, context))
        return TimeValue(Decimal.__rsub__(self, other))

    def __rtruediv__(self, other, context=None):
        if PY2:
            return TimeValue(Decimal.__rtruediv__(self, other, context))
        return TimeValue(Decimal.__rtruediv__(self, other))

    def __sub__(self, other, context=None):
        if PY2:
            return TimeValue(Decimal.__sub__(self, other, context))
        return TimeValue(Decimal.__sub__(self, other))

    def __truediv__(self, other, context=None):
        if PY2:
            return TimeValue(Decimal.__truediv__(self, other, context))
        return TimeValue(Decimal.__truediv__(self, other))



