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

from aeneas.syncmap.smfgtabular import SyncMapFormatGenericTabular


class SyncMapFormatAudacity(SyncMapFormatGenericTabular):

    TAG = u"SyncMapFormatAudacity"

    DEFAULT = "aud"
    """
    Alias for ``MACHINE``.
    """

    HUMAN = "audh"
    """
    Tab-separated plain text,
    with human-readable time values
    and fragment text::

        00:00:00.000   00:00:01.234   Text of fragment 1
        00:00:01.234   00:00:05.678   Text of fragment 2
        00:00:05.678   00:00:07.890   Text of fragment 3

    * Multiple levels: no
    * Multiple lines: no

    .. versionadded:: 1.5.0
    """

    MACHINE = "audm"
    """
    Tab-separated plain text,
    with machine-readable time values
    and fragment text,
    compatible with ``Audacity``::

        0.000   1.234   Text fragment 1
        1.234   5.678   Text fragment 2
        5.678   7.890   Text fragment 3

    * Multiple levels: no
    * Multiple lines: no

    .. versionadded:: 1.5.0
    """

    MACHINE_ALIASES = [DEFAULT, MACHINE]

    FIELD_DELIMITER = u"\t"

    FIELDS = {
        "begin": 0,
        "end": 1,
        "text": 2
    }

    TEXT_DELIMITER = None
