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


class SyncMapHeadTailFormat(object):
    """
    Enumeration of the supported output formats
    for the head and tail of
    the synchronization maps.

    .. versionadded:: 1.2.0
    """

    ADD = "add"
    """
    Add two empty sync map fragments,
    one at the begin and one at the end of the sync map,
    corresponding to the head and the tail.

    For example::

        0.000 0.500 HEAD
        0.500 1.234 First fragment
        1.234 5.678 Second fragment
        5.678 7.000 Third fragment
        7.000 7.890 TAIL

    becomes::

        0.000 0.500
        0.500 1.234 First fragment
        1.234 5.678 Second fragment
        5.678 7.000 Third fragment
        7.000 7.890

    """

    HIDDEN = "hidden"
    """
    Do not output sync map fragments for the head and tail.


    For example::

        0.000 0.500 HEAD
        0.500 1.234 First fragment
        1.234 5.678 Second fragment
        5.678 7.000 Third fragment
        7.000 7.890 TAIL

    becomes::

        0.500 1.234 First fragment
        1.234 5.678 Second fragment
        5.678 7.000 Third fragment

    """

    STRETCH = "stretch"
    """
    Set the `begin` attribute of the first sync map fragment to `0`,
    and the `end` attribute of the last sync map fragment to
    the length of the audio file.

    For example::

        0.000 0.500 HEAD
        0.500 1.234 First fragment
        1.234 5.678 Second fragment
        5.678 7.000 Third fragment
        7.000 7.890 TAIL

    becomes::

        0.000 1.234 First fragment
        1.234 5.678 Second fragment
        5.678 7.890 Third fragment

    """

    ALLOWED_VALUES = [ADD, HIDDEN, STRETCH]
    """ List of all the allowed values """
