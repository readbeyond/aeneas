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

from aeneas.syncmap.smfgtabular import SyncMapFormatGenericTabular


class SyncMapFormatCSV(SyncMapFormatGenericTabular):
    """
    Handler for comma-separated values (CSV) I/O format.
    """

    TAG = u"SyncMapFormatCSV"

    DEFAULT = "csv"

    HUMAN = "csvh"

    MACHINE = "csvm"

    MACHINE_ALIASES = [DEFAULT, MACHINE]

    FIELD_DELIMITER = u","

    FIELDS = {
        "identifier": 0,
        "begin": 1,
        "end": 2,
        "text": 3
    }

    TEXT_DELIMITER = u"\""
