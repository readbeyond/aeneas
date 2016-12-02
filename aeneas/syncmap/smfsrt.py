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

from aeneas.syncmap.smfgsubtitles import SyncMapFormatGenericSubtitles
import aeneas.globalfunctions as gf


class SyncMapFormatSRT(SyncMapFormatGenericSubtitles):
    """
    Handler for SubRip (SRT) I/O format.
    """

    TAG = u"SyncMapFormatSRT"

    DEFAULT = "srt"

    #
    # NOTE not strictly necessary, since SyncMapFormatGenericSubtitles
    #      has these values set to those needed by SRT
    #      but for clarity we repeat them here
    #
    def __init__(self, variant=DEFAULT, parameters=None, rconf=None, logger=None):
        super(SyncMapFormatSRT, self).__init__(variant=variant, parameters=parameters, rconf=rconf, logger=logger)
        self.header_string = None
        self.header_might_not_have_trailing_blank_line = False
        self.footer_string = None
        self.cue_has_identifier = True
        self.cue_has_optional_identifier = False
        self.time_values_separator = u" --> "
        self.line_break_symbol = u"\n"
        self.parse_time_function = gf.time_from_srt
        self.format_time_function = gf.time_to_srt
