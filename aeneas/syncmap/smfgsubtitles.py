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

from aeneas.syncmap.smfbase import SyncMapFormatBase
import aeneas.globalfunctions as gf


class SyncMapFormatGenericSubtitles(SyncMapFormatBase):
    """
    Base class for subtitles-like I/O format handlers.
    """

    TAG = u"SyncMapFormatGenericSubtitles"

    DEFAULT = "subtitles"
    """
    The code for the default variant
    associated with this format.
    """

    def __init__(self, variant=DEFAULT, parameters=None, rconf=None, logger=None):
        super(SyncMapFormatGenericSubtitles, self).__init__(variant=variant, parameters=parameters, rconf=rconf, logger=logger)

        #
        # NOTE since we store functions (parse_..., format_...)
        #      we prefer making these instance members rather than class members
        #
        self.header_string = None
        """
        If not ``None``, the file has the given header.
        """

        self.header_might_not_have_trailing_blank_line = False
        """
        If ``True``, the header line might not be followed by a blank line.
        """

        self.footer_string = None
        """
        If not ``None``, the file has the given footer.
        """

        self.cue_has_identifier = True
        """
        If ``True``, each cue has an identifier in its first line.
        """

        self.cue_has_optional_identifier = False
        """
        If ``True``, each cue might have an identifier in its first line.
        """

        self.time_values_separator = u" --> "
        """
        The separator between time values.
        """

        self.line_break_symbol = u"\n"
        """
        Symbol to place between CC lines.
        """

        self.parse_time_function = gf.time_from_srt
        """
        Function to parse time values.
        """

        self.format_time_function = gf.time_to_srt
        """
        Function to format time values.
        """

    def ignore_block(self, block_lines):
        """
        Return ``True`` if the given block of lines should be ignored.
        """
        return False

    def parse(self, input_text, syncmap):
        def get_block(input_lines, i):
            """
            Get all the non-empty, consecutive lines, starting from index i,
            and then skip all subsequent consecutive empty lines,
            returning the index of the next non-empty line
            (i.e., the line where the next block starts).
            """
            acc = []
            while (i < len(input_lines)) and (input_lines[i] != u""):
                acc.append(input_lines[i])
                i += 1
            while (i < len(input_lines)) and (input_lines[i] == u""):
                i += 1
            return (acc, i)

        def parse_time_string(string):
            """
            Parse begin and end time from the given string.
            """
            split = string.split(self.time_values_separator)
            if len(split) < 2:
                self.log_exc(u"The following timing string is malformed: '%s'" % (string), None, True, ValueError)
            #
            # certain formats might have time lines like:
            # "00:00:20,000 --> 00:00:22,000  X1:40 X2:600 Y1:20 Y2:50"
            # so:
            # begin = "00:00:20,000"
            # end = "00:00:22,000  X1:40 X2:600 Y1:20 Y2:50"
            # so, for end, we need to split over spaces
            # and keep only the first element
            #
            begin, end = split[0:2]
            begin = begin.strip()
            end = ((end.strip()).split(u" "))[0]
            begin = self.parse_time_function(begin)
            end = self.parse_time_function(end)
            return (begin, end)

        input_lines = [l.strip() for l in input_text.splitlines()]
        i = 0
        cue_index = 1

        # skip header, if any
        if self.header_string is not None:
            while i < len(input_lines):
                if input_lines[i].startswith(self.header_string):
                    if self.header_might_not_have_trailing_blank_line:
                        i += 1
                    else:
                        acc, i = get_block(input_lines, i)
                    break
                i += 1

        # skip any blank lines after the header
        while (i < len(input_lines)) and (input_lines[i] == u""):
            i += 1

        # input_lines[i] is not empty
        while i < len(input_lines):
            acc, i = get_block(input_lines, i)

            if len(acc) < 1:
                # no block => break
                break

            if (
                    (self.footer_string is not None) and
                    (acc[0].startswith(self.footer_string))
            ):
                # we reached the footer => break
                break

            if not self.ignore_block(acc):
                # get identifier, if any
                j = 0
                identifier = u"f%06d" % cue_index
                if self.cue_has_identifier:
                    identifier = acc[j]
                    j += 1
                elif self.cue_has_optional_identifier:
                    if self.time_values_separator not in acc[j]:
                        # no time values separator in this line => it is an identifier
                        identifier = acc[j]
                        j += 1

                # get timing string and parse it
                begin, end = parse_time_string(acc[j])
                j += 1

                # get text lines
                lines = (u"\n".join(acc[j:])).split(self.line_break_symbol)

                # append fragment
                self._add_fragment(
                    syncmap=syncmap,
                    identifier=identifier,
                    lines=lines,
                    begin=begin,
                    end=end
                )

                # increase cue index
                cue_index += 1

    def format(self, syncmap):
        msg = []
        if self.header_string is not None:
            msg.append(self.header_string)
            msg.append(u"")
        for i, fragment in enumerate(syncmap.fragments, 1):
            text = fragment.text_fragment
            if self.cue_has_identifier or self.cue_has_optional_identifier:
                msg.append(u"%d" % i)
            msg.append(u"%s%s%s" % (
                self.format_time_function(fragment.begin),
                self.time_values_separator,
                self.format_time_function(fragment.end),
            ))
            lines = self.line_break_symbol.join(text.lines)
            msg.append(lines)
            msg.append(u"")
        if self.footer_string is not None:
            msg.append(self.footer_string)
        else:
            msg.append(u"")
        return u"\n".join(msg)
