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

from aeneas.logger import Loggable
from aeneas.syncmap.fragment import SyncMapFragment
from aeneas.textfile import TextFragment


class SyncMapFormatBase(Loggable):
    """
    Base class for I/O handlers.
    """

    TAG = u"SyncMapFormatBase"

    def __init__(self, variant=None, parameters=None, rconf=None, logger=None):
        """
        TBW

        :param variant: the code of the format variant to read or write
        :type variant: string
        :param parameters: additional parameters to read or write
        :type paramenters: ``None`` or ``dict``
        """
        super(SyncMapFormatBase, self).__init__(rconf=rconf, logger=logger)
        self.variant = variant
        self.parameters = parameters

    @classmethod
    def _add_fragment(cls, syncmap, identifier, lines, begin, end, language=None):
        """
        Add a new fragment to ``syncmap``.

        :param syncmap: the syncmap to append to
        :type syncmap: :class:`~aeneas.syncmap.SyncMap`
        :param identifier: the identifier
        :type identifier: string
        :param lines: the lines of the text
        :type lines: list of string
        :param begin: the begin time
        :type begin: :class:`~aeneas.exacttiming.TimeValue`
        :param end: the end time
        :type end: :class:`~aeneas.exacttiming.TimeValue`
        :param language: the language
        :type language: string
        """
        syncmap.add_fragment(
            SyncMapFragment(
                text_fragment=TextFragment(
                    identifier=identifier,
                    lines=lines,
                    language=language
                ),
                begin=begin,
                end=end
            )
        )

    def parse(self, input_text, syncmap):
        """
        Parse the given ``input_text`` and
        append the extracted fragments to ``syncmap``.

        :param input_text: the input text as a Unicode string (read from file)
        :type input_text: string
        :param syncmap: the syncmap to append to
        :type syncmap: :class:`~aeneas.syncmap.SyncMap`
        """
        self.log_exc(u"%s is abstract and cannot be called directly" % (self.TAG), None, True, NotImplementedError)

    def format(self, syncmap):
        """
        Format the given ``syncmap`` as a Unicode string.

        :param syncmap: the syncmap to output
        :type syncmap: :class:`~aeneas.syncmap.SyncMap`
        :rtype: string
        """
        self.log_exc(u"%s is abstract and cannot be called directly" % (self.TAG), None, True, NotImplementedError)
