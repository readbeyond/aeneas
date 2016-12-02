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

from aeneas.syncmap.smfgxml import SyncMapFormatGenericXML
import aeneas.globalfunctions as gf


class SyncMapFormatXMLLegacy(SyncMapFormatGenericXML):
    """
    Handler for XML (legacy) I/O format. Deprecated.
    """

    TAG = u"SyncMapFormatXMLLegacy"

    DEFAULT = "xml_legacy"

    def parse(self, input_text, syncmap):
        from lxml import etree
        root = etree.fromstring(gf.safe_bytes(input_text))
        for frag in root:
            for child in frag:
                if child.tag == "identifier":
                    identifier = gf.safe_unicode(child.text)
                elif child.tag == "start":
                    begin = gf.time_from_ssmmm(child.text)
                elif child.tag == "end":
                    end = gf.time_from_ssmmm(child.text)
            # TODO read text from additional text_file?
            self._add_fragment(
                syncmap=syncmap,
                identifier=identifier,
                lines=[u""],
                begin=begin,
                end=end
            )

    def format(self, syncmap):
        msg = []
        msg.append(u"<?xml version=\"1.0\" encoding=\"UTF-8\" ?>")
        msg.append(u"<map>")
        for fragment in syncmap.fragments:
            msg.append(u" <fragment>")
            msg.append(u"  <identifier>%s</identifier>" % fragment.text_fragment.identifier)
            msg.append(u"  <start>%s</start>" % gf.time_to_ssmmm(fragment.begin))
            msg.append(u"  <end>%s</end>" % gf.time_to_ssmmm(fragment.end))
            msg.append(u" </fragment>")
        msg.append(u"</map>")
        return u"\n".join(msg)
