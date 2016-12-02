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


class SyncMapFormatXML(SyncMapFormatGenericXML):
    """
    Handler for XML I/O format.
    """

    TAG = u"SyncMapFormatXML"

    DEFAULT = "xml"

    def parse(self, input_text, syncmap):
        from lxml import etree
        root = etree.fromstring(gf.safe_bytes(input_text))
        for frag in root:
            identifier = gf.safe_unicode(frag.get("id"))
            begin = gf.time_from_ssmmm(frag.get("begin"))
            end = gf.time_from_ssmmm(frag.get("end"))
            lines = []
            for child in frag:
                if child.tag == "line":
                    lines.append(gf.safe_unicode(child.text))
            self._add_fragment(
                syncmap=syncmap,
                identifier=identifier,
                lines=lines,
                begin=begin,
                end=end
            )

    def format(self, syncmap):
        from lxml import etree

        def visit_children(node, parent_elem):
            """ Recursively visit the fragments_tree """
            for child in node.children_not_empty:
                fragment = child.value
                fragment_elem = etree.SubElement(parent_elem, "fragment")
                fragment_elem.attrib["id"] = fragment.text_fragment.identifier
                fragment_elem.attrib["begin"] = gf.time_to_ssmmm(fragment.begin)
                fragment_elem.attrib["end"] = gf.time_to_ssmmm(fragment.end)
                for line in fragment.text_fragment.lines:
                    line_elem = etree.SubElement(fragment_elem, "line")
                    line_elem.text = line
                children_elem = etree.SubElement(fragment_elem, "children")
                visit_children(child, children_elem)

        map_elem = etree.Element("map")
        visit_children(syncmap.fragments_tree, map_elem)
        return self._tree_to_string(map_elem)
