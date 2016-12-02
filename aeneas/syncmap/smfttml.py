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


class SyncMapFormatTTML(SyncMapFormatGenericXML):
    """
    Handler for TTML I/O format.
    """

    TAG = u"SyncMapFormatTTML"

    TTML = "ttml"

    DFXP = "dfxp"

    DEFAULT = TTML

    def parse(self, input_text, syncmap):
        from lxml import etree
        ttml_ns = "{http://www.w3.org/ns/ttml}"
        xml_ns = "{http://www.w3.org/XML/1998/namespace}"
        root = etree.fromstring(gf.safe_bytes(input_text))
        language = root.get(xml_ns + "lang")
        for elem in root.iter(ttml_ns + "p"):
            identifier = gf.safe_unicode(elem.get(xml_ns + "id"))
            begin = gf.time_from_ttml(elem.get("begin"))
            end = gf.time_from_ttml(elem.get("end"))
            fragment_lines = self._get_lines_from_node_text(elem)
            self._add_fragment(
                syncmap=syncmap,
                identifier=identifier,
                language=language,
                lines=fragment_lines,
                begin=begin,
                end=end
            )

    def format(self, syncmap):
        from lxml import etree
        # get language
        language = None
        if (self.parameters is not None) and ("language" in self.parameters):
            language = self.parameters["language"]
        elif len(syncmap.fragments) > 0:
            language = syncmap.fragments[0].text_fragment.language
        if language is None:
            language = ""

        # namespaces
        ttml_ns = "http://www.w3.org/ns/ttml"
        xml_ns = "http://www.w3.org/XML/1998/namespace"
        ns_map = {None: ttml_ns}

        # build tree
        tt_elem = etree.Element("{%s}tt" % ttml_ns, nsmap=ns_map)
        tt_elem.attrib["{%s}lang" % xml_ns] = language
        # TODO add metadata from parameters here?
        # COMMENTED head_elem = etree.SubElement(tt_elem, "{%s}head" % ttml_ns)
        body_elem = etree.SubElement(tt_elem, "{%s}body" % ttml_ns)
        div_elem = etree.SubElement(body_elem, "{%s}div" % ttml_ns)

        if syncmap.is_single_level:
            # single level
            for fragment in syncmap.fragments:
                text = fragment.text_fragment
                p_string = u"<p xml:id=\"%s\" begin=\"%s\" end=\"%s\">%s</p>" % (
                    text.identifier,
                    gf.time_to_ttml(fragment.begin),
                    gf.time_to_ttml(fragment.end),
                    u"<br/>".join(text.lines)
                )
                p_elem = etree.fromstring(p_string)
                div_elem.append(p_elem)
        else:
            # TODO support generic multiple levels
            # multiple levels
            for par_child in syncmap.fragments_tree.children_not_empty:
                text = par_child.value.text_fragment
                p_elem = etree.SubElement(div_elem, "{%s}p" % ttml_ns)
                p_elem.attrib["id"] = text.identifier
                for sen_child in par_child.children_not_empty:
                    text = sen_child.value.text_fragment
                    sen_span_elem = etree.SubElement(p_elem, "{%s}span" % ttml_ns)
                    sen_span_elem.attrib["id"] = text.identifier
                    for wor_child in sen_child.children_not_empty:
                        fragment = wor_child.value
                        wor_span_elem = etree.SubElement(sen_span_elem, "{%s}span" % ttml_ns)
                        wor_span_elem.attrib["id"] = fragment.text_fragment.identifier
                        wor_span_elem.attrib["begin"] = gf.time_to_ttml(fragment.begin)
                        wor_span_elem.attrib["end"] = gf.time_to_ttml(fragment.end)
                        wor_span_elem.text = u"<br/>".join(fragment.text_fragment.lines)
        # write tree
        return self._tree_to_string(tt_elem)
