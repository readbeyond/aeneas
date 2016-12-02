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

from aeneas.syncmap.fragment import SyncMapFragment
from aeneas.syncmap.smfgxml import SyncMapFormatGenericXML
from aeneas.textfile import TextFragment
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf


class SyncMapFormatEAF(SyncMapFormatGenericXML):
    """
    Handler for ELAN I/O format (EAF).
    """

    TAG = u"SyncMapFormatEAF"

    DEFAULT = "eaf"

    def parse(self, input_text, syncmap):
        from lxml import etree
        # namespaces
        xsi = "http://www.w3.org/2001/XMLSchema-instance"
        ns_map = {"xsi": xsi}
        # get root
        root = etree.fromstring(gf.safe_bytes(input_text))
        # get time slots
        time_slots = dict()
        for ts in root.iter("TIME_SLOT"):
            time_slots[ts.get("TIME_SLOT_ID")] = gf.time_from_ssmmm(ts.get("TIME_VALUE")) / 1000
        # parse annotations
        for alignable in root.iter("ALIGNABLE_ANNOTATION"):
            identifier = gf.safe_unicode(alignable.get("ANNOTATION_ID"))
            begin = time_slots[alignable.get("TIME_SLOT_REF1")]
            end = time_slots[alignable.get("TIME_SLOT_REF2")]
            lines = []
            for value in alignable.iter("ANNOTATION_VALUE"):
                lines.append(gf.safe_unicode(value.text))
            self._add_fragment(
                syncmap=syncmap,
                identifier=identifier,
                lines=lines,
                begin=begin,
                end=end
            )

    def format(self, syncmap):
        from lxml import etree
        # namespaces
        xsi = "http://www.w3.org/2001/XMLSchema-instance"
        ns_map = {"xsi": xsi}
        # build doc
        doc = etree.Element("ANNOTATION_DOCUMENT", nsmap=ns_map)
        doc.attrib["{%s}noNamespaceSchemaLocation" % xsi] = "http://www.mpi.nl/tools/elan/EAFv2.8.xsd"
        doc.attrib["AUTHOR"] = "aeneas"
        doc.attrib["DATE"] = gf.datetime_string(time_zone=True)
        doc.attrib["FORMAT"] = "2.8"
        doc.attrib["VERSION"] = "2.8"
        # header
        header = etree.SubElement(doc, "HEADER")
        header.attrib["MEDIA_FILE"] = ""
        header.attrib["TIME_UNITS"] = "milliseconds"
        if (self.parameters is not None) and (gc.PPN_TASK_OS_FILE_EAF_AUDIO_REF in self.parameters) and (self.parameters[gc.PPN_TASK_OS_FILE_EAF_AUDIO_REF] is not None):
            media = etree.SubElement(header, "MEDIA_DESCRIPTOR")
            media.attrib["MEDIA_URL"] = self.parameters[gc.PPN_TASK_OS_FILE_EAF_AUDIO_REF]
            media.attrib["MIME_TYPE"] = gf.mimetype_from_path(self.parameters[gc.PPN_TASK_OS_FILE_EAF_AUDIO_REF])
        # time order
        time_order = etree.SubElement(doc, "TIME_ORDER")
        # tier
        tier = etree.SubElement(doc, "TIER")
        tier.attrib["LINGUISTIC_TYPE_REF"] = "utterance"
        tier.attrib["TIER_ID"] = "tier1"
        for i, fragment in enumerate(syncmap.fragments, 1):
            # time slots
            begin_id = "ts%06db" % i
            end_id = "ts%06de" % i
            slot = etree.SubElement(time_order, "TIME_SLOT")
            slot.attrib["TIME_SLOT_ID"] = begin_id
            slot.attrib["TIME_VALUE"] = "%d" % (fragment.begin * 1000)
            slot = etree.SubElement(time_order, "TIME_SLOT")
            slot.attrib["TIME_SLOT_ID"] = end_id
            slot.attrib["TIME_VALUE"] = "%d" % (fragment.end * 1000)
            # annotation
            annotation = etree.SubElement(tier, "ANNOTATION")
            alignable = etree.SubElement(annotation, "ALIGNABLE_ANNOTATION")
            alignable.attrib["ANNOTATION_ID"] = fragment.text_fragment.identifier
            alignable.attrib["TIME_SLOT_REF1"] = begin_id
            alignable.attrib["TIME_SLOT_REF2"] = end_id
            value = etree.SubElement(alignable, "ANNOTATION_VALUE")
            value.text = u" ".join(fragment.text_fragment.lines)
        # linguistic type
        ling = etree.SubElement(doc, "LINGUISTIC_TYPE")
        ling.attrib["LINGUISTIC_TYPE_ID"] = "utterance"
        ling.attrib["TIME_ALIGNABLE"] = "true"
        # write tree
        return self._tree_to_string(doc)
