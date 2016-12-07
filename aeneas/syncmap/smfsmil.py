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

from aeneas.syncmap.missingparametererror import SyncMapMissingParameterError
from aeneas.syncmap.smfgxml import SyncMapFormatGenericXML
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf


class SyncMapFormatSMIL(SyncMapFormatGenericXML):
    """
    Handler for SMIL for EPUB 3 I/O format.
    """

    TAG = u"SyncMapFormatSMIL"

    DEFAULT = "smil"

    HUMAN = "smilh"

    MACHINE = "smilm"

    MACHINE_ALIASES = [MACHINE]

    def __init__(self, variant=DEFAULT, parameters=None, rconf=None, logger=None):
        super(SyncMapFormatSMIL, self).__init__(variant=variant, parameters=parameters, rconf=rconf, logger=logger)
        if self.variant in self.MACHINE_ALIASES:
            self.format_time_function = gf.time_to_ssmmm
        else:
            self.format_time_function = gf.time_to_hhmmssmmm

    def parse(self, input_text, syncmap):
        """
        Read from SMIL file.

        Limitations:
        1. parses only ``<par>`` elements, in order
        2. timings must have ``hh:mm:ss.mmm`` or ``ss.mmm`` format (autodetected)
        3. both ``clipBegin`` and ``clipEnd`` attributes of ``<audio>`` must be populated
        """
        from lxml import etree
        smil_ns = "{http://www.w3.org/ns/SMIL}"
        root = etree.fromstring(gf.safe_bytes(input_text))
        for par in root.iter(smil_ns + "par"):
            for child in par:
                if child.tag == (smil_ns + "text"):
                    identifier = gf.safe_unicode(gf.split_url(child.get("src"))[1])
                elif child.tag == (smil_ns + "audio"):
                    begin_text = child.get("clipBegin")
                    if ":" in begin_text:
                        begin = gf.time_from_hhmmssmmm(begin_text)
                    else:
                        begin = gf.time_from_ssmmm(begin_text)
                    end_text = child.get("clipEnd")
                    if ":" in end_text:
                        end = gf.time_from_hhmmssmmm(end_text)
                    else:
                        end = gf.time_from_ssmmm(end_text)
            # TODO read text from additional text_file?
            self._add_fragment(
                syncmap=syncmap,
                identifier=identifier,
                lines=[u""],
                begin=begin,
                end=end
            )

    def format(self, syncmap):
        # check for required parameters
        for key in [
                gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF,
                gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF
        ]:
            if gf.safe_get(self.parameters, key, None) is None:
                self.log_exc(u"Parameter %s must be specified for format %s" % (key, self.variant), None, True, SyncMapMissingParameterError)

        from lxml import etree
        # we are sure we have them
        text_ref = self.parameters[gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF]
        audio_ref = self.parameters[gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF]

        # namespaces
        smil_ns = "http://www.w3.org/ns/SMIL"
        epub_ns = "http://www.idpf.org/2007/ops"
        ns_map = {None: smil_ns, "epub": epub_ns}

        # build tree
        smil_elem = etree.Element("{%s}smil" % smil_ns, nsmap=ns_map)
        smil_elem.attrib["version"] = "3.0"
        body_elem = etree.SubElement(smil_elem, "{%s}body" % smil_ns)
        seq_elem = etree.SubElement(body_elem, "{%s}seq" % smil_ns)
        seq_elem.attrib["id"] = u"seq000001"
        seq_elem.attrib["{%s}textref" % epub_ns] = text_ref

        if syncmap.is_single_level:
            # single level
            for i, fragment in enumerate(syncmap.fragments, 1):
                text = fragment.text_fragment
                par_elem = etree.SubElement(seq_elem, "{%s}par" % smil_ns)
                par_elem.attrib["id"] = "par%06d" % (i)
                text_elem = etree.SubElement(par_elem, "{%s}text" % smil_ns)
                text_elem.attrib["src"] = "%s#%s" % (text_ref, text.identifier)
                audio_elem = etree.SubElement(par_elem, "{%s}audio" % smil_ns)
                audio_elem.attrib["src"] = audio_ref
                audio_elem.attrib["clipBegin"] = self.format_time_function(fragment.begin)
                audio_elem.attrib["clipEnd"] = self.format_time_function(fragment.end)
        else:
            # TODO support generic multiple levels
            # multiple levels
            for par_index, par_child in enumerate(syncmap.fragments_tree.children_not_empty, 1):
                par_seq_elem = etree.SubElement(seq_elem, "{%s}seq" % smil_ns)
                # COMMENTED par_seq_elem.attrib["id"] = "p%06d" % (par_index)
                par_seq_elem.attrib["{%s}type" % epub_ns] = "paragraph"
                par_seq_elem.attrib["{%s}textref" % epub_ns] = text_ref + "#" + par_child.value.text_fragment.identifier
                for sen_index, sen_child in enumerate(par_child.children_not_empty, 1):
                    sen_seq_elem = etree.SubElement(par_seq_elem, "{%s}seq" % smil_ns)
                    # COMMENTED sen_seq_elem.attrib["id"] = par_seq_elem.attrib["id"] + "s%06d" % (sen_index)
                    sen_seq_elem.attrib["{%s}type" % epub_ns] = "sentence"
                    sen_seq_elem.attrib["{%s}textref" % epub_ns] = text_ref + "#" + sen_child.value.text_fragment.identifier
                    for wor_index, wor_child in enumerate(sen_child.children_not_empty, 1):
                        fragment = wor_child.value
                        text = fragment.text_fragment
                        wor_seq_elem = etree.SubElement(sen_seq_elem, "{%s}seq" % smil_ns)
                        # COMMENTED wor_seq_elem.attrib["id"] = sen_seq_elem.attrib["id"] + "w%06d" % (wor_index)
                        wor_seq_elem.attrib["{%s}type" % epub_ns] = "word"
                        wor_seq_elem.attrib["{%s}textref" % epub_ns] = text_ref + "#" + text.identifier
                        wor_par_elem = etree.SubElement(wor_seq_elem, "{%s}par" % smil_ns)
                        text_elem = etree.SubElement(wor_par_elem, "{%s}text" % smil_ns)
                        text_elem.attrib["src"] = "%s#%s" % (text_ref, text.identifier)
                        audio_elem = etree.SubElement(wor_par_elem, "{%s}audio" % smil_ns)
                        audio_elem.attrib["src"] = audio_ref
                        audio_elem.attrib["clipBegin"] = self.format_time_function(fragment.begin)
                        audio_elem.attrib["clipEnd"] = self.format_time_function(fragment.end)
        return self._tree_to_string(smil_elem, xml_declaration=False)
