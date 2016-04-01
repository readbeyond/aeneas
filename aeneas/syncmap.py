#!/usr/bin/env python
# coding=utf-8

"""
A synchronization map, or sync map,
is a map from text fragments to time intervals.

This module contains the following classes:

* :class:`~aeneas.syncmap.SyncMap`, represents a sync map as a tree of sync map fragments;
* :class:`~aeneas.syncmap.SyncMapFormat`, an enumeration of the supported output formats;
* :class:`~aeneas.syncmap.SyncMapFragment`, connects a text fragment with a ``begin`` and ``end`` time values;
* :class:`~aeneas.syncmap.SyncMapHeadTailFormat`, an enumeration of the supported formats for the sync map head/tail;
* :class:`~aeneas.syncmap.SyncMapMissingParameterError`, an error raised when reading sync maps from file.

.. warning:: This module is likely to be refactored in a future version
"""

from __future__ import absolute_import
from __future__ import print_function
from functools import partial
from itertools import chain
from lxml import etree
import io
import json
import os

from aeneas.logger import Loggable
from aeneas.textfile import TextFragment
from aeneas.timevalue import Decimal
from aeneas.timevalue import TimeValue
from aeneas.tree import Tree
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.5.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class SyncMapMissingParameterError(Exception):
    """
    Error raised when a parameter implied by the SyncMapFormat is missing.
    """
    pass



class SyncMapFormat(object):
    """
    Enumeration of the supported output formats
    for a synchronization map.
    """

    AUD = "aud"
    """
    Alias for AUDM
    """

    AUDH = "audh"
    """
    Tab-separated plain text,
    with human-readable time values
    and fragment text::

        00:00:00.000   00:00:01.234   Text of fragment 1
        00:00:01.234   00:00:05.678   Text of fragment 2
        00:00:05.678   00:00:07.890   Text of fragment 3

    * Multiple levels: no
    * Multiple lines: no

    .. versionadded:: 1.5.0
    """

    AUDM = "audm"
    """
    Tab-separated plain text,
    with machine-readable time values
    and fragment text,
    compatible with ``Audacity``::

        0.000   1.234   Text fragment 1
        1.234   5.678   Text fragment 2
        5.678   7.890   Text fragment 3

    * Multiple levels: no
    * Multiple lines: no

    .. versionadded:: 1.5.0
    """

    CSV = "csv"
    """
    Alias for CSVM
    """

    CSVH = "csvh"
    """
    Comma-separated values (CSV),
    with human-readable time values::

        f001,00:00:00.000,00:00:01.234,First fragment text
        f002,00:00:01.234,00:00:05.678,Second fragment text
        f003,00:00:05.678,00:00:07.890,Third fragment text

    * Multiple levels: no
    * Multiple lines: no

    .. versionadded:: 1.0.4
    """

    CSVM = "csvm"
    """
    Comma-separated values (CSV),
    with machine-readable time values::

        f001,0.000,1.234,First fragment text
        f002,1.234,5.678,Second fragment text
        f003,5.678,7.890,Third fragment text

    * Multiple levels: no
    * Multiple lines: no

    .. versionadded:: 1.2.0
    """

    DFXP = "dfxp"
    """
    Alias for TTML

    .. versionadded:: 1.4.1
    """

    EAF = "eaf"
    """
    ELAN EAF::

        <?xml version="1.0" encoding="UTF-8"?>
        <ANNOTATION_DOCUMENT AUTHOR="aeneas" DATE="2016-01-01T00:00:00+00:00" FORMAT="2.8" VERSION="2.8" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv2.8.xsd">
            <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds"></HEADER>
            <TIME_ORDER>
                <TIME_SLOT TIME_SLOT_ID="ts001b" TIME_VALUE="0"/>
                <TIME_SLOT TIME_SLOT_ID="ts001e" TIME_VALUE="1234"/>
                <TIME_SLOT TIME_SLOT_ID="ts002b" TIME_VALUE="1234"/>
                <TIME_SLOT TIME_SLOT_ID="ts002e" TIME_VALUE="5678"/>
                <TIME_SLOT TIME_SLOT_ID="ts003b" TIME_VALUE="5678"/>
                <TIME_SLOT TIME_SLOT_ID="ts003e" TIME_VALUE="7890"/>
            </TIME_ORDER>
            <TIER LINGUISTIC_TYPE_REF="utterance" TIER_ID="tier1">
                <ANNOTATION>
                    <ALIGNABLE_ANNOTATION ANNOTATION_ID="f001" TIME_SLOT_REF1="ts001b" TIME_SLOT_REF2="ts001e">
                        <ANNOTATION_VALUE>First fragment text</ANNOTATION_VALUE>
                    </ALIGNABLE_ANNOTATION>
                </ANNOTATION>
                <ANNOTATION>
                    <ALIGNABLE_ANNOTATION ANNOTATION_ID="f002" TIME_SLOT_REF1="ts002b" TIME_SLOT_REF2="ts002e">
                        <ANNOTATION_VALUE>First fragment text</ANNOTATION_VALUE>
                    </ALIGNABLE_ANNOTATION>
                </ANNOTATION>
                <ANNOTATION>
                    <ALIGNABLE_ANNOTATION ANNOTATION_ID="f003" TIME_SLOT_REF1="ts003b" TIME_SLOT_REF2="ts003e">
                        <ANNOTATION_VALUE>First fragment text</ANNOTATION_VALUE>
                    </ALIGNABLE_ANNOTATION>
                </ANNOTATION>
            </TIER>
            <LINGUISTIC_TYPE LINGUISTIC_TYPE_ID="utterance" TIME_ALIGNABLE="true"/>
        </ANNOTATION_DOCUMENT>

    * Multiple levels: no
    * Multiple lines: no

    .. versionadded:: 1.5.0
    """

    JSON = "json"
    """
    JSON::

        {
         "fragments": [
          {
           "id": "f001",
           "language": "en",
           "begin": 0.000,
           "end": 1.234,
           "children": [],
           "lines": [
            "First fragment text"
           ]
          },
          {
           "id": "f002",
           "language": "en",
           "begin": 1.234,
           "end": 5.678,
           "children": [],
           "lines": [
            "Second fragment text",
            "Second line of second fragment"
           ]
          },
          {
           "id": "f003",
           "language": "en",
           "begin": 5.678,
           "end": 7.890,
           "children": [],
           "lines": [
            "Third fragment text",
            "Second line of third fragment"
           ]
          }
         ]
        }

    * Multiple levels: yes (output only)
    * Multiple lines: yes

    .. versionadded:: 1.2.0
    """

    RBSE = "rbse"
    """
    JSON compatible with ``rb_smil_emulator.js``::

        {
         "smil_ids": [
          "f001",
          "f002",
          "f003",
         ],
         "smil_data": [
          { "id": "f001", "begin": 0.000, "end": 1.234 },
          { "id": "f002", "begin": 1.234, "end": 5.678 },
          { "id": "f003", "begin": 5.678, "end": 7.890 }
         ]
        }

    * Multiple levels: no
    * Multiple lines: no

    Deprecated, it will be removed in v2.0.0.

    .. deprecated:: 1.5.0

    .. versionadded:: 1.2.0
    """

    SBV = "sbv"
    """
    SubViewer (SBV/SUB) caption/subtitle format,
    with multiple lines per fragment are separated by a newline character::

        [SUBTITLE]
        00:00:00.000,00:00:01.234
        First fragment text

        00:00:01.234,00:00:05.678
        Second fragment text
        Second line of second fragment

        00:00:05.678,00:00:07.890
        Third fragment text
        Second line of third fragment

    * Multiple levels: no
    * Multiple lines: yes

    .. versionadded:: 1.4.1
    """

    SMIL = "smil"
    """
    Alias for SMILH
    """

    SMILH = "smilh"
    """
    SMIL (as in the EPUB 3 Media Overlay specification),
    with human-readable time values::

        <smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">
         <body>
          <seq id="seq000001" epub:textref="p001.xhtml">
           <par id="par000001">
            <text src="p001.xhtml#f001"/>
            <audio clipBegin="00:00:00.000" clipEnd="00:00:01.234" src="../Audio/p001.mp3"/>
           </par>
           <par id="par000002">
            <text src="p001.xhtml#f002"/>
            <audio clipBegin="00:00:01.234" clipEnd="00:00:05.678" src="../Audio/p001.mp3"/>
           </par>
           <par id="par000003">
            <text src="p001.xhtml#f003"/>
            <audio clipBegin="00:00:05.678" clipEnd="00:00:07.890" src="../Audio/p001.mp3"/>
           </par>
          </seq>
         </body>
        </smil>

    * Multiple levels: yes (output only)
    * Multiple lines: no

    .. versionadded:: 1.2.0
    """

    SMILM = "smilm"
    """
    SMIL (as in the EPUB 3 Media Overlay specification),
    with machine-readable time values::

        <smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">
         <body>
          <seq id="seq000001" epub:textref="p001.xhtml">
           <par id="par000001">
            <text src="p001.xhtml#f001"/>
            <audio clipBegin="0.000" clipEnd="1.234" src="../Audio/p001.mp3"/>
           </par>
           <par id="par000002">
            <text src="p001.xhtml#f002"/>
            <audio clipBegin="1.234" clipEnd="5.678" src="../Audio/p001.mp3"/>
           </par>
           <par id="par000003">
            <text src="p001.xhtml#f003"/>
            <audio clipBegin="5.678" clipEnd="7.890" src="../Audio/p001.mp3"/>
           </par>
          </seq>
         </body>
        </smil>

    * Multiple levels: yes (output only)
    * Multiple lines: no

    .. versionadded:: 1.2.0
    """

    SRT = "srt"
    """
    SubRip (SRT) caption/subtitle format
    (it might have multiple lines per fragment)::

        1
        00:00:00,000 --> 00:00:01,234
        First fragment text

        2
        00:00:01,234 --> 00:00:05,678
        Second fragment text
        Second line of second fragment

        3
        00:00:05,678 --> 00:00:07,890
        Third fragment text
        Second line of third fragment

    * Multiple levels: no
    * Multiple lines: yes
    """

    SSV = "ssv"
    """
    Space-separated plain text::

        0.000 1.234 f001 "First fragment text"
        1.234 5.678 f002 "Second fragment text"
        5.678 7.890 f003 "Third fragment text"

    * Multiple levels: no
    * Multiple lines: no

    .. versionadded:: 1.0.4
    """

    SSVH = "ssvh"
    """
    Space-separated plain text,
    with human-readable time values::

        00:00:00.000 00:00:01.234 f001 "First fragment text"
        00:00:01.234 00:00:05.678 f002 "Second fragment text"
        00:00:05.678 00:00:07.890 f003 "Third fragment text"

    * Multiple levels: no
    * Multiple lines: no

    .. versionadded:: 1.0.4
    """

    SSVM = "ssvm"
    """
    Space-separated plain text,
    with machine-readable time values::

        0.000 1.234 f001 "First fragment text"
        1.234 5.678 f002 "Second fragment text"
        5.678 7.890 f003 "Third fragment text"

    * Multiple levels: no
    * Multiple lines: no

    .. versionadded:: 1.2.0
    """

    SUB = "sub"
    """
    SubViewer (SBV/SUB) caption/subtitle format,
    with multiple lines per fragment are separated by [br]::

        [SUBTITLE]
        00:00:00.000,00:00:01.234
        First fragment text

        00:00:01.234,00:00:05.678
        Second fragment text[br]Second line of second fragment

        00:00:05.678,00:00:07.890
        Third fragment text[br]Second line of third fragment

    * Multiple levels: no
    * Multiple lines: yes

    .. versionadded:: 1.4.1
    """

    TAB = "tab"
    """
    Deprecated, it will be removed in v2.0.0. Use TSV instead.

    .. deprecated:: 1.0.3
    """

    TSV = "tsv"
    """
    Alias for TSVM
    """

    TSVH = "tsvh"
    """
    Tab-separated plain text,
    with human-readable time values::

        00:00:00.000   00:00:01.234   f001
        00:00:01.234   00:00:05.678   f002
        00:00:05.678   00:00:07.890   f003

    * Multiple levels: no
    * Multiple lines: no

    .. versionadded:: 1.0.4
    """

    TSVM = "tsvm"
    """
    Tab-separated plain text,
    with machine-readable time values,
    compatible with ``Audacity``::

        0.000   1.234   f001
        1.234   5.678   f002
        5.678   7.890   f003

    * Multiple levels: no
    * Multiple lines: no

    .. versionadded:: 1.2.0
    """

    TTML = "ttml"
    """
    TTML caption/subtitle format
    (it might have multiple lines per fragment)::

        <?xml version="1.0" encoding="UTF-8" ?>
        <tt xmlns="http://www.w3.org/ns/ttml">
         <body>
          <div>
           <p xml:id="f001" begin="0.000" end="1.234">
            First fragment text
           </p>
           <p xml:id="f002" begin="1.234" end="5.678">
            Second fragment text<br/>Second line of second fragment
           </p>
           <p xml:id="f003" begin="5.678" end="7.890">
            Third fragment text<br/>Second line of third fragment
           </p>
          </div>
         </body>
        </tt>

    * Multiple levels: yes (output only)
    * Multiple lines: yes
    """

    TXT = "txt"
    """
    Alias for TXTM
    """

    TXTH = "txth"
    """
    Space-separated plain text
    with human-readable time values::

        f001 00:00:00.000 00:00:01.234 "First fragment text"
        f002 00:00:01.234 00:00:05.678 "Second fragment text"
        f003 00:00:05.678 00:00:07.890 "Third fragment text"

    * Multiple levels: no
    * Multiple lines: no

    .. versionadded:: 1.0.4
    """

    TXTM = "txtm"
    """
    Space-separated plain text,
    with machine-readable time values,
    compatible with ``SonicVisualizer``::

        f001 0.000 1.234 "First fragment text"
        f002 1.234 5.678 "Second fragment text"
        f003 5.678 7.890 "Third fragment text"

    * Multiple levels: no
    * Multiple lines: no

    .. versionadded:: 1.2.0
    """

    VTT = "vtt"
    """
    WebVTT caption/subtitle format::

        WEBVTT

        1
        00:00:00.000 --> 00:00:01.234
        First fragment text

        2
        00:00:01.234 --> 00:00:05.678
        Second fragment text
        Second line of second fragment

        3
        00:00:05.678 --> 00:00:07.890
        Third fragment text
        Second line of third fragment

    * Multiple levels: no
    * Multiple lines: yes
    """

    XML = "xml"
    """
    XML::

        <?xml version="1.0" encoding="UTF-8" ?>
        <map>
         <fragment id="f001" begin="0.000" end="1.234">
          <line>First fragment text</line>
          <children></children>
         </fragment>
         <fragment id="f002" begin="1.234" end="5.678">
          <line>Second fragment text</line>
          <line>Second line of second fragment</line>
          <children></children>
         </fragment>
         <fragment id="f003" begin="5.678" end="7.890">
          <line>Third fragment text</line>
          <line>Second line of third fragment</line>
          <children></children>
         </fragment>
        </map>

    * Multiple levels: yes (output only)
    * Multiple lines: yes
    """

    XML_LEGACY = "xml_legacy"
    """
    XML, legacy format::

        <?xml version="1.0" encoding="UTF-8" ?>
        <map>
         <fragment>
          <identifier>f001</identifier>
          <start>0.000</start>
          <end>1.234</end>
         </fragment>
         <fragment>
          <identifier>f002</identifier>
          <start>1.234</start>
          <end>5.678</end>
         </fragment>
         <fragment>
          <identifier>f003</identifier>
          <start>5.678</start>
          <end>7.890</end>
         </fragment>
        </map>

    * Multiple levels: no
    * Multiple lines: no

    Deprecated, it will be removed in v2.0.0. Use XML instead.

    .. deprecated:: 1.2.0
    """

    ALLOWED_VALUES = [
        AUD,
        AUDH,
        AUDM,
        CSV,
        CSVH,
        CSVM,
        DFXP,
        EAF,
        JSON,
        RBSE,
        SBV,
        SMIL,
        SMILH,
        SMILM,
        SRT,
        SSV,
        SSVH,
        SSVM,
        SUB,
        TAB,
        TSV,
        TSVH,
        TSVM,
        TTML,
        TXT,
        TXTH,
        TXTM,
        VTT,
        XML,
        XML_LEGACY
    ]
    """ List of all the allowed values """



class SyncMap(Loggable):
    """
    A synchronization map, that is, a tree of
    :class:`~aeneas.syncmap.SyncMapFragment`
    objects.
    """

    FINETUNEAS_REPLACEMENTS = [
        ["<!-- AENEAS_REPLACE_COMMENT_BEGIN -->", "<!-- AENEAS_REPLACE_COMMENT_BEGIN"],
        ["<!-- AENEAS_REPLACE_COMMENT_END -->", "AENEAS_REPLACE_COMMENT_END -->"],
        ["<!-- AENEAS_REPLACE_UNCOMMENT_BEGIN", "<!-- AENEAS_REPLACE_UNCOMMENT_BEGIN -->"],
        ["AENEAS_REPLACE_UNCOMMENT_END -->", "<!-- AENEAS_REPLACE_UNCOMMENT_END -->"],
        ["// AENEAS_REPLACE_SHOW_ID", "showID = true;"],
        ["// AENEAS_REPLACE_ALIGN_TEXT", "alignText = \"left\""],
        ["// AENEAS_REPLACE_CONTINUOUS_PLAY", "continuousPlay = true;"],
        ["// AENEAS_REPLACE_TIME_FORMAT", "timeFormatHHMMSSmmm = true;"],
    ]
    FINETUNEAS_REPLACE_AUDIOFILEPATH = "// AENEAS_REPLACE_AUDIOFILEPATH"
    FINETUNEAS_REPLACE_FRAGMENTS = "// AENEAS_REPLACE_FRAGMENTS"
    FINETUNEAS_REPLACE_OUTPUT_FORMAT = "// AENEAS_REPLACE_OUTPUT_FORMAT"
    FINETUNEAS_REPLACE_SMIL_AUDIOREF = "// AENEAS_REPLACE_SMIL_AUDIOREF"
    FINETUNEAS_REPLACE_SMIL_PAGEREF = "// AENEAS_REPLACE_SMIL_PAGEREF"
    FINETUNEAS_ALLOWED_FORMATS = [
        "csv",
        "json",
        "smil",
        "srt",
        "ssv",
        "ttml",
        "tsv",
        "txt",
        "vtt",
        "xml"
    ]
    FINETUNEAS_PATH = "res/finetuneas.html"

    TAG = u"SyncMap"

    def __init__(self, rconf=None, logger=None):
        super(SyncMap, self).__init__(rconf=rconf, logger=logger)
        self.fragments_tree = Tree()

    def __len__(self):
        return len(self.fragments)

    def __unicode__(self):
        return u"\n".join([f.__unicode__() for f in self.fragments])

    def __str__(self):
        return gf.safe_str(self.__unicode__())

    @property
    def fragments_tree(self):
        """
        Return the current tree of fragments.

        :rtype: :class:`~aeneas.tree.Tree`
        """
        return self.__fragments_tree
    @fragments_tree.setter
    def fragments_tree(self, fragments_tree):
        self.__fragments_tree = fragments_tree

    @property
    def is_single_level(self):
        """
        Return ``True`` if the sync map
        has only one level, that is,
        if it is a list of fragments
        rather than a hierarchical tree.

        :rtype: bool
        """
        return self.fragments_tree.height <= 2

    @property
    def fragments(self):
        """
        The current list of sync map fragments
        which are the children of the root node
        of the sync map tree.

        :rtype: list of :class:`~aeneas.syncmap.SyncMapFragment`
        """
        return self.fragments_tree.vchildren_not_empty

    @property
    def json_string(self):
        """
        Return a JSON representation of the sync map.

        :rtype: string

        .. versionadded:: 1.3.1
        """
        def visit_children(node):
            """ Recursively visit the fragments_tree """
            output_fragments = []
            for child in node.children_not_empty:
                fragment = child.value
                text = fragment.text_fragment
                output_fragments.append({
                    "id" : text.identifier,
                    "language" : text.language,
                    "lines" : text.lines,
                    "begin" : gf.time_to_ssmmm(fragment.begin),
                    "end" : gf.time_to_ssmmm(fragment.end),
                    "children": visit_children(child)
                })
            return output_fragments
        output_fragments = visit_children(self.fragments_tree)
        return gf.safe_unicode(
            json.dumps({"fragments": output_fragments}, indent=1, sort_keys=True)
        )

    def add_fragment(self, fragment, as_last=True):
        """
        Add the given sync map fragment,
        as the first or last child of the root node
        of the sync map tree.

        :param fragment: the sync map fragment to be added
        :type  fragment: :class:`~aeneas.syncmap.SyncMapFragment`
        :param bool as_last: if ``True``, append fragment; otherwise prepend it
        :raises: TypeError: if ``fragment`` is ``None`` or
                            it is not an instance of :class:`~aeneas.syncmap.SyncMapFragment`
        """
        if not isinstance(fragment, SyncMapFragment):
            self.log_exc(u"fragment is not an instance of SyncMapFragment", None, True, TypeError)
        self.fragments_tree.add_child(Tree(value=fragment), as_last=as_last)

    def clear(self):
        """
        Clear the sync map, removing all the current fragments.
        """
        self.log(u"Clearing sync map")
        self.fragments_tree = Tree()

    def output_html_for_tuning(
            self,
            audio_file_path,
            output_file_path,
            parameters=None
    ):
        """
        Output an HTML file for fine tuning the sync map manually.

        :param string audio_file_path: the path to the associated audio file
        :param string output_file_path: the path to the output file to write
        :param dict parameters: additional parameters

        .. versionadded:: 1.3.1
        """
        if not gf.file_can_be_written(output_file_path):
            self.log_exc(u"Cannot output HTML file '%s'. Wrong permissions?" % (output_file_path), None, True, OSError)
        if parameters is None:
            parameters = {}
        audio_file_path_absolute = gf.fix_slash(os.path.abspath(audio_file_path))
        template_path_absolute = gf.absolute_path(self.FINETUNEAS_PATH, __file__)
        with io.open(template_path_absolute, "r", encoding="utf-8") as file_obj:
            template = file_obj.read()
        for repl in self.FINETUNEAS_REPLACEMENTS:
            template = template.replace(repl[0], repl[1])
        template = template.replace(
            self.FINETUNEAS_REPLACE_AUDIOFILEPATH,
            u"audioFilePath = \"file://%s\";" % audio_file_path_absolute
        )
        template = template.replace(
            self.FINETUNEAS_REPLACE_FRAGMENTS,
            u"fragments = (%s).fragments;" % self.json_string
        )
        if gc.PPN_TASK_OS_FILE_FORMAT in parameters:
            output_format = parameters[gc.PPN_TASK_OS_FILE_FORMAT]
            if output_format in self.FINETUNEAS_ALLOWED_FORMATS:
                template = template.replace(
                    self.FINETUNEAS_REPLACE_OUTPUT_FORMAT,
                    u"outputFormat = \"%s\";" % output_format
                )
                if output_format == "smil":
                    for key, placeholder, replacement in [
                            (
                                gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF,
                                self.FINETUNEAS_REPLACE_SMIL_AUDIOREF,
                                "audioref = \"%s\";"
                            ),
                            (
                                gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF,
                                self.FINETUNEAS_REPLACE_SMIL_PAGEREF,
                                "pageref = \"%s\";"
                            ),
                    ]:
                        if key in parameters:
                            template = template.replace(
                                placeholder,
                                replacement % parameters[key]
                            )
        with io.open(output_file_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(template)

    def read(self, sync_map_format, input_file_path, parameters=None):
        """
        Read sync map fragments from the given file in the specified format,
        and add them the current (this) sync map.

        Return ``True`` if the call succeeded,
        ``False`` if an error occurred.

        :param sync_map_format: the format of the sync map
        :type  sync_map_format: :class:`~aeneas.syncmap.SyncMapFormat`
        :param string input_file_path: the path to the input file to read
        :param dict parameters: additional parameters (e.g., for ``SMIL`` input)
        :raises: ValueError: if ``sync_map_format`` is ``None`` or it is not an allowed value
        :raises: OSError: if ``input_file_path`` does not exist
        """
        map_read_function = {
            SyncMapFormat.AUD: partial(self._read_aud, parse_time=gf.time_from_ssmmm),
            SyncMapFormat.AUDH: partial(self._read_aud, parse_time=gf.time_from_hhmmssmmm),
            SyncMapFormat.AUDM: partial(self._read_aud, parse_time=gf.time_from_ssmmm),
            SyncMapFormat.CSV: partial(self._read_csv, parse_time=gf.time_from_ssmmm),
            SyncMapFormat.CSVH: partial(self._read_csv, parse_time=gf.time_from_hhmmssmmm),
            SyncMapFormat.CSVM: partial(self._read_csv, parse_time=gf.time_from_ssmmm),
            SyncMapFormat.DFXP: self._read_ttml,
            SyncMapFormat.EAF: self._read_eaf,
            SyncMapFormat.JSON: self._read_json,
            SyncMapFormat.RBSE: self._read_rbse,
            SyncMapFormat.SBV: partial(self._read_sub, use_newline=True),
            SyncMapFormat.SMIL: self._read_smil,
            SyncMapFormat.SMILH: self._read_smil,
            SyncMapFormat.SMILM: self._read_smil,
            SyncMapFormat.SRT: self._read_srt,
            SyncMapFormat.SSV: partial(self._read_ssv, parse_time=gf.time_from_ssmmm),
            SyncMapFormat.SSVH: partial(self._read_ssv, parse_time=gf.time_from_hhmmssmmm),
            SyncMapFormat.SSVM: partial(self._read_ssv, parse_time=gf.time_from_ssmmm),
            SyncMapFormat.SUB: partial(self._read_sub, use_newline=False),
            SyncMapFormat.TAB: partial(self._read_tsv, parse_time=gf.time_from_ssmmm),
            SyncMapFormat.TSV: partial(self._read_tsv, parse_time=gf.time_from_ssmmm),
            SyncMapFormat.TSVH: partial(self._read_tsv, parse_time=gf.time_from_hhmmssmmm),
            SyncMapFormat.TSVM: partial(self._read_tsv, parse_time=gf.time_from_ssmmm),
            SyncMapFormat.TTML: self._read_ttml,
            SyncMapFormat.TXT: partial(self._read_txt, parse_time=gf.time_from_ssmmm),
            SyncMapFormat.TXTH: partial(self._read_txt, parse_time=gf.time_from_hhmmssmmm),
            SyncMapFormat.TXTM: partial(self._read_txt, parse_time=gf.time_from_ssmmm),
            SyncMapFormat.VTT: self._read_vtt,
            SyncMapFormat.XML: self._read_xml,
            SyncMapFormat.XML_LEGACY: self._read_xml_legacy,
        }
        if sync_map_format is None:
            self.log_exc(u"Sync map format is None", None, True, ValueError)
        if sync_map_format not in map_read_function:
            self.log_exc(u"Sync map format '%s' is not allowed" % (sync_map_format), None, True, ValueError)
        if not gf.file_can_be_read(input_file_path):
            self.log_exc(u"Cannot read sync map file '%s'. Wrong permissions?" % (input_file_path), None, True, OSError)

        self.log([u"Input format:     '%s'", sync_map_format])
        self.log([u"Input path:       '%s'", input_file_path])
        self.log([u"Input parameters: '%s'", parameters])

        # open file for reading
        self.log(u"Opening input file")
        with io.open(input_file_path, "r", encoding="utf-8") as input_file:
            map_read_function[sync_map_format](input_file)

        # overwrite language if requested
        language = gf.safe_get(parameters, gc.PPN_SYNCMAP_LANGUAGE, None)
        if language is not None:
            self.log([u"Overwriting language to '%s'", language])
            for fragment in self.fragments:
                fragment.text_fragment.language = language

    def write(self, sync_map_format, output_file_path, parameters=None):
        """
        Write the current sync map to file in the requested format.

        Return ``True`` if the call succeeded,
        ``False`` if an error occurred.

        :param sync_map_format: the format of the sync map
        :type  sync_map_format: :class:`~aeneas.syncmap.SyncMapFormat`
        :param string output_file_path: the path to the output file to write
        :param dict parameters: additional parameters (e.g., for ``SMIL`` output)
        :raises: ValueError: if ``sync_map_format`` is ``None`` or it is not an allowed value
        :raises: TypeError: if a required parameter is missing
        :raises: OSError: if ``output_file_path`` cannot be written
        """
        map_write_function = {
            SyncMapFormat.AUD: partial(self._write_aud, format_time=gf.time_to_ssmmm),
            SyncMapFormat.AUDH: partial(self._write_aud, format_time=gf.time_to_hhmmssmmm),
            SyncMapFormat.AUDM: partial(self._write_aud, format_time=gf.time_to_ssmmm),
            SyncMapFormat.CSV: partial(self._write_csv, format_time=gf.time_to_ssmmm),
            SyncMapFormat.CSVH: partial(self._write_csv, format_time=gf.time_to_hhmmssmmm),
            SyncMapFormat.CSVM: partial(self._write_csv, format_time=gf.time_to_ssmmm),
            SyncMapFormat.DFXP: partial(self._write_ttml, parameters=parameters),
            SyncMapFormat.EAF: partial(self._write_eaf, parameters=parameters),
            SyncMapFormat.JSON: self._write_json,
            SyncMapFormat.RBSE: self._write_rbse,
            SyncMapFormat.SBV: partial(self._write_sub, use_newline=True),
            SyncMapFormat.SMIL: partial(self._write_smil, format_time=gf.time_to_hhmmssmmm, parameters=parameters),
            SyncMapFormat.SMILH: partial(self._write_smil, format_time=gf.time_to_hhmmssmmm, parameters=parameters),
            SyncMapFormat.SMILM: partial(self._write_smil, format_time=gf.time_to_ssmmm, parameters=parameters),
            SyncMapFormat.SRT: self._write_srt,
            SyncMapFormat.SSV: partial(self._write_ssv, format_time=gf.time_to_ssmmm),
            SyncMapFormat.SSVH: partial(self._write_ssv, format_time=gf.time_to_hhmmssmmm),
            SyncMapFormat.SSVM: partial(self._write_ssv, format_time=gf.time_to_ssmmm),
            SyncMapFormat.SUB: partial(self._write_sub, use_newline=False),
            SyncMapFormat.TAB: partial(self._write_tsv, format_time=gf.time_to_ssmmm),
            SyncMapFormat.TSV: partial(self._write_tsv, format_time=gf.time_to_ssmmm),
            SyncMapFormat.TSVH: partial(self._write_tsv, format_time=gf.time_to_hhmmssmmm),
            SyncMapFormat.TSVM: partial(self._write_tsv, format_time=gf.time_to_ssmmm),
            SyncMapFormat.TTML: partial(self._write_ttml, parameters=parameters),
            SyncMapFormat.TXT: partial(self._write_txt, format_time=gf.time_to_ssmmm),
            SyncMapFormat.TXTH: partial(self._write_txt, format_time=gf.time_to_hhmmssmmm),
            SyncMapFormat.TXTM: partial(self._write_txt, format_time=gf.time_to_ssmmm),
            SyncMapFormat.VTT: self._write_vtt,
            SyncMapFormat.XML: self._write_xml,
            SyncMapFormat.XML_LEGACY: self._write_xml_legacy,
        }
        if sync_map_format is None:
            self.log_exc(u"Sync map format is None", None, True, ValueError)
        if sync_map_format not in map_write_function:
            self.log_exc(u"Sync map format '%s' is not allowed" % (sync_map_format), None, True, ValueError)
        if not gf.file_can_be_written(output_file_path):
            self.log_exc(u"Cannot write sync map file '%s'. Wrong permissions?" % (output_file_path), None, True, OSError)

        self.log([u"Output format:     '%s'", sync_map_format])
        self.log([u"Output path:       '%s'", output_file_path])
        self.log([u"Output parameters: '%s'", parameters])

        # create dir hierarchy, if needed
        gf.ensure_parent_directory(output_file_path)

        # check required parameters
        if sync_map_format in [
                SyncMapFormat.SMIL,
                SyncMapFormat.SMILH,
                SyncMapFormat.SMILM
        ]:
            for key in [
                    gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF,
                    gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF
            ]:
                if gf.safe_get(parameters, key, None) is None:
                    self.log_exc(u"Parameter %s must be specified for format %s" % (key, sync_map_format), None, True, SyncMapMissingParameterError)

        # open file for writing
        self.log(u"Opening output file")
        with io.open(output_file_path, "w", encoding="utf-8") as output_file:
            map_write_function[sync_map_format](output_file)

    def _read_aud(self, input_file, parse_time):
        """ Read from AUD file """
        identifier_index = 1
        for line in input_file.readlines():
            split = line.strip().split("\t")
            self.add_fragment(
                SyncMapFragment(
                    text_fragment=TextFragment(
                        identifier = u"f" + str(identifier_index).zfill(6),
                        lines=[split[2]]
                    ),
                    begin=parse_time(split[0]),
                    end=parse_time(split[1])
                )
            )
            identifier_index += 1

    def _write_aud(self, output_file, format_time):
        """ Write to AUD file """
        msg = []
        for fragment in self.fragments:
            msg.append(
                u"%s\t%s\t%s" % (
                    format_time(fragment.begin),
                    format_time(fragment.end),
                    u" ".join(fragment.text_fragment.lines)
                )
            )
        output_file.write(u"\n".join(msg))

    def _read_csv(self, input_file, parse_time):
        """ Read from CSV file """
        for line in input_file.readlines():
            split = line.strip().split(u",")
            self.add_fragment(
                SyncMapFragment(
                    text_fragment=TextFragment(
                        identifier=split[0],
                        lines=[(u",".join(split[3:]))[1:-1]]
                    ),
                    begin=parse_time(split[1]),
                    end=parse_time(split[2])
                )
            )

    def _write_csv(self, output_file, format_time):
        """ Write to CSV file """
        msg = []
        for fragment in self.fragments:
            msg.append(
                u"%s,%s,%s,\"%s\"" % (
                    fragment.text_fragment.identifier,
                    format_time(fragment.begin),
                    format_time(fragment.end),
                    fragment.text_fragment.text
                )
            )
        output_file.write(u"\n".join(msg))

    def _read_eaf(self, input_file):
        """ Read from EAF file """
        # namespaces
        xsi = "http://www.w3.org/2001/XMLSchema-instance"
        ns_map = {"xsi" : xsi}
        # get root
        root = etree.fromstring(gf.safe_bytes(input_file.read()))
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
            self.add_fragment(
                SyncMapFragment(
                    text_fragment=TextFragment(
                        identifier=identifier,
                        lines=lines
                    ),
                    begin=begin,
                    end=end
                )
            )

    def _write_eaf(self, output_file, parameters=None):
        """ Write to EAF file """
        # namespaces
        xsi = "http://www.w3.org/2001/XMLSchema-instance"
        ns_map = {"xsi" : xsi}
        # build doc
        doc = etree.Element("ANNOTATION_DOCUMENT", nsmap=ns_map)
        doc.attrib["{%s}noNamespaceSchemaLocation" % xsi] = "http://www.mpi.nl/tools/elan/EAFv2.8.xsd"
        doc.attrib["AUTHOR"] = "aeneas"
        doc.attrib["DATE"] = "2016-01-01T00:00:00+00:00"
        doc.attrib["FORMAT"] = "2.8"
        doc.attrib["VERSION"] = "2.8"
        # header
        header = etree.SubElement(doc, "HEADER")
        header.attrib["MEDIA_FILE"] = ""
        header.attrib["TIME_UNITS"] = "milliseconds"
        if (not parameters is None) and ("audio_file_path_absolute" in parameters):
            media = etree.SubElement(header, "MEDIA_DESCRIPTOR")
            media.attrib["MEDIA_URL"] = "file://%s" % parameters["audio_file_path_absolute"]
            media.attrib["MIME_TYPE"] = gf.mimetype_from_path(parameters["audio_file_path_absolute"])
        # time order
        time_order = etree.SubElement(doc, "TIME_ORDER")
        # tier
        tier = etree.SubElement(doc, "TIER")
        tier.attrib["LINGUISTIC_TYPE_REF"] = "utterance"
        tier.attrib["TIER_ID"] = "tier1"
        i = 1
        for fragment in self.fragments:
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
            i += 1
        # linguistic type
        ling = etree.SubElement(doc, "LINGUISTIC_TYPE")
        ling.attrib["LINGUISTIC_TYPE_ID"] = "utterance"
        ling.attrib["TIME_ALIGNABLE"] = "true"
        # write tree
        self._write_tree_to_file(doc, output_file, xml_declaration=True)

    def _read_json(self, input_file):
        """ Read from JSON file """
        contents_dict = json.loads(input_file.read())
        for fragment in contents_dict["fragments"]:
            self.add_fragment(
                SyncMapFragment(
                    text_fragment=TextFragment(
                        identifier=fragment["id"],
                        language=fragment["language"],
                        lines=fragment["lines"]
                    ),
                    begin=gf.time_from_ssmmm(fragment["begin"]),
                    end=gf.time_from_ssmmm(fragment["end"])
                )
            )

    def _write_json(self, output_file):
        """ Write to JSON file """
        output_file.write(self.json_string)

    def _read_rbse(self, input_file):
        """ Read from RBSE file """
        contents_dict = json.loads(input_file.read())
        for fragment in contents_dict["smil_data"]:
            self.add_fragment(
                SyncMapFragment(
                    text_fragment=TextFragment(
                        identifier=fragment["id"],
                        lines=[u""] # TODO read text from additional text_file?
                    ),
                    begin=gf.time_from_ssmmm(fragment["begin"]),
                    end=gf.time_from_ssmmm(fragment["end"])
                )
            )

    def _write_rbse(self, output_file):
        """ Write to RBSE file """
        smil_data = []
        smil_ids = []
        for fragment in self.fragments:
            text = fragment.text_fragment
            smil_data.append({
                "id" : text.identifier,
                "begin" : gf.time_to_ssmmm(fragment.begin),
                "end" : gf.time_to_ssmmm(fragment.end)
            })
            smil_ids.append(text.identifier)
        output_file.write(
            gf.safe_unicode(
                json.dumps(
                    obj={
                        "smil_ids": smil_ids,
                        "smil_data": smil_data
                    },
                    indent=1,
                    sort_keys=True
                )
            )
        )

    def _read_smil(self, input_file):
        """
        Read from SMIL file.

        Limitations:
        1. parses only <par> elements, in order
        2. timings must have hh:mm:ss.mmm or ss.mmm format (autodetected)
        3. both clipBegin and clipEnd attributes of <audio> must be populated
        """
        smil_ns = "{http://www.w3.org/ns/SMIL}"
        root = etree.fromstring(gf.safe_bytes(input_file.read()))
        for par in root.iter(smil_ns + "par"):
            for child in par:
                if child.tag == (smil_ns + "text"):
                    identifier = gf.safe_unicode(gf.split_url(child.get("src"))[1])
                elif child.tag == (smil_ns + "audio"):
                    begin = gf.time_from_hhmmssmmm(child.get("clipBegin"))
                    if begin is None:
                        begin = gf.time_from_ssmmm(child.get("clipBegin"))
                    end = gf.time_from_hhmmssmmm(child.get("clipEnd"))
                    if end is None:
                        end = gf.time_from_ssmmm(child.get("clipEnd"))
            self.add_fragment(
                SyncMapFragment(
                    text_fragment=TextFragment(
                        identifier=identifier,
                        lines=[u""] # TODO read text from additional text_file?
                    ),
                    begin=begin,
                    end=end
                )
            )

    def _write_smil(self, output_file, format_time, parameters):
        """ Write to SMIL file """
        # we are sure we have them
        text_ref = parameters[gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF]
        audio_ref = parameters[gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF]

        # namespaces
        smil_ns = "http://www.w3.org/ns/SMIL"
        epub_ns = "http://www.idpf.org/2007/ops"
        ns_map = {None : smil_ns, "epub" : epub_ns}

        # build tree
        smil_elem = etree.Element("{%s}smil" % smil_ns, nsmap=ns_map)
        smil_elem.attrib["version"] = "3.0"
        body_elem = etree.SubElement(smil_elem, "{%s}body" % smil_ns)
        seq_elem = etree.SubElement(body_elem, "{%s}seq" % smil_ns)
        seq_elem.attrib["id"] = "seq" + str(1).zfill(6)
        seq_elem.attrib["{%s}textref" % epub_ns] = text_ref

        if self.is_single_level:
            # single level
            i = 1
            for fragment in self.fragments:
                text = fragment.text_fragment
                par_elem = etree.SubElement(seq_elem, "{%s}par" % smil_ns)
                par_elem.attrib["id"] = "par" + str(i).zfill(6)
                text_elem = etree.SubElement(par_elem, "{%s}text" % smil_ns)
                text_elem.attrib["src"] = "%s#%s" % (text_ref, text.identifier)
                audio_elem = etree.SubElement(par_elem, "{%s}audio" % smil_ns)
                audio_elem.attrib["src"] = audio_ref
                audio_elem.attrib["clipBegin"] = format_time(fragment.begin)
                audio_elem.attrib["clipEnd"] = format_time(fragment.end)
                i += 1
        else:
            # TODO support generic multiple levels
            # multiple levels
            par_index = 1
            for par_child in self.fragments_tree.children_not_empty:
                par_seq_elem = etree.SubElement(seq_elem, "{%s}seq" % smil_ns)
                #par_seq_elem.attrib["id"] = "p" + str(par_index).zfill(6)
                par_seq_elem.attrib["{%s}type" % epub_ns] = "paragraph"
                par_seq_elem.attrib["{%s}textref" % epub_ns] = text_ref + "#" + par_child.value.text_fragment.identifier
                sen_index = 1
                for sen_child in par_child.children_not_empty:
                    sen_seq_elem = etree.SubElement(par_seq_elem, "{%s}seq" % smil_ns)
                    #sen_seq_elem.attrib["id"] = par_seq_elem.attrib["id"] + "s" + str(sen_index).zfill(6)
                    sen_seq_elem.attrib["{%s}type" % epub_ns] = "sentence"
                    sen_seq_elem.attrib["{%s}textref" % epub_ns] = text_ref + "#" + sen_child.value.text_fragment.identifier
                    wor_index = 1
                    for wor_child in sen_child.children_not_empty:
                        fragment = wor_child.value
                        text = fragment.text_fragment
                        wor_seq_elem = etree.SubElement(sen_seq_elem, "{%s}seq" % smil_ns)
                        #wor_seq_elem.attrib["id"] = sen_seq_elem.attrib["id"] + "s" + str(wor_index).zfill(6)
                        wor_seq_elem.attrib["{%s}type" % epub_ns] = "word"
                        wor_seq_elem.attrib["{%s}textref" % epub_ns] = text_ref + "#" + text.identifier
                        wor_par_elem = etree.SubElement(wor_seq_elem, "{%s}par" % smil_ns)
                        text_elem = etree.SubElement(wor_par_elem, "{%s}text" % smil_ns)
                        text_elem.attrib["src"] = "%s#%s" % (text_ref, text.identifier)
                        audio_elem = etree.SubElement(wor_par_elem, "{%s}audio" % smil_ns)
                        audio_elem.attrib["src"] = audio_ref
                        audio_elem.attrib["clipBegin"] = format_time(fragment.begin)
                        audio_elem.attrib["clipEnd"] = format_time(fragment.end)
                        wor_index +=1
                    sen_index +=1
                par_index +=1
        # write tree
        self._write_tree_to_file(smil_elem, output_file, xml_declaration=False)

    def _read_srt(self, input_file):
        """ Read from SRT file """
        lines = input_file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if len(line) > 0:
                identifier_index = int(line)
                identifier = u"f" + str(identifier_index).zfill(6)
                i += 1
                if i < len(lines):
                    line = lines[i].strip()
                    timings = line.split(" --> ")
                    if len(timings) == 2:
                        begin = gf.time_from_hhmmssmmm(timings[0], decimal_separator=",")
                        end = gf.time_from_hhmmssmmm(timings[1], decimal_separator=",")
                        fragment_lines = []
                        while (i + 1 < len(lines)) and (len(line) > 0):
                            i += 1
                            line = lines[i].strip()
                            if len(line) > 0:
                                fragment_lines.append(line)
                        # should never happen, but just in case...
                        if len(fragment_lines) == 0:
                            fragment_lines = [u""]
                        self.add_fragment(
                            SyncMapFragment(
                                text_fragment=TextFragment(
                                    identifier=identifier,
                                    lines=fragment_lines
                                ),
                                begin=begin,
                                end=end
                            )
                        )
            i += 1

    def _write_srt(self, output_file):
        """ Write to SRT file """
        msg = []
        i = 1
        for fragment in self.fragments:
            text = fragment.text_fragment
            msg.append(u"%d" % i)
            msg.append(u"%s --> %s" % (
                gf.time_to_srt(fragment.begin),
                gf.time_to_srt(fragment.end)
            ))
            msg.extend(text.lines)
            msg.append(u"")
            i += 1
        # add an extra \n at the end
        msg.append(u"")
        output_file.write(u"\n".join(msg))

    def _read_sub(self, input_file, use_newline=False):
        """ Read from SUB file """
        lines = input_file.readlines()
        i = 0
        identifier_index = 1
        in_subtitle = False
        # TODO read [INFORMATION] header?
        while i < len(lines):
            line = lines[i].strip()
            if len(line) > 0:
                if (not in_subtitle) and (line == u"[SUBTITLE]"):
                    in_subtitle = True
                if in_subtitle:
                    timings = line.split(",")
                    if len(timings) == 2:
                        identifier = u"f" + str(identifier_index).zfill(6)
                        identifier_index += 1
                        begin = gf.time_from_hhmmssmmm(timings[0])
                        end = gf.time_from_hhmmssmmm(timings[1])
                        fragment_lines = []
                        while (i + 1 < len(lines)) and (len(line) > 0):
                            i += 1
                            line = lines[i].strip()
                            if use_newline:
                                line_split = [l for l in [line] if len(l) > 0]
                            else:
                                line_split = [l for l in line.split(u"[br]") if len(l) > 0]
                            if len(line_split) > 0:
                                fragment_lines.extend(line_split)
                        # should never happen, but just in case...
                        if len(fragment_lines) == 0:
                            fragment_lines = [u""]
                        self.add_fragment(
                            SyncMapFragment(
                                text_fragment=TextFragment(
                                    identifier=identifier,
                                    lines=fragment_lines
                                ),
                                begin=begin,
                                end=end
                            )
                        )
            i += 1

    def _write_sub(self, output_file, use_newline=False):
        """ Write to SUB file """
        msg = []
        # TODO write [INFORMATION] header?
        msg.append(u"[SUBTITLE]")
        for fragment in self.fragments:
            text = fragment.text_fragment
            msg.append(u"%s,%s" % (
                gf.time_to_hhmmssmmm(fragment.begin),
                gf.time_to_hhmmssmmm(fragment.end)
            ))
            if use_newline:
                msg.extend(text.lines)
            else:
                msg.append(u"[br]".join(text.lines))
            msg.append(u"")
        msg.append(u"[END SUBTITLE]")
        # add an extra \n at the end
        msg.append(u"")
        output_file.write(u"\n".join(msg))

    def _read_ssv(self, input_file, parse_time):
        """ Read from SSV file """
        for line in input_file.readlines():
            split = line.strip().split(" ")
            self.add_fragment(
                SyncMapFragment(
                    text_fragment=TextFragment(
                        identifier=split[2],
                        lines=[(u" ".join(split[3:]))[1:-1]]
                    ),
                    begin=parse_time(split[0]),
                    end=parse_time(split[1])
                )
            )

    def _write_ssv(self, output_file, format_time):
        """ Write to SSV file """
        msg = []
        for fragment in self.fragments:
            msg.append(
                u"%s %s %s \"%s\"" % (
                    format_time(fragment.begin),
                    format_time(fragment.end),
                    fragment.text_fragment.identifier,
                    fragment.text_fragment.text
                )
            )
        output_file.write(u"\n".join(msg))

    def _read_tsv(self, input_file, parse_time):
        """ Read from TSV file """
        for line in input_file.readlines():
            split = line.strip().split("\t")
            self.add_fragment(
                SyncMapFragment(
                    text_fragment=TextFragment(
                        identifier=split[2],
                        lines=[u""] # TODO read text from additional text_file?
                    ),
                    begin=parse_time(split[0]),
                    end=parse_time(split[1])
                )
            )

    def _write_tsv(self, output_file, format_time):
        """ Write to TSV file """
        msg = []
        for fragment in self.fragments:
            msg.append(
                u"%s\t%s\t%s" % (
                    format_time(fragment.begin),
                    format_time(fragment.end),
                    fragment.text_fragment.identifier
                )
            )
        output_file.write(u"\n".join(msg))

    def _read_ttml(self, input_file):
        """ Read from TTML file """
        ttml_ns = "{http://www.w3.org/ns/ttml}"
        xml_ns = "{http://www.w3.org/XML/1998/namespace}"
        root = etree.fromstring(gf.safe_bytes(input_file.read()))
        language = root.get(xml_ns + "lang")
        for elem in root.iter(ttml_ns + "p"):
            identifier = gf.safe_unicode(elem.get(xml_ns + "id"))
            begin = gf.time_from_ttml(elem.get("begin"))
            end = gf.time_from_ttml(elem.get("end"))
            fragment_lines = self._get_lines_from_node_text(elem)
            self.add_fragment(
                SyncMapFragment(
                    text_fragment=TextFragment(
                        identifier=identifier,
                        language=language,
                        lines=fragment_lines
                    ),
                    begin=begin,
                    end=end
                )
            )

    def _write_ttml(self, output_file, parameters):
        """ Write to TTML file """
        # get language
        language = None
        if (parameters is not None) and ("language" in parameters):
            language = parameters["language"]
        elif len(self.fragments) > 0:
            language = self.fragments[0].text_fragment.language
        if language is None:
            language = ""

        # namespaces
        ttml_ns = "http://www.w3.org/ns/ttml"
        xml_ns = "http://www.w3.org/XML/1998/namespace"
        ns_map = {None : ttml_ns}

        # build tree
        tt_elem = etree.Element("{%s}tt" % ttml_ns, nsmap=ns_map)
        tt_elem.attrib["{%s}lang" % xml_ns] = language
        # TODO add metadata from parameters here?
        #head_elem = etree.SubElement(tt_elem, "{%s}head" % ttml_ns)
        body_elem = etree.SubElement(tt_elem, "{%s}body" % ttml_ns)
        div_elem = etree.SubElement(body_elem, "{%s}div" % ttml_ns)

        if self.is_single_level:
            # single level
            for fragment in self.fragments:
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
            for par_child in self.fragments_tree.children_not_empty:
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
        self._write_tree_to_file(tt_elem, output_file)

    def _read_txt(self, input_file, parse_time):
        """ Read from TXT file """
        for line in input_file.readlines():
            split = line.strip().split(" ")
            self.add_fragment(
                SyncMapFragment(
                    text_fragment=TextFragment(
                        identifier=split[0],
                        lines=[(u" ".join(split[3:]))[1:-1]]
                    ),
                    begin=parse_time(split[1]),
                    end=parse_time(split[2])
                )
            )

    def _write_txt(self, output_file, format_time):
        """ Write to TXT file """
        msg = []
        for fragment in self.fragments:
            msg.append(
                u"%s %s %s \"%s\"" % (
                    fragment.text_fragment.identifier,
                    format_time(fragment.begin),
                    format_time(fragment.end),
                    fragment.text_fragment.text
                )
            )
        output_file.write(u"\n".join(msg))

    def _read_vtt(self, input_file):
        """ Read from WebVTT file """
        lines = input_file.readlines()
        # ignore the first line containing "WEBVTT" and the following blank line
        i = 2
        while i < len(lines):
            line = lines[i].strip()
            if len(line) > 0:
                identifier_index = int(line)
                identifier = u"f" + str(identifier_index).zfill(6)
                i += 1
                if i < len(lines):
                    line = lines[i].strip()
                    timings = line.split(" --> ")
                    if len(timings) == 2:
                        begin = gf.time_from_hhmmssmmm(timings[0])
                        end = gf.time_from_hhmmssmmm(timings[1])
                        fragment_lines = []
                        while (i + 1 < len(lines)) and (len(line) > 0):
                            i += 1
                            line = lines[i].strip()
                            if len(line) > 0:
                                fragment_lines.append(line)
                        # should never happen, but just in case...
                        if len(fragment_lines) == 0:
                            fragment_lines = [u""]
                        self.add_fragment(
                            SyncMapFragment(
                                text_fragment=TextFragment(
                                    identifier=identifier,
                                    lines=fragment_lines
                                ),
                                begin=begin,
                                end=end
                            )
                        )
            i += 1

    def _write_vtt(self, output_file):
        """ Write to WebVTT file """
        msg = []
        i = 1
        msg.append(u"WEBVTT")
        msg.append(u"")
        for fragment in self.fragments:
            text = fragment.text_fragment
            msg.append(u"%d" % i)
            msg.append(u"%s --> %s" % (
                gf.time_to_hhmmssmmm(fragment.begin),
                gf.time_to_hhmmssmmm(fragment.end)
            ))
            msg.extend(text.lines)
            msg.append(u"")
            i += 1
        # add an extra \n at the end
        msg.append(u"")
        output_file.write(u"\n".join(msg))

    def _read_xml(self, input_file):
        """ Read from XML file """
        root = etree.fromstring(gf.safe_bytes(input_file.read()))
        for frag in root:
            identifier = gf.safe_unicode(frag.get("id"))
            begin = gf.time_from_ssmmm(frag.get("begin"))
            end = gf.time_from_ssmmm(frag.get("end"))
            lines = []
            for child in frag:
                if child.tag == "line":
                    lines.append(gf.safe_unicode(child.text))
            self.add_fragment(
                SyncMapFragment(
                    text_fragment=TextFragment(
                        identifier=identifier,
                        lines=lines
                    ),
                    begin=begin,
                    end=end
                )
            )

    def _write_xml(self, output_file):
        """ Write to XML file """
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
        visit_children(self.fragments_tree, map_elem)
        self._write_tree_to_file(map_elem, output_file)

    def _read_xml_legacy(self, input_file):
        """ Read from XML file (legacy format) """
        root = etree.fromstring(gf.safe_bytes(input_file.read()))
        for frag in root:
            for child in frag:
                if child.tag == "identifier":
                    identifier = gf.safe_unicode(child.text)
                elif child.tag == "start":
                    begin = gf.time_from_ssmmm(child.text)
                elif child.tag == "end":
                    end = gf.time_from_ssmmm(child.text)
            self.add_fragment(
                SyncMapFragment(
                    text_fragment=TextFragment(
                        identifier=identifier,
                        lines=[u""] # TODO read text from additional text_file?
                    ),
                    begin=begin,
                    end=end
                )
            )

    def _write_xml_legacy(self, output_file):
        """ Write to XML file (legacy format) """
        msg = []
        msg.append(u"<?xml version=\"1.0\" encoding=\"UTF-8\" ?>")
        msg.append(u"<map>")
        for fragment in self.fragments:
            msg.append(u" <fragment>")
            msg.append(u"  <identifier>%s</identifier>" % fragment.text_fragment.identifier)
            msg.append(u"  <start>%s</start>" % gf.time_to_ssmmm(fragment.begin))
            msg.append(u"  <end>%s</end>" % gf.time_to_ssmmm(fragment.end))
            msg.append(u" </fragment>")
        msg.append(u"</map>")
        output_file.write(u"\n".join(msg))

    @classmethod
    def _write_tree_to_file(
            cls,
            root_element,
            output_file,
            pretty_print=True,
            xml_declaration=True
        ):
        """
        Write an ``lxml`` tree to the given output file.
        """
        string = etree.tostring(
            root_element,
            encoding="UTF-8",
            method="xml",
            xml_declaration=xml_declaration,
            pretty_print=pretty_print
        )
        output_file.write(gf.safe_unicode(string))

    @classmethod
    def _get_lines_from_node_text(cls, node):
        """
        Given an ``lxml`` node, get lines from ``node.text``,
        where the line separator is ``<br xmlns=... />``.
        """
        # TODO more robust parsing
        parts = ([node.text] + list(chain(*([etree.tostring(c, with_tail=False), c.tail] for c in node.getchildren()))) + [node.tail])
        parts = [gf.safe_unicode(p) for p in parts]
        parts = [p.strip() for p in parts if not p.startswith(u"<br ")]
        parts = [p for p in parts if len(p) > 0]
        uparts = []
        for part in parts:
            uparts.append(gf.safe_unicode(part))
        return uparts



class SyncMapFragment(object):
    """
    A sync map fragment, that is,
    a text fragment and an associated time interval ``[begin, end]``.

    :param text_fragment: the text fragment
    :type  text_fragment: :class:`~aeneas.textfile.TextFragment`
    :param begin: the begin time of the audio interval
    :type  begin: :class:`~aeneas.timevalue.TimeValue`
    :param end: the end time of the audio interval
    :type  end: :class:`~aeneas.timevalue.TimeValue`
    :param float confidence: the confidence of the audio timing
    """

    TAG = u"SyncMapFragment"

    def __init__(
            self,
            text_fragment=None,
            begin=None,
            end=None,
            confidence=1.0
        ):
        self.text_fragment = text_fragment
        self.begin = begin
        self.end = end
        self.confidence = confidence

    def __unicode__(self):
        return u"%s %.3f %.3f" % (
            self.text_fragment.identifier,
            self.begin,
            self.end
        )

    def __str__(self):
        return gf.safe_str(self.__unicode__())

    @property
    def text_fragment(self):
        """
        The text fragment associated with this sync map fragment.

        :rtype: :class:`~aeneas.textfile.TextFragment`
        """
        return self.__text_fragment
    @text_fragment.setter
    def text_fragment(self, text_fragment):
        self.__text_fragment = text_fragment

    @property
    def begin(self):
        """
        The begin time of this sync map fragment.

        :rtype: :class:`~aeneas.timevalue.TimeValue`
        """
        return self.__begin
    @begin.setter
    def begin(self, begin):
        self.__begin = begin

    @property
    def end(self):
        """
        The end time of this sync map fragment.

        :rtype: :class:`~aeneas.timevalue.TimeValue`
        """
        return self.__end
    @end.setter
    def end(self, end):
        self.__end = end

    @property
    def confidence(self):
        """
        The confidence of the audio timing, from ``0.0`` to ``1.0``.

        Currently this value is not used, and it is always ``1.0``.

        :rtype: float
        """
        return self.__confidence
    @confidence.setter
    def confidence(self, confidence):
        self.__confidence = confidence

    @property
    def audio_duration(self):
        """
        The audio duration of this sync map fragment,
        as end time minus begin time.

        :rtype: :class:`~aeneas.timevalue.TimeValue`
        """
        if (self.begin is None) or (self.end is None):
            return TimeValue("0.000")
        return self.end - self.begin

    @property
    def chars(self):
        """
        Return the number of characters of the text fragment,
        not including the line separators.

        :rtype: int

        .. versionadded:: 1.2.0
        """
        if self.text_fragment is None:
            return 0
        return self.text_fragment.chars

    @property
    def rate(self):
        """
        The rate, in characters/second, of this fragment.

        :rtype: None or Decimal

        .. versionadded:: 1.2.0
        """
        if self.audio_duration == TimeValue("0.000"):
            return None
        return Decimal(self.chars / self.audio_duration)



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



