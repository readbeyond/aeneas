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

from aeneas.syncmap.smfaudacity import SyncMapFormatAudacity
from aeneas.syncmap.smfcsv import SyncMapFormatCSV
from aeneas.syncmap.smfeaf import SyncMapFormatEAF
from aeneas.syncmap.smfjson import SyncMapFormatJSON
from aeneas.syncmap.smfrbse import SyncMapFormatRBSE
from aeneas.syncmap.smfsmil import SyncMapFormatSMIL
from aeneas.syncmap.smfsrt import SyncMapFormatSRT
from aeneas.syncmap.smfssv import SyncMapFormatSSV
from aeneas.syncmap.smfsub import SyncMapFormatSUB
from aeneas.syncmap.smftextgrid import SyncMapFormatTextGrid
from aeneas.syncmap.smftsv import SyncMapFormatTSV
from aeneas.syncmap.smfttml import SyncMapFormatTTML
from aeneas.syncmap.smftxt import SyncMapFormatTXT
from aeneas.syncmap.smfvtt import SyncMapFormatVTT
from aeneas.syncmap.smfxml import SyncMapFormatXML
from aeneas.syncmap.smfxmllegacy import SyncMapFormatXMLLegacy


class SyncMapFormat(object):
    """
    Enumeration of the supported output formats
    for a synchronization map.
    """

    AUD = SyncMapFormatAudacity.DEFAULT
    """
    Alias for ``AUDM``.
    """

    AUDH = SyncMapFormatAudacity.HUMAN
    """
    Tab-separated plain text,
    with human-readable time values
    and fragment text::

        00:00:00.000   00:00:01.234   Text of fragment 1
        00:00:01.234   00:00:05.678   Text of fragment 2
        00:00:05.678   00:00:07.890   Text of fragment 3

    * Multiple levels: no
    * Multiple lines: no

    See also http://manual.audacityteam.org/man/label_tracks.html#export

    .. versionadded:: 1.5.0
    """

    AUDM = SyncMapFormatAudacity.MACHINE
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

    See also http://manual.audacityteam.org/man/label_tracks.html#export

    .. versionadded:: 1.5.0
    """

    CSV = SyncMapFormatCSV.DEFAULT
    """
    Alias for ``CSVM``.
    """

    CSVH = SyncMapFormatCSV.HUMAN
    """
    Comma-separated values (CSV),
    with human-readable time values::

        f001,00:00:00.000,00:00:01.234,"First fragment text"
        f002,00:00:01.234,00:00:05.678,"Second fragment text"
        f003,00:00:05.678,00:00:07.890,"Third fragment text"

    * Multiple levels: no
    * Multiple lines: no

    Please note that the text is assumed to be contained
    in double quotes ("..."),
    which are stripped when reading from file,
    and added back when writing to file.

    .. versionadded:: 1.0.4
    """

    CSVM = SyncMapFormatCSV.MACHINE
    """
    Comma-separated values (CSV),
    with machine-readable time values::

        f001,0.000,1.234,"First fragment text"
        f002,1.234,5.678,"Second fragment text"
        f003,5.678,7.890,"Third fragment text"

    * Multiple levels: no
    * Multiple lines: no

    Please note that the text is assumed to be contained
    in double quotes ("..."),
    which are stripped when reading from file,
    and added back when writing to file.

    .. versionadded:: 1.2.0
    """

    DFXP = SyncMapFormatTTML.DFXP
    """
    Alias for ``TTML``.

    .. versionadded:: 1.4.1
    """

    EAF = SyncMapFormatEAF.DEFAULT
    """
    ELAN EAF::

        <?xml version="1.0" encoding="UTF-8"?>
        <ANNOTATION_DOCUMENT AUTHOR="aeneas" DATE="2016-01-01T00:00:00+00:00" FORMAT="2.8" VERSION="2.8" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv2.8.xsd">
            <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds" />
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

    See also https://tla.mpi.nl/tla-news/documentation-of-eaf-elan-annotation-format/

    .. versionadded:: 1.5.0
    """

    JSON = SyncMapFormatJSON.DEFAULT
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

    RBSE = SyncMapFormatRBSE.DEFAULT
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

    See also https://github.com/pettarin/rb_smil_emulator

    Deprecated, it will be removed in v2.0.0.

    .. deprecated:: 1.5.0

    .. versionadded:: 1.2.0
    """

    SBV = SyncMapFormatSUB.SBV
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

    See also https://wiki.videolan.org/SubViewer/

    Note that the ``[INFORMATION]`` header is ignored when reading,
    and it is not produced when writing.
    Moreover, extensions
    (i.e., ``[COLF]``, ``[SIZE]``, ``[FONT]``)
    are not supported.
    """

    SMIL = SyncMapFormatSMIL.DEFAULT
    """
    Alias for ``SMILH``.
    """

    SMILH = SyncMapFormatSMIL.HUMAN
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

    See also http://www.idpf.org/epub3/latest/mediaoverlays

    .. versionadded:: 1.2.0
    """

    SMILM = SyncMapFormatSMIL.MACHINE
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

    See also http://www.idpf.org/epub3/latest/mediaoverlays

    .. versionadded:: 1.2.0
    """

    SRT = SyncMapFormatSRT.DEFAULT
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

    See also https://wiki.videolan.org/SubRip/

    Note that extensions
    (i.e., ``<b>``, ``<s>``, ``<u>``, ``<i>``, ``<font>``)
    are not supported.
    """

    SSV = SyncMapFormatSSV.DEFAULT
    """
    Alias for ``SSVM``.

    .. versionadded:: 1.0.4
    """

    SSVH = SyncMapFormatSSV.HUMAN
    """
    Space-separated plain text,
    with human-readable time values::

        00:00:00.000 00:00:01.234 f001 "First fragment text"
        00:00:01.234 00:00:05.678 f002 "Second fragment text"
        00:00:05.678 00:00:07.890 f003 "Third fragment text"

    * Multiple levels: no
    * Multiple lines: no

    Please note that the text is assumed to be contained
    in double quotes ("..."),
    which are stripped when reading from file,
    and added back when writing to file.

    .. versionadded:: 1.0.4
    """

    SSVM = SyncMapFormatSSV.MACHINE
    """
    Space-separated plain text,
    with machine-readable time values::

        0.000 1.234 f001 "First fragment text"
        1.234 5.678 f002 "Second fragment text"
        5.678 7.890 f003 "Third fragment text"

    * Multiple levels: no
    * Multiple lines: no

    Please note that the text is assumed to be contained
    in double quotes ("..."),
    which are stripped when reading from file,
    and added back when writing to file.

    .. versionadded:: 1.2.0
    """

    SUB = SyncMapFormatSUB.SUB
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

    See also https://wiki.videolan.org/SubViewer/

    Note that the ``[INFORMATION]`` header is ignored when reading,
    and it is not produced when writing.
    Moreover, extensions
    (i.e., ``[COLF]``, ``[SIZE]``, ``[FONT]``)
    are not supported.

    .. versionadded:: 1.4.1
    """

    TAB = SyncMapFormatTSV.TAB
    """
    Deprecated, it will be removed in v2.0.0.
    Use ``TSV`` instead.

    .. deprecated:: 1.0.3
    """

    TEXTGRID = SyncMapFormatTextGrid.DEFAULT
    """
    Alias for ``TEXTGRID_LONG``.
    """

    TEXTGRID_LONG = SyncMapFormatTextGrid.LONG
    """
    Praat full TextGrid format::

        File type = "ooTextFile"
        Object class = "TextGrid"

        xmin = 0.0
        xmax = 7.89
        tiers? <exists>
        size = 1
        item []:
            item [1]:
                class = "IntervalTier"
                name = "Token"
                xmin = 0.0
                xmax = 7.89
                intervals: size = 3
                intervals [1]:
                    xmin = 0.0
                    xmax = 1.234
                    text = "First fragment text"
                intervals [2]:
                    xmin = 1.234
                    xmax = 5.678
                    text = "Second fragment text"
                intervals [3]:
                    xmin = 5.678
                    xmax = 7.89
                    text = "Third fragment text"

    * Multiple levels: no (not yet)
    * Multiple lines: no

    See also http://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html

    Note that at the moment reading support is limited
    to the first tier in the TextGrid file.

    .. versionadded:: 1.7.0
    """

    TEXTGRID_SHORT = SyncMapFormatTextGrid.SHORT
    """
    Praat short TextGrid format::

        File type = "ooTextFile"
        Object class = "TextGrid"

        0.0
        7.89
        <exists>
        1
        "IntervalTier"
        "Token"
        0.0
        7.89
        3
        0.0
        1.234
        "First fragment text"
        1.234
        5.678
        "Second fragment text"
        5.678
        7.89
        "Third fragment text"

    * Multiple levels: no (not yet)
    * Multiple lines: no

    See also http://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html

    Note that at the moment reading support is limited
    to the first tier in the TextGrid file.

    .. versionadded:: 1.7.0
    """

    TSV = SyncMapFormatTSV.DEFAULT
    """
    Alias for ``TSVM``.
    """

    TSVH = SyncMapFormatTSV.HUMAN
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

    TSVM = SyncMapFormatTSV.MACHINE
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

    TTML = SyncMapFormatTTML.DEFAULT
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

    See also https://www.w3.org/TR/ttml1/

    * Multiple levels: yes (output only)
    * Multiple lines: yes
    """

    TXT = SyncMapFormatTXT.DEFAULT
    """
    Alias for ``TXTM``.
    """

    TXTH = SyncMapFormatTXT.HUMAN
    """
    Space-separated plain text
    with human-readable time values::

        f001 00:00:00.000 00:00:01.234 "First fragment text"
        f002 00:00:01.234 00:00:05.678 "Second fragment text"
        f003 00:00:05.678 00:00:07.890 "Third fragment text"

    * Multiple levels: no
    * Multiple lines: no

    Please note that the text is assumed to be contained
    in double quotes ("..."),
    which are stripped when reading from file,
    and added back when writing to file.

    .. versionadded:: 1.0.4
    """

    TXTM = SyncMapFormatTXT.MACHINE
    """
    Space-separated plain text,
    with machine-readable time values,
    compatible with ``SonicVisualizer``::

        f001 0.000 1.234 "First fragment text"
        f002 1.234 5.678 "Second fragment text"
        f003 5.678 7.890 "Third fragment text"

    * Multiple levels: no
    * Multiple lines: no

    Please note that the text is assumed to be contained
    in double quotes ("..."),
    which are stripped when reading from file,
    and added back when writing to file.

    .. versionadded:: 1.2.0
    """

    VTT = SyncMapFormatVTT.DEFAULT
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

    See also https://w3c.github.io/webvtt/

    Note that WebVTT files using tabs as separators
    cannot be read at the moment.
    Use spaces instead or pre-process your files,
    replacing tabs with spaces.
    """

    XML = SyncMapFormatXML.DEFAULT
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

    XML_LEGACY = SyncMapFormatXMLLegacy.DEFAULT
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

    Deprecated, it will be removed in v2.0.0.
    Use ``XML`` instead.

    .. deprecated:: 1.2.0
    """

    CODE_TO_CLASS = {
        AUD: SyncMapFormatAudacity,
        AUDH: SyncMapFormatAudacity,
        AUDM: SyncMapFormatAudacity,
        CSV: SyncMapFormatCSV,
        CSVH: SyncMapFormatCSV,
        CSVM: SyncMapFormatCSV,
        DFXP: SyncMapFormatTTML,
        EAF: SyncMapFormatEAF,
        JSON: SyncMapFormatJSON,
        RBSE: SyncMapFormatRBSE,
        SBV: SyncMapFormatSUB,
        SMIL: SyncMapFormatSMIL,
        SMILH: SyncMapFormatSMIL,
        SMILM: SyncMapFormatSMIL,
        SRT: SyncMapFormatSRT,
        SSV: SyncMapFormatSSV,
        SSVH: SyncMapFormatSSV,
        SSVM: SyncMapFormatSSV,
        SUB: SyncMapFormatSUB,
        TAB: SyncMapFormatTSV,
        TEXTGRID: SyncMapFormatTextGrid,
        TEXTGRID_LONG: SyncMapFormatTextGrid,
        TEXTGRID_SHORT: SyncMapFormatTextGrid,
        TSV: SyncMapFormatTSV,
        TSVH: SyncMapFormatTSV,
        TSVM: SyncMapFormatTSV,
        TTML: SyncMapFormatTTML,
        TXT: SyncMapFormatTXT,
        TXTH: SyncMapFormatTXT,
        TXTM: SyncMapFormatTXT,
        VTT: SyncMapFormatVTT,
        XML: SyncMapFormatXML,
        XML_LEGACY: SyncMapFormatXMLLegacy,
    }

    ALLOWED_VALUES = sorted(list(CODE_TO_CLASS.keys()))
    """ List of all the allowed values """
