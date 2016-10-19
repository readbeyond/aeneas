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

"""
Enumeration of the supported output formats
for a synchronization map.
"""

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
    AUDH = SyncMapFormatAudacity.HUMAN
    AUDM = SyncMapFormatAudacity.MACHINE
    CSV = SyncMapFormatCSV.DEFAULT
    CSVH = SyncMapFormatCSV.HUMAN
    CSVM = SyncMapFormatCSV.MACHINE
    DFXP = SyncMapFormatTTML.DFXP
    EAF = SyncMapFormatEAF.DEFAULT
    JSON = SyncMapFormatJSON.DEFAULT
    RBSE = SyncMapFormatRBSE.DEFAULT
    SBV = SyncMapFormatSUB.SBV
    SMIL = SyncMapFormatSMIL.DEFAULT
    SMILH = SyncMapFormatSMIL.HUMAN
    SMILM = SyncMapFormatSMIL.MACHINE
    SRT = SyncMapFormatSRT.DEFAULT
    SSV = SyncMapFormatSSV.DEFAULT
    SSVH = SyncMapFormatSSV.HUMAN
    SSVM = SyncMapFormatSSV.MACHINE
    SUB = SyncMapFormatSUB.SUB
    TAB = SyncMapFormatTSV.TAB
    TEXTGRID = SyncMapFormatTextGrid.DEFAULT
    TEXTGRID_SHORT = SyncMapFormatTextGrid.SHORT
    TSV = SyncMapFormatTSV.DEFAULT
    TSVH = SyncMapFormatTSV.HUMAN
    TSVM = SyncMapFormatTSV.MACHINE
    TTML = SyncMapFormatTTML.DEFAULT
    TXT = SyncMapFormatTXT.DEFAULT
    TXTH = SyncMapFormatTXT.HUMAN
    TXTM = SyncMapFormatTXT.MACHINE
    VTT = SyncMapFormatVTT.DEFAULT
    XML = SyncMapFormatXML.DEFAULT
    XML_LEGACY = SyncMapFormatXMLLegacy.DEFAULT

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
