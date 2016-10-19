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
TBW
"""

from __future__ import absolute_import
from __future__ import print_function
import json

from aeneas.syncmap.smfbase import SyncMapFormatBase
import aeneas.globalfunctions as gf


class SyncMapFormatRBSE(SyncMapFormatBase):

    TAG = u"SyncMapFormatRBSE"

    DEFAULT = "rbse"
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

    def parse(self, input_text, syncmap):
        contents_dict = json.loads(input_text)
        for fragment in contents_dict["smil_data"]:
            # TODO read text from additional text_file?
            self._add_fragment(
                syncmap=syncmap,
                identifier=fragment["id"],
                lines=[u""],
                begin=gf.time_from_ssmmm(fragment["begin"]),
                end=gf.time_from_ssmmm(fragment["end"])
            )

    def format(self, syncmap):
        smil_data = []
        smil_ids = []
        for fragment in syncmap.fragments:
            text = fragment.text_fragment
            smil_data.append({
                "id": text.identifier,
                "begin": gf.time_to_ssmmm(fragment.begin),
                "end": gf.time_to_ssmmm(fragment.end)
            })
            smil_ids.append(text.identifier)
        return gf.safe_unicode(
            json.dumps(
                obj={
                    "smil_ids": smil_ids,
                    "smil_data": smil_data
                },
                indent=1,
                sort_keys=True
            )
        )
