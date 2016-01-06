#!/usr/bin/env python
# coding=utf-8

"""
A SyncMap is an abstraction for a synchronization map,
that is, a map from text fragments to time intervals.

This module contains three classes:

1. :class:`aeneas.syncmap.SyncMap` is the main class,
   representing a list of sync map fragments,
   and exposing a function to output it to file in several formats.
2. :class:`aeneas.syncmap.SyncMapFragment`
   represents the single sync map fragments,
   that is, a :class:`aeneas.textfile.TextFragment`
   and the corrisponding pair of `begin` and `end` times.
3. :class:`aeneas.syncmap.SyncMapFormat` is an enumeration
   of the supported output formats.
"""

from __future__ import absolute_import
from __future__ import print_function
from functools import partial
from itertools import chain
from lxml import etree
import io
import json
import os

from aeneas.logger import Logger
from aeneas.textfile import TextFragment
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.4.0"
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
    for the synchronization maps.
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

    .. versionadded:: 1.0.4
    """

    CSVM = "csvm"
    """
    Comma-separated values (CSV),
    with machine-readable time values::

        f001,0.000,1.234,First fragment text
        f002,1.234,5.678,Second fragment text
        f003,5.678,7.890,Third fragment text

    .. versionadded:: 1.2.0
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
           "lines": [
            "First fragment text"
           ]
          },
          {
           "id": "f002",
           "language": "en",
           "begin": 1.234,
           "end": 5.678,
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
           "lines": [
            "Third fragment text",
            "Second line of third fragment"
           ]
          }
         ]
        }

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

    .. versionadded:: 1.2.0
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
          <seq id="s000001" epub:textref="p001.xhtml">
           <par id="p000001">
            <text src="p001.xhtml#f001"/>
            <audio clipBegin="00:00:00.000" clipEnd="00:00:01.234" src="../Audio/p001.mp3"/>
           </par>
           <par id="p000002">
            <text src="p001.xhtml#f002"/>
            <audio clipBegin="00:00:01.234" clipEnd="00:00:05.678" src="../Audio/p001.mp3"/>
           </par>
           <par id="p000003">
            <text src="p001.xhtml#f003"/>
            <audio clipBegin="00:00:05.678" clipEnd="00:00:07.890" src="../Audio/p001.mp3"/>
           </par>
          </seq>
         </body>
        </smil>

    .. versionadded:: 1.2.0
    """

    SMILM = "smilm"
    """
    SMIL (as in the EPUB 3 Media Overlay specification),
    with machine-readable time values::

        <smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">
         <body>
          <seq id="s000001" epub:textref="p001.xhtml">
           <par id="p000001">
            <text src="p001.xhtml#f001"/>
            <audio clipBegin="0.000" clipEnd="1.234" src="../Audio/p001.mp3"/>
           </par>
           <par id="p000002">
            <text src="p001.xhtml#f002"/>
            <audio clipBegin="1.234" clipEnd="5.678" src="../Audio/p001.mp3"/>
           </par>
           <par id="p000003">
            <text src="p001.xhtml#f003"/>
            <audio clipBegin="5.678" clipEnd="7.890" src="../Audio/p001.mp3"/>
           </par>
          </seq>
         </body>
        </smil>

    .. versionadded:: 1.2.0
    """

    SRT = "srt"
    """
    SRT caption/subtitle format
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

    """

    SSV = "ssv"
    """
    Space-separated plain text::

        0.000 1.234 f001 "First fragment text"
        1.234 5.678 f002 "Second fragment text"
        5.678 7.890 f003 "Third fragment text"

    .. versionadded:: 1.0.4
    """

    SSVH = "ssvh"
    """
    Space-separated plain text,
    with human-readable time values::

        00:00:00.000 00:00:01.234 f001 "First fragment text"
        00:00:01.234 00:00:05.678 f002 "Second fragment text"
        00:00:05.678 00:00:07.890 f003 "Third fragment text"

    .. versionadded:: 1.0.4
    """

    SSVM = "ssvm"
    """
    Space-separated plain text,
    with machine-readable time values::

        0.000 1.234 f001 "First fragment text"
        1.234 5.678 f002 "Second fragment text"
        5.678 7.890 f003 "Third fragment text"

    .. versionadded:: 1.2.0
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

    .. versionadded:: 1.2.0
    """

    VTT = "vtt"
    """
    WebVTT caption/subtitle format
    (it might have multiple lines per fragment)::

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

    """

    XML = "xml"
    """
    XML (it might have multiple lines per fragment)::

        <?xml version="1.0" encoding="UTF-8" ?>
        <map>
         <fragment id="f001" begin="0.000" end="1.234">
          <line>First fragment text</line>
         </fragment>
         <fragment id="f002" begin="1.234" end="5.678">
          <line>Second fragment text</line>
          <line>Second line of second fragment</line>
         </fragment>
         <fragment id="f003" begin="5.678" end="7.890">
          <line>Third fragment text</line>
          <line>Second line of third fragment</line>
         </fragment>
        </map>

    """

    XML_LEGACY = "xml_legacy"
    """
    XML, legacy format.

    Deprecated, it will be removed in v2.0.0. Use XML instead.

    .. deprecated:: 1.2.0

    ::

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

    """


    ALLOWED_VALUES = [
        CSV,
        CSVH,
        CSVM,
        JSON,
        RBSE,
        SMIL,
        SMILH,
        SMILM,
        SRT,
        SSV,
        SSVH,
        SSVM,
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



class SyncMap(object):
    """
    A synchronization map, that is, a list of
    :class:`aeneas.syncmap.SyncMapFragment`
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

    def __init__(self, logger=None):
        self.fragments = []
        self.logger = Logger()
        if logger is not None:
            self.logger = logger

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def __len__(self):
        return len(self.fragments)

    def __unicode__(self):
        return u"\n".join([f.__unicode__() for f in self.fragments])

    def __str__(self):
        return gf.safe_str(self.__unicode__())

    def to_json(self):
        """
        Return a JSON representation of the sync map.

        :rtype: Unicode string

        .. versionadded:: 1.3.1
        """
        output_fragments = []
        for fragment in self.fragments:
            text = fragment.text_fragment
            output_fragments.append({
                "id" : text.identifier,
                "language" : text.language,
                "lines" : text.lines,
                "begin" : gf.time_to_ssmmm(fragment.begin),
                "end" : gf.time_to_ssmmm(fragment.end)
            })
        return gf.safe_unicode(
            json.dumps({"fragments": output_fragments}, indent=1, sort_keys=True)
        )

    def append_fragment(self, fragment):
        """
        Append the given sync map fragment.

        :param fragment: the sync map fragment to be appended
        :type  fragment: :class:`aeneas.syncmap.SyncMapFragment`

        :raise TypeError: if ``fragment`` is ``None`` or
                          it is not an instance of ``SyncMapFragment``
        """
        if not isinstance(fragment, SyncMapFragment):
            raise TypeError("fragment is not an instance of SyncMapFragment")
        self.fragments.append(fragment)

    @property
    def fragments(self):
        """
        The current list of sync map fragments.

        :rtype: list of :class:`aeneas.syncmap.SyncMapFragment`
        """
        return self.__fragments
    @fragments.setter
    def fragments(self, fragments):
        self.__fragments = fragments

    def clear(self):
        """
        Clear the sync map.
        """
        self._log(u"Clearing sync map")
        self.fragments = []

    def output_html_for_tuning(
            self,
            audio_file_path,
            output_file_path,
            parameters=None
    ):
        """
        Output an HTML file for fine tuning the sync map manually.

        :param audio_file_path: the path to the associated audio file
        :type  audio_file_path: string (path)
        :param output_file_path: the path to the output file to write
        :type  output_file_path: string (path)
        :param parameters: additional parameters
        :type  parameters: dict

        .. versionadded:: 1.3.1
        """
        if not gf.file_can_be_written(output_file_path):
            raise OSError("Cannot output HTML file '%s' (wrong permissions?)" % output_file_path)
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
            u"fragments = (%s).fragments;" % self.to_json()
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
        and append them the current (this) sync map.

        Return ``True`` if the call succeeded,
        ``False`` if an error occurred.

        :param sync_map_format: the format of the sync map
        :type  sync_map_format: :class:`aeneas.syncmap.SyncMapFormat` enum
        :param input_file_path: the path to the input file to read
        :type  input_file_path: string (path)
        :param parameters: additional parameters (e.g., for SMIL input)
        :type  parameters: dict

        :raise ValueError: if ``sync_map_format`` is ``None`` or it is not an allowed value
        :raise OSError: if ``input_file_path`` does not exist
        """
        map_read_function = {
            SyncMapFormat.CSV: partial(self._read_csv, parse_time=gf.time_from_ssmmm),
            SyncMapFormat.CSVH: partial(self._read_csv, parse_time=gf.time_from_hhmmssmmm),
            SyncMapFormat.CSVM: partial(self._read_csv, parse_time=gf.time_from_ssmmm),
            SyncMapFormat.JSON: self._read_json,
            SyncMapFormat.RBSE: self._read_rbse,
            SyncMapFormat.SMIL: self._read_smil,
            SyncMapFormat.SMILH: self._read_smil,
            SyncMapFormat.SMILM: self._read_smil,
            SyncMapFormat.SRT: self._read_srt,
            SyncMapFormat.SSV: partial(self._read_ssv, parse_time=gf.time_from_ssmmm),
            SyncMapFormat.SSVH: partial(self._read_ssv, parse_time=gf.time_from_hhmmssmmm),
            SyncMapFormat.SSVM: partial(self._read_ssv, parse_time=gf.time_from_ssmmm),
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
            raise ValueError("Sync map format is None")
        if sync_map_format not in map_read_function:
            raise ValueError("Sync map format '%s' is not allowed" % sync_map_format)
        if not gf.file_can_be_read(input_file_path):
            raise OSError("Cannot read sync map file '%s' (wrong permissions?)" % input_file_path)

        self._log([u"Input format:     '%s'", sync_map_format])
        self._log([u"Input path:       '%s'", input_file_path])
        self._log([u"Input parameters: '%s'", parameters])

        # open file for reading
        self._log(u"Opening input file")
        with io.open(input_file_path, "r", encoding="utf-8") as input_file:
            map_read_function[sync_map_format](input_file)

        # overwrite language if requested
        language = gf.safe_get(parameters, gc.PPN_SYNCMAP_LANGUAGE, None)
        if language is not None:
            self._log([u"Overwriting language to '%s'", language])
            for fragment in self.fragments:
                fragment.text_fragment.language = language

    def write(self, sync_map_format, output_file_path, parameters=None):
        """
        Write the current sync map to file in the required format.

        Return ``True`` if the call succeeded,
        ``False`` if an error occurred.

        :param sync_map_format: the format of the sync map
        :type  sync_map_format: :class:`aeneas.syncmap.SyncMapFormat` enum
        :param output_file_path: the path to the output file to write
        :type  output_file_path: string (path)
        :param parameters: additional parameters (e.g., for SMIL output)
        :type  parameters: dict

        :raise ValueError: if ``sync_map_format`` is ``None`` or it is not an allowed value
        :raise TypeError: if a required parameter is missing
        :raise OSError: if ``output_file_path`` cannot be written
        """
        map_write_function = {
            SyncMapFormat.CSV: partial(self._write_csv, format_time=gf.time_to_ssmmm),
            SyncMapFormat.CSVH: partial(self._write_csv, format_time=gf.time_to_hhmmssmmm),
            SyncMapFormat.CSVM: partial(self._write_csv, format_time=gf.time_to_ssmmm),
            SyncMapFormat.JSON: self._write_json,
            SyncMapFormat.RBSE: self._write_rbse,
            SyncMapFormat.SMIL: partial(self._write_smil, format_time=gf.time_to_hhmmssmmm, parameters=parameters),
            SyncMapFormat.SMILH: partial(self._write_smil, format_time=gf.time_to_hhmmssmmm, parameters=parameters),
            SyncMapFormat.SMILM: partial(self._write_smil, format_time=gf.time_to_ssmmm, parameters=parameters),
            SyncMapFormat.SRT: self._write_srt,
            SyncMapFormat.SSV: partial(self._write_ssv, format_time=gf.time_to_ssmmm),
            SyncMapFormat.SSVH: partial(self._write_ssv, format_time=gf.time_to_hhmmssmmm),
            SyncMapFormat.SSVM: partial(self._write_ssv, format_time=gf.time_to_ssmmm),
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
            raise ValueError("Sync map format is None")
        if sync_map_format not in map_write_function:
            raise ValueError("Sync map format '%s' is not allowed" % sync_map_format)
        if not gf.file_can_be_written(output_file_path):
            raise OSError("Cannot output sync map file '%s' (wrong permissions?)" % output_file_path)

        self._log([u"Output format:     '%s'", sync_map_format])
        self._log([u"Output path:       '%s'", output_file_path])
        self._log([u"Output parameters: '%s'", parameters])

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
                    msg = u"Parameter %s must be specified for format %s" % (key, sync_map_format)
                    self._log(msg, Logger.CRITICAL)
                    raise SyncMapMissingParameterError(msg)

        # open file for writing
        self._log(u"Opening output file")
        with io.open(output_file_path, "w", encoding="utf-8") as output_file:
            map_write_function[sync_map_format](output_file)

    def _read_csv(self, input_file, parse_time):
        """ Read from CSV file """
        for line in input_file.readlines():
            split = line.strip().split(u",")
            self.append_fragment(
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

    def _read_json(self, input_file):
        """ Read from JSON file """
        contents_dict = json.loads(input_file.read())
        for fragment in contents_dict["fragments"]:
            self.append_fragment(
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
        output_file.write(self.to_json())

    def _read_rbse(self, input_file):
        """ Read from RBSE file """
        contents_dict = json.loads(input_file.read())
        for fragment in contents_dict["smil_data"]:
            self.append_fragment(
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
            self.append_fragment(
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
        seq_elem.attrib["id"] = "s" + str(1).zfill(6)
        seq_elem.attrib["{%s}textref" % epub_ns] = text_ref
        i = 1
        for fragment in self.fragments:
            text = fragment.text_fragment
            par_elem = etree.SubElement(seq_elem, "{%s}par" % smil_ns)
            par_elem.attrib["id"] = "p" + str(i).zfill(6)
            text_elem = etree.SubElement(par_elem, "{%s}text" % smil_ns)
            text_elem.attrib["src"] = "%s#%s" % (text_ref, text.identifier)
            audio_elem = etree.SubElement(par_elem, "{%s}audio" % smil_ns)
            audio_elem.attrib["src"] = audio_ref
            audio_elem.attrib["clipBegin"] = format_time(fragment.begin)
            audio_elem.attrib["clipEnd"] = format_time(fragment.end)
            i += 1

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
                        self.append_fragment(
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

    def _read_ssv(self, input_file, parse_time):
        """ Read from SSV file """
        for line in input_file.readlines():
            split = line.strip().split(" ")
            self.append_fragment(
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
            self.append_fragment(
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
            self.append_fragment(
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

        # write tree
        self._write_tree_to_file(tt_elem, output_file)

    def _read_txt(self, input_file, parse_time):
        """ Read from TXT file """
        for line in input_file.readlines():
            split = line.strip().split(" ")
            self.append_fragment(
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
                        self.append_fragment(
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
            self.append_fragment(
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
        map_elem = etree.Element("map")
        for fragment in self.fragments:
            fragment_elem = etree.SubElement(map_elem, "fragment")
            fragment_elem.attrib["id"] = fragment.text_fragment.identifier
            fragment_elem.attrib["begin"] = gf.time_to_ssmmm(fragment.begin)
            fragment_elem.attrib["end"] = gf.time_to_ssmmm(fragment.end)
            for line in fragment.text_fragment.lines:
                line_elem = etree.SubElement(fragment_elem, "line")
                line_elem.text = line
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
            self.append_fragment(
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
        Write an lxml tree to the given output file.
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
        Given an lxml node, get lines from node.text,
        where the line separator is "<br xmlns=... />".
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
    a text fragment and an associated time interval.

    :param text_fragment: the text fragment
    :type  text_fragment: :class:`aeneas.textfile.TextFragment`
    :param begin: the begin time of the audio interval
    :type  begin: float
    :param end: the end time of the audio interval
    :type  end: float
    :param confidence: the confidence of the audio timing
    :type  confidence: float
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
        return u"%s %.3f %.3f %.3f" % (
            self.text_fragment.identifier,
            self.begin,
            self.end,
            self.confidence
        )

    def __str__(self):
        return gf.safe_str(self.__unicode__())

    def __len__(self):
        return self.end - self.begin

    @property
    def audio_duration(self):
        """
        The audio duration of this sync map fragment,
        as end time minus begin time.

        :rtype: float
        """
        return len(self)

    @property
    def text_fragment(self):
        """
        The text fragment associated with this sync map fragment.

        :rtype: :class:`aeneas.textfile.TextFragment`
        """
        return self.__text_fragment
    @text_fragment.setter
    def text_fragment(self, text_fragment):
        self.__text_fragment = text_fragment

    @property
    def begin(self):
        """
        The begin time of this sync map fragment.

        :rtype: float
        """
        return self.__begin
    @begin.setter
    def begin(self, begin):
        self.__begin = begin

    @property
    def end(self):
        """
        The end time of this sync map fragment.

        :rtype: float
        """
        return self.__end
    @end.setter
    def end(self, end):
        self.__end = end

    @property
    def confidence(self):
        """
        The confidence of the audio timing, from 0 to 1.

        NOTE: currently always set to 1.0.

        :rtype: float
        """
        return self.__confidence
    @confidence.setter
    def confidence(self, confidence):
        self.__confidence = confidence



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



