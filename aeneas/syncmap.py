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

import codecs
import os

import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf
from aeneas.logger import Logger

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.1.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class SyncMap(object):
    """
    A synchronization map, that is, a list of
    :class:`aeneas.syncmap.SyncMapFragment`
    objects.
    """

    TAG = "SyncMap"

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

    def __str__(self):
        return "\n".join([str(f) for f in self.fragments])

    def append(self, fragment):
        """
        Append the given sync map fragment.

        :param fragment: the sync map fragment to be appended
        :type  fragment: :class:`aeneas.syncmap.SyncMapFragment`
        """
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
        self._log("Clearing sync map")
        self.fragments = []

    def output(self, sync_map_format, output_file_path, parameters=None):
        """
        Output the current sync map to file in the required format.

        Return ``True`` if the call succeeded,
        ``False`` if an error occurred.

        :param sync_map_format: the format of the sync map
        :type  sync_map_format: string (from :class:`aeneas.syncmap.SyncMapFormat` enumeration)
        :param output_file_path: the path to the output file to write
        :type  output_file_path: string (path)
        :param parameters: additional parameters (e.g., for SMIL output)
        :type  parameters: dict
        :rtype: bool
        """
        self._log(["Output format:     '%s'", sync_map_format])
        self._log(["Output path:       '%s'", output_file_path])
        self._log(["Output parameters: '%s'", parameters])

        # create dir hierarchy, if needed
        parent_directory = os.path.dirname(os.path.abspath(output_file_path))
        if not os.path.exists(parent_directory):
            self._log(["Creating directory '%s'", parent_directory])
            os.makedirs(parent_directory)

        # check required parameters
        if sync_map_format == SyncMapFormat.SMIL:
            if not gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF in parameters:
                return False
            if not gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF in parameters:
                return False
            text_ref = parameters[gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF]
            audio_ref = parameters[gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF]
            if (text_ref is None) or (audio_ref is None):
                return False

        try:
            # open file for writing
            self._log("Opening output file")
            output_file = codecs.open(output_file_path, "w", "utf-8")

            # output in the requested format
            if sync_map_format == SyncMapFormat.CSV:
                self._output_csv(output_file, gf.time_to_ssmmm)
            elif sync_map_format == SyncMapFormat.CSVH:
                self._output_csv(output_file, gf.time_to_hhmmssmmm)
            elif sync_map_format == SyncMapFormat.JSON:
                self._output_json(output_file)
            elif sync_map_format == SyncMapFormat.SMIL:
                self._output_smil(output_file, parameters)
            elif sync_map_format == SyncMapFormat.SRT:
                self._output_srt(output_file)
            elif sync_map_format == SyncMapFormat.SSV:
                self._output_ssv(output_file, gf.time_to_ssmmm)
            elif sync_map_format == SyncMapFormat.SSVH:
                self._output_ssv(output_file, gf.time_to_hhmmssmmm)
            elif sync_map_format == SyncMapFormat.TAB:
                self._output_tsv(output_file, gf.time_to_ssmmm)
            elif sync_map_format == SyncMapFormat.TSV:
                self._output_tsv(output_file, gf.time_to_ssmmm)
            elif sync_map_format == SyncMapFormat.TSVH:
                self._output_tsv(output_file, gf.time_to_hhmmssmmm)
            elif sync_map_format == SyncMapFormat.TTML:
                self._output_ttml(output_file, parameters)
            elif sync_map_format == SyncMapFormat.TXT:
                self._output_txt(output_file, gf.time_to_ssmmm)
            elif sync_map_format == SyncMapFormat.TXTH:
                self._output_txt(output_file, gf.time_to_hhmmssmmm)
            elif sync_map_format == SyncMapFormat.VTT:
                self._output_vtt(output_file)
            elif sync_map_format == SyncMapFormat.XML:
                self._output_xml(output_file)
            else:
                output_file.close()
                return False

            # close file and return
            output_file.close()
            return True
        except:
            return False

    def _output_csv(self, output_file, format_time):
        """
        Output to CSV
        """
        for fragment in self.fragments:
            text = fragment.text_fragment
            output_file.write("%s,%s,%s,\"%s\"\n" % (
                text.identifier,
                format_time(fragment.begin),
                format_time(fragment.end),
                text.text
            ))

    def _output_json(self, output_file):
        """
        Output to JSON
        """
        output_file.write("{\n")
        output_file.write(" \"smil_ids\": [\n")
        string = ",\n".join(["  \"%s\"" % (
            f.text_fragment.identifier
        ) for f in self.fragments])
        output_file.write(string)
        output_file.write("\n")
        output_file.write(" ],\n")
        output_file.write(" \"smil_data\": [\n")
        string = ",\n".join(["  { \"id\": \"%s\", \"begin\": %s, \"end\": %s }" % (
            f.text_fragment.identifier,
            gf.time_to_ssmmm(f.begin),
            gf.time_to_ssmmm(f.end),
        ) for f in self.fragments])
        output_file.write(string)
        output_file.write("\n")
        output_file.write(" ]\n")
        output_file.write("}\n")

    def _output_smil(self, output_file, parameters=None):
        """
        Output to SMIL
        """
        text_ref = parameters[gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF]
        audio_ref = parameters[gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF]
        output_file.write("<smil xmlns=\"http://www.w3.org/ns/SMIL\" xmlns:epub=\"http://www.idpf.org/2007/ops\" version=\"3.0\">\n")
        output_file.write(" <body>\n")
        output_file.write("  <seq id=\"s%s\" epub:textref=\"%s\">\n" % (
            str(1).zfill(6),
            text_ref
        ))
        i = 1
        for fragment in self.fragments:
            text = fragment.text_fragment
            output_file.write("   <par id=\"p%s\">\n" % (str(i).zfill(6)))
            output_file.write("    <text src=\"%s#%s\"/>\n" % (
                text_ref,
                text.identifier
            ))
            output_file.write("    <audio clipBegin=\"%s\" clipEnd=\"%s\" src=\"%s\"/>\n" % (
                gf.time_to_hhmmssmmm(fragment.begin),
                gf.time_to_hhmmssmmm(fragment.end),
                audio_ref
            ))
            output_file.write("   </par>\n")
            i += 1
        output_file.write("  </seq>\n")
        output_file.write(" </body>\n")
        output_file.write("</smil>\n")

    def _output_srt(self, output_file):
        """
        Output to SRT
        """
        i = 1
        for fragment in self.fragments:
            text = fragment.text_fragment
            output_file.write("%d\n" % i)
            output_file.write("%s --> %s\n" % (
                gf.time_to_srt(fragment.begin),
                gf.time_to_srt(fragment.end)
            ))
            for line in text.lines:
                output_file.write("%s\n" % line)
            output_file.write("\n")
            i += 1

    def _output_ssv(self, output_file, format_time):
        """
        Output to SSV
        """
        for fragment in self.fragments:
            text = fragment.text_fragment
            output_file.write("%s %s %s \"%s\"\n" % (
                format_time(fragment.begin),
                format_time(fragment.end),
                text.identifier,
                text.text
            ))

    def _output_tsv(self, output_file, format_time):
        """
        Output to TSV
        """
        for fragment in self.fragments:
            text = fragment.text_fragment
            output_file.write("%s\t%s\t%s\n" % (
                format_time(fragment.begin),
                format_time(fragment.end),
                text.identifier
            ))

    def _output_ttml(self, output_file, parameters=None):
        """
        Output to TTML
        """
        output_file.write("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n")
        output_file.write("<tt xmlns=\"http://www.w3.org/ns/ttml\">\n")
        # TODO add metadata from parameters here?
        # output_file.write(" <head/>\n")
        output_file.write(" <body>\n")
        output_file.write("  <div>\n")
        for fragment in self.fragments:
            text = fragment.text_fragment
            output_file.write("   <p xml:id=\"%s\" begin=\"%s\" end=\"%s\">\n" % (
                text.identifier,
                gf.time_to_ssmmm(fragment.begin),
                gf.time_to_ssmmm(fragment.end)
            ))
            output_file.write("    %s\n" % "<br/>\n    ".join(text.lines))
            output_file.write("   </p>\n")
        output_file.write("  </div>\n")
        output_file.write(" </body>\n")
        output_file.write("</tt>")

    def _output_txt(self, output_file, format_time):
        """
        Output to TXT
        """
        for fragment in self.fragments:
            text = fragment.text_fragment
            output_file.write("%s %s %s \"%s\"\n" % (
                text.identifier,
                format_time(fragment.begin),
                format_time(fragment.end),
                text.text
            ))

    def _output_vtt(self, output_file):
        """
        Output to WebVTT
        """
        output_file.write("WEBVTT\n\n")
        i = 1
        for fragment in self.fragments:
            text = fragment.text_fragment
            output_file.write("%d\n" % i)
            output_file.write("%s --> %s\n" % (
                gf.time_to_hhmmssmmm(fragment.begin),
                gf.time_to_hhmmssmmm(fragment.end)
            ))
            for line in text.lines:
                output_file.write("%s\n" % line)
            output_file.write("\n")
            i += 1

    def _output_xml(self, output_file):
        """
        Output to XML
        """
        output_file.write("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n")
        output_file.write("<map>\n")
        for fragment in self.fragments:
            text = fragment.text_fragment
            output_file.write(" <fragment>\n")
            output_file.write("  <identifier>%s</identifier>\n" % text.identifier)
            output_file.write("  <start>%s</start>\n" % gf.time_to_ssmmm(fragment.begin))
            output_file.write("  <end>%s</end>\n" % gf.time_to_ssmmm(fragment.end))
            output_file.write(" </fragment>\n")
        output_file.write("</map>")



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
    """

    TAG = "SyncMapFragment"

    def __init__(self, text_fragment=None, begin=None, end=None):
        self.text_fragment = text_fragment
        self.begin = begin
        self.end = end

    def __str__(self):
        return "%s %f %f" % (
            self.text_fragment.identifier,
            self.begin,
            self.end
        )

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



class SyncMapFormat(object):
    """
    Enumeration of the supported output formats
    for the synchronization maps.
    """

    CSV = "csv"
    """
    Comma-separated values (CSV)::

        f001,0.000,1.234,First fragment text
        f002,1.234,5.678,Second fragment text
        f003,5.678,7.890,Third fragment text

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

    JSON = "json"
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

    """

    SMIL = "smil"
    """
    SMIL (as in the EPUB 3 Media Overlay specification)::

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

    """

    SRT = "srt"
    """
    SRT caption/subtitle format::

        1
        00:00:00,000 --> 00:00:01,234
        First fragment text

        2
        00:00:01,234 --> 00:00:05,678
        Second fragment text

        3
        00:00:05,678 --> 00:00:07,890
        Third fragment text

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

    TAB = "tab"
    """
    Deprecated, it will be removed in v2.0.0. Use TSV instead.

    .. deprecated:: 1.0.3
    """

    TSV = "tsv"
    """
    Tab-separated plain text, compatible with ``Audacity``::

        0.000   1.234   f001
        1.234   5.678   f002
        5.678   7.890   f003

    .. versionadded:: 1.0.3
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

    TTML = "ttml"
    """
    TTML caption/subtitle format::

        <?xml version="1.0" encoding="UTF-8" ?>
        <tt xmlns="http://www.w3.org/ns/ttml">
         <body>
          <div>
           <p xml:id="f001" begin="0.000" end="1.234">
            First fragment text
           </p>
           <p xml:id="f002" begin="1.234" end="5.678">
            Second fragment text
           </p>
           <p xml:id="f003" begin="5.678" end="7.890">
            Third fragment text
           </p>
          </div>
         </body>
        </tt>

    """

    TXT = "txt"
    """
    Space-separated plain text, compatible with ``SonicVisualizer``::

        f001 0.000 1.234 "First fragment text"
        f002 1.234 5.678 "Second fragment text"
        f003 5.678 7.890 "Third fragment text"

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

    VTT = "vtt"
    """
    WebVTT caption/subtitle format::

        WEBVTT

        1
        00:00:00,000 --> 00:00:01,234
        First fragment text

        2
        00:00:01,234 --> 00:00:05,678
        Second fragment text

        3
        00:00:05,678 --> 00:00:07,890
        Third fragment text

    """

    XML = "xml"
    """
    XML::

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
        JSON,
        SMIL,
        SRT,
        SSV,
        SSVH,
        TAB,
        TSV,
        TSVH,
        TTML,
        TXT,
        TXTH,
        VTT,
        XML
    ]
    """ List of all the allowed values """



