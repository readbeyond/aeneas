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
from aeneas.language import Language
from aeneas.logger import Logger

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl (www.readbeyond.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.0.1"
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
        if logger != None:
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
        self._log("Output format: '%s'" % sync_map_format)
        self._log("Output path: '%s'" % output_file_path)
        self._log("Output parameters: '%s'" % parameters)

        # create dir hierarchy, if needed
        parent_directory = os.path.dirname(output_file_path)
        if not os.path.exists(parent_directory):
            self._log("Creating directory '%s'" % parent_directory)
            os.makedirs(parent_directory)

        # check required parameters
        if sync_map_format == SyncMapFormat.SMIL:
            if not gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF in parameters:
                return False
            if not gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF in parameters:
                return False
            text_ref = parameters[gc.PPN_TASK_OS_FILE_SMIL_PAGE_REF]
            audio_ref = parameters[gc.PPN_TASK_OS_FILE_SMIL_AUDIO_REF]
            if (text_ref == None) or (audio_ref == None):
                return False

        try:
            # open file for writing
            self._log("Opening output file")
            output_file = codecs.open(output_file_path, "w", "utf-8")

            # output in the requested format
            if sync_map_format == SyncMapFormat.CSV:
                self._output_csv(output_file)
            elif sync_map_format == SyncMapFormat.JSON:
                self._output_json(output_file)
            elif sync_map_format == SyncMapFormat.SMIL:
                self._output_smil(output_file, parameters)
            elif sync_map_format == SyncMapFormat.SRT:
                self._output_srt(output_file)
            elif sync_map_format == SyncMapFormat.TAB:
                self._output_tab(output_file)
            elif sync_map_format == SyncMapFormat.TTML:
                self._output_ttml(output_file, parameters)
            elif sync_map_format == SyncMapFormat.TXT:
                self._output_txt(output_file)
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

    def _output_csv(self, output_file):
        """
        Output to CSV
        """
        for fragment in self.fragments:
            text = fragment.text_fragment
            output_file.write("%s,%s,%s,\"%s\"\n" % (
                text.identifier,
                gf.time_to_ssmmm(fragment.begin),
                gf.time_to_ssmmm(fragment.end),
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
        output_file.write("  <seq id=\"s%s\" epub:textref=\"%s\">\n" % (str(1).zfill(6), text_ref))
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
            output_file.write("%s\n" % text.text)
            output_file.write("\n")
            i += 1

    def _output_tab(self, output_file):
        """
        Output to TAB
        """
        for fragment in self.fragments:
            text = fragment.text_fragment
            output_file.write("%s\t%s\t%s\n" % (
                gf.time_to_ssmmm(fragment.begin),
                gf.time_to_ssmmm(fragment.end),
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
            output_file.write("    %s\n" % text.text)
            output_file.write("   </p>\n")
        output_file.write("  </div>\n")
        output_file.write(" </body>\n")
        output_file.write("</tt>")

    def _output_txt(self, output_file):
        """
        Output to TXT
        """
        for fragment in self.fragments:
            text = fragment.text_fragment
            output_file.write("%s %s %s \"%s\"\n" % (
                text.identifier,
                gf.time_to_ssmmm(fragment.begin),
                gf.time_to_ssmmm(fragment.end),
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
            output_file.write("%s\n" % text.text)
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
        return "%s %f %f" % (self.text_fragment.identifier, self.begin, self.end)

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
    """ Comma-separated values (CSV) """

    JSON = "json"
    """ JSON compatible with ``rb_smil_emulator.js`` """

    SMIL = "smil"
    """ SMIL (as EPUB 3 Media Overlay specification) """

    SRT = "srt"
    """ SRT """

    TAB = "tab"
    """ Tab-separated plain text, compatible with ``Audacity`` """

    TTML = "ttml"
    """ TTML """

    TXT = "txt"
    """ Space-separated plain text, compatible with ``SonicVisualizer`` """

    VTT = "vtt"
    """ WebVTT """

    XML = "xml"
    """ XML """

    ALLOWED_VALUES = [CSV, JSON, SMIL, SRT, TAB, TTML, TXT, VTT, XML]
    """ List of all the allowed values """



