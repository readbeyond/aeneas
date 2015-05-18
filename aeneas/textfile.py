#!/usr/bin/env python
# coding=utf-8

"""
A structure describing the properties of a text file.
"""

import BeautifulSoup
import codecs
import os
import re

import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf
from aeneas.idsortingalgorithm import IDSortingAlgorithm
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

class TextFileFormat(object):
    """
    Enumeration of the supported formats for text files.
    """

    PARSED = "parsed"
    """ The text file contains the fragments,
    one per line, with the following syntax:
    `id|text`
    where `id` is the fragment identifier
    and `text` is the text of the fragment. """

    PLAIN = "plain"
    """ The text file contains the fragments,
    one per line, without explicitly-assigned identifiers. """

    UNPARSED = "unparsed"
    """ The text file is a well-formed HTML/XHTML file,
    where the text fragments have already been marked up.

    The text fragments will be extracted by matching
    the ``id`` and/or ``class`` attributes of each elements
    with the provided regular expressions. """

    ALLOWED_VALUES = [PARSED, PLAIN, UNPARSED]
    """ List of all the allowed values """



class TextFragment(object):
    """
    A text fragment.

    :param identifier: the identifier of the fragment
    :type  identifier: unicode
    :param language: the language of the text of the fragment
    :type  language: string (from :class:`aeneas.language.Language` enumeration)
    :param text: the text of the fragment
    :type  text: unicode
    """

    TAG = "TextFragment"

    def __init__(self, identifier=None, language=None, text=None):
        self.identifier = identifier
        self.language = language
        self.text = text

    def __str__(self):
        return "%s %s" % (self.identifier, self.text)

    @property
    def identifier(self):
        """
        The identifier of the text fragment.

        :rtype: unicode 
        """
        return self.__identifier
    @identifier.setter
    def identifier(self, identifier):
        self.__identifier = identifier

    @property
    def language(self):
        """
        The language of the text fragment.

        :rtype: string (from :class:`aeneas.language.Language` enumeration)
        """
        return self.__language
    @language.setter
    def language(self, language):
        self.__language = language

    @property
    def text(self):
        """
        The text of the text fragment.

        :rtype: unicode
        """
        return self.__text
    @text.setter
    def text(self, text):
        self.__text = text



class TextFile(object):
    """
    A list of text fragments.

    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = "TextFile"

    def __init__(
            self,
            file_path=None,
            file_format=None,
            parameters=None,
            logger=None
        ):
        self.file_path = file_path
        self.file_format = file_format
        self.parameters = parameters
        self.fragments = []
        self.logger = Logger()
        if logger != None:
            self.logger = logger
        if (self.file_path != None) and (self.file_format != None):
            self._read_from_file()

    def __len__(self):
        return len(self.fragments)

    def __str__(self):
        return "\n".join([str(f) for f in self.fragments])

    def _log(self, message, severity=Logger.DEBUG):
        self.logger.log(message, severity, self.TAG)

    @property
    def fragments(self):
        """
        The current list of text fragments.

        :rtype: list of :class:`aeneas.textfile.TextFragment`
        """
        return self.__fragments
    @fragments.setter
    def fragments(self, fragments):
        self.__fragments = fragments

    def set_language(self, language):
        """
        Set the given language for all the text fragments.

        :param language: the language of the text fragments
        :type  language: string (from :class:`aeneas.language.Language` enumeration)
        """
        self._log("Setting language: '%s'" % language)
        for fragment in self.fragments:
            fragment.language = language

    def clear(self):
        """
        Clear the list of text fragments.
        """
        self._log("Clearing text fragments")
        self.fragments = []

    def read_from_list(self, lines):
        """
        Read text fragments from a given list of strings::

            [fragment_1, fragment_2, ..., fragment_n]

        :param lines: the text fragments
        :type  lines: list of strings
        """
        self._log("Reading text fragments from list")
        self._read_plain(lines)

    def read_from_list_with_ids(self, lines):
        """
        Read text fragments from a given list of lists::

            [[id_1, text_1], [id_2, text_2], ..., [id_n, text_n]].

        :param lines: the list of ``[id, text]`` fragments (see above)
        :type  lines: list of pairs (see above)
        """
        self._log("Reading text fragments from list with ids")
        self._create_text_fragments(lines)

    def _read_from_file(self):
        """
        Read text fragments from file.
        """

        # test if we can read the given file
        if not os.path.isfile(self.file_path):
            msg = "File '%s' cannot be read" % self.file_path
            self._log(msg, Logger.CRITICAL)
            raise OSError(msg)

        if self.file_format not in TextFileFormat.ALLOWED_VALUES:
            msg = "Text file format '%s' is not supported." % self.file_format
            self._log(msg, Logger.CRITICAL)
            raise ValueError(msg)

        # read the contents of the file
        self._log("Reading contents of file '%s'" % self.file_path)
        text_file = codecs.open(self.file_path, "r", "utf-8")
        lines = text_file.readlines()
        text_file.close()

        # clear text fragments
        self.clear()

        # parse the contents
        if self.file_format == TextFileFormat.PARSED:
            self._log("Reading from format PARSED")
            self._read_parsed(lines)
        if self.file_format == TextFileFormat.PLAIN:
            self._log("Reading from format PLAIN")
            self._read_plain(lines)
        if self.file_format == TextFileFormat.UNPARSED:
            self._log("Reading from format UNPARSED")
            self._read_unparsed(lines, self.parameters)

        # log the number of fragments
        self._log("Parsed %d fragments" % len(self.fragments))

    def _read_parsed(self, lines):
        """
        Read text fragments from a parsed format text file.

        :param lines: the lines of the parsed text file
        :type  lines: list of strings
        """
        self._log("Parsing fragments from parsed text format")
        pairs = []
        for line in lines:
            if gc.PARSED_TEXT_SEPARATOR in line:
                first, second = line.split(gc.PARSED_TEXT_SEPARATOR)
                identifier = first.strip()
                text = second.strip()
                pairs.append([identifier, text])
        self._create_text_fragments(pairs)

    def _read_plain(self, lines):
        """
        Read text fragments from a plain format text file.

        :param lines: the lines of the plain text file
        :type  lines: list of strings
        """
        self._log("Parsing fragments from plain text format")
        i = 1
        pairs = []
        for line in lines:
            identifier = "f" + str(i).zfill(6)
            text = line.strip()
            pairs.append([identifier, text])
            i += 1
        self._create_text_fragments(pairs)

    def _read_unparsed(self, lines, parameters):
        """
        Read text fragments from an unparsed format text file.

        :param lines: the lines of the unparsed text file
        :type  lines: list of strings
        :param parameters: additional parameters for parsing
                           (e.g., class/id regex strings)
        :type  parameters: dict
        """
        #
        # TODO better and/or parametric parsing,
        #      for example, removing tags but keeping text, etc.
        #
        self._log("Parsing fragments from unparsed text format")
        pairs = []

        # get filter attributes
        attributes = dict()
        if gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX in parameters:
            class_regex_string = parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_CLASS_REGEX]
            if class_regex_string != None:
                self._log("Regex for class: '%s'" % class_regex_string)
                class_regex = re.compile(r".*\b" + class_regex_string + r"\b.*")
                attributes['class'] = class_regex
        if gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX in parameters:
            id_regex_string = parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX]
            if id_regex_string != None:
                self._log("Regex for id: '%s'" % id_regex_string)
                id_regex = re.compile(r".*\b" + id_regex_string + r"\b.*")
                attributes['id'] = id_regex

        # get id sorting algorithm
        id_sort = IDSortingAlgorithm.UNSORTED
        if gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT in parameters:
            id_sort = parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT]
        self._log("Sorting text fragments using '%s'" % id_sort)

        # transform text in a soup object
        self._log("Creating soup")
        soup = BeautifulSoup.BeautifulSoup("\n".join(lines))

        # extract according to class_regex and id_regex
        text_from_id = dict()
        ids = []
        self._log("Finding elements matching attributes '%s'" % attributes)
        nodes = soup.findAll(attrs=attributes)
        for node in nodes:
            try:
                f_id = node['id']
                f_text = node.text
                text_from_id[f_id] = f_text
                ids.append(f_id)
            except KeyError:
                self._log("KeyError while parsing a node", Logger.WARNING)

        # sort by ID as requested
        self._log("Sorting text fragments")
        sorted_ids = IDSortingAlgorithm(id_sort).sort(ids)

        # append to fragments
        self._log("Appending fragments")
        for key in sorted_ids:
            pairs.append([key, text_from_id[key]])
        self._create_text_fragments(pairs)

    def _create_text_fragments(self, pairs):
        """
        Create text fragment objects and append them to this list.

        :param pairs: a list of lists, each being [id, text]
        :type  pairs: list of lists of two strings
        """
        self._log("Creating TextFragment objects")
        for pair in pairs:
            fragment = TextFragment(identifier=pair[0], text=pair[1])
            self.fragments.append(fragment)



