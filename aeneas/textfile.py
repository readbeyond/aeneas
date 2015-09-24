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
from aeneas.idsortingalgorithm import IDSortingAlgorithm
from aeneas.logger import Logger

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.1.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class TextFileFormat(object):
    """
    Enumeration of the supported formats for text files.
    """

    SUBTITLES = "subtitles"
    """
    The text file contains the fragments,
    each fragment is contained in one or more consecutive lines,
    separated by (at least) a blank line,
    without explicitly-assigned identifiers.
    Use this format if you want to output to SRT/TTML/VTT
    and you want to keep multilines in the output file::

        Fragment on a single row

        Fragment on two rows
        because it is quite long

        Another one liner

        Another fragment
        on two rows

    """

    PARSED = "parsed"
    """
    The text file contains the fragments,
    one per line, with the syntax ``id|text``,
    where `id` is a non-empty fragment identifier
    and `text` is the text of the fragment::

        f001|Text of the first fragment
        f002|Text of the second fragment
        f003|Text of the third fragment

    """

    PLAIN = "plain"
    """
    The text file contains the fragments,
    one per line, without explicitly-assigned identifiers::

        Text of the first fragment
        Text of the second fragment
        Text of the third fragment

    """

    UNPARSED = "unparsed"
    """
    The text file is a well-formed HTML/XHTML file,
    where the text fragments have already been marked up.

    The text fragments will be extracted by matching
    the ``id`` and/or ``class`` attributes of each elements
    with the provided regular expressions::

        <?xml version="1.0" encoding="UTF-8"?>
        <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en" xml:lang="en">
         <head>
          <meta charset="utf-8"/>
          <meta name="viewport" content="width=768,height=1024"/>
          <link rel="stylesheet" href="../Styles/style.css" type="text/css"/>
          <title>Sonnet I</title>
         </head>
         <body>
          <div id="divTitle">
           <h1><span class="ra" id="f001">I</span></h1>
          </div>
          <div id="divSonnet">
           <p>
            <span class="ra" id="f002">From fairest creatures we desire increase,</span><br/>
            <span class="ra" id="f003">That thereby beauty’s rose might never die,</span><br/>
            <span class="ra" id="f004">But as the riper should by time decease,</span><br/>
            <span class="ra" id="f005">His tender heir might bear his memory:</span><br/>
            <span class="ra" id="f006">But thou contracted to thine own bright eyes,</span><br/>
            <span class="ra" id="f007">Feed’st thy light’s flame with self-substantial fuel,</span><br/>
            <span class="ra" id="f008">Making a famine where abundance lies,</span><br/>
            <span class="ra" id="f009">Thy self thy foe, to thy sweet self too cruel:</span><br/>
            <span class="ra" id="f010">Thou that art now the world’s fresh ornament,</span><br/>
            <span class="ra" id="f011">And only herald to the gaudy spring,</span><br/>
            <span class="ra" id="f012">Within thine own bud buriest thy content,</span><br/>
            <span class="ra" id="f013">And tender churl mak’st waste in niggarding:</span><br/>
            <span class="ra" id="f014">Pity the world, or else this glutton be,</span><br/>
            <span class="ra" id="f015">To eat the world’s due, by the grave and thee.</span>
           </p>
          </div>
         </body>
        </html>

    """

    ALLOWED_VALUES = [SUBTITLES, PARSED, PLAIN, UNPARSED]
    """ List of all the allowed values """



class TextFragment(object):
    """
    A text fragment.

    :param identifier: the identifier of the fragment
    :type  identifier: unicode
    :param language: the language of the text of the fragment
    :type  language: string (from :class:`aeneas.language.Language` enumeration)
    :param lines: the lines in which text is split up
    :type  lines: list of unicode
    """

    TAG = "TextFragment"

    def __init__(self, identifier=None, language=None, lines=None):
        self.identifier = identifier
        self.language = language
        self.lines = lines

    def __str__(self):
        return ("%s %s" % (self.identifier, self.text)).encode('utf-8')

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
    def lines(self):
        """
        The lines of the text fragment.

        :rtype: list of unicode
        """
        return self.__lines
    @lines.setter
    def lines(self, lines):
        self.__lines = lines

    @property
    def text(self):
        """
        The text of the text fragment.

        :rtype: unicode
        """
        return u" ".join(self.lines)



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
        if logger is not None:
            self.logger = logger
        if (self.file_path is not None) and (self.file_format is not None):
            self._read_from_file()

    def __len__(self):
        return len(self.fragments)

    def __str__(self):
        return "\n".join([str(f) for f in self.fragments])

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
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
        self._log(["Setting language: '%s'", language])
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
        pairs = []
        for line in lines:
            pairs.append([line[0], [line[1]]])
        self._create_text_fragments(pairs)

    def _read_from_file(self):
        """
        Read text fragments from file.
        """

        # test if we can read the given file
        if not os.path.isfile(self.file_path):
            self._log(["File '%s' cannot be read", self.file_path], Logger.CRITICAL)
            raise OSError("Input file cannot be read")

        if self.file_format not in TextFileFormat.ALLOWED_VALUES:
            self._log(["Text file format '%s' is not supported.", self.file_format], Logger.CRITICAL)
            raise ValueError("Text file format not supported")

        # read the contents of the file
        self._log(["Reading contents of file '%s'", self.file_path])
        text_file = codecs.open(self.file_path, "r", "utf-8")
        lines = text_file.readlines()
        text_file.close()

        # clear text fragments
        self.clear()

        # parse the contents
        if self.file_format == TextFileFormat.SUBTITLES:
            self._log("Reading from format SUBTITLES")
            self._read_subtitles(lines)
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
        self._log(["Parsed %d fragments", len(self.fragments)])

    def _read_subtitles(self, lines):
        """
        Read text fragments from a subtitles format text file.

        :param lines: the lines of the subtitles text file
        :type  lines: list of strings
        """
        self._log("Parsing fragments from subtitles text format")
        lines = [line.strip() for line in lines]
        pairs = []
        i = 1
        current = 0
        while current < len(lines):
            line_text = lines[current]
            if len(line_text) > 0:
                fragment_lines = [line_text]
                following = current + 1
                while (following < len(lines) and (len(lines[following]) > 0)):
                    fragment_lines.append(lines[following])
                    following += 1
                identifier = "f" + str(i).zfill(6)
                pairs.append([identifier, fragment_lines])
                current = following
                i += 1
            current += 1
        self._create_text_fragments(pairs)

    def _read_parsed(self, lines):
        """
        Read text fragments from a parsed format text file.

        :param lines: the lines of the parsed text file
        :type  lines: list of strings
        """
        self._log("Parsing fragments from parsed text format")
        pairs = []
        for line in lines:
            pieces = line.split(gc.PARSED_TEXT_SEPARATOR)
            if len(pieces) == 2:
                identifier = pieces[0].strip()
                text = pieces[1].strip()
                if len(identifier) > 0:
                    pairs.append([identifier, [text]])
        self._create_text_fragments(pairs)

    def _read_plain(self, lines):
        """
        Read text fragments from a plain format text file.

        :param lines: the lines of the plain text file
        :type  lines: list of strings
        """
        self._log("Parsing fragments from plain text format")
        lines = [line.strip() for line in lines]
        pairs = []
        i = 1
        for line in lines:
            identifier = "f" + str(i).zfill(6)
            text = line.strip()
            pairs.append([identifier, [text]])
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
            if class_regex_string is not None:
                self._log(["Regex for class: '%s'", class_regex_string])
                class_regex = re.compile(r".*\b" + class_regex_string + r"\b.*")
                attributes['class'] = class_regex
        if gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX in parameters:
            id_regex_string = parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_REGEX]
            if id_regex_string is not None:
                self._log(["Regex for id: '%s'", id_regex_string])
                id_regex = re.compile(r".*\b" + id_regex_string + r"\b.*")
                attributes['id'] = id_regex

        # get id sorting algorithm
        id_sort = IDSortingAlgorithm.UNSORTED
        if gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT in parameters:
            id_sort = parameters[gc.PPN_JOB_IS_TEXT_UNPARSED_ID_SORT]
        self._log(["Sorting text fragments using '%s'", id_sort])

        # transform text in a soup object
        self._log("Creating soup")
        soup = BeautifulSoup.BeautifulSoup("\n".join(lines))

        # extract according to class_regex and id_regex
        text_from_id = dict()
        ids = []
        self._log(["Finding elements matching attributes '%s'", attributes])
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
            pairs.append([key, [text_from_id[key]]])
        self._create_text_fragments(pairs)

    def _create_text_fragments(self, pairs):
        """
        Create text fragment objects and append them to this list.

        :param pairs: a list of lists, each being [id, [line_1, ..., line_n]]
        :type  pairs: list of lists (see above)
        """
        self._log("Creating TextFragment objects")
        for pair in pairs:
            fragment = TextFragment(identifier=pair[0], lines=pair[1])
            self.fragments.append(fragment)



