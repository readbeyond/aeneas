#!/usr/bin/env python
# coding=utf-8

"""
aeneas.tools is a collection of modules
that can be run as separate programs by the end user.
"""

import os
import sys

from aeneas.textfile import TextFile
from aeneas.textfile import TextFileFormat

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.3.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

def _check_format(text_format):
    if text_format not in TextFileFormat.ALLOWED_VALUES:
        print "[ERRO] File format '%s' is not allowed" % (text_format)
        print "[ERRO] Allowed text file formats: %s" % (" ".join(TextFileFormat.ALLOWED_VALUES))
        sys.exit(1)

def _get_text_from_string(string, logger):
    if not isinstance(string, unicode):
        try:
            string = string.decode("utf-8")
        except UnicodeDecodeError:
            print "[ERRO] Unable to decode the given text to Unicode"
            sys.exit(1)
    text_file = TextFile(logger=logger)
    text_file.read_from_list(string.split("|"))
    return text_file

def _get_text_from_file(text_file_path, text_format, parameters, logger):
    try:
        text_file = TextFile(text_file_path, text_format, parameters, logger=logger)
    except IOError:
        print "[ERRO] Cannot read file '%s'" % (text_file_path)
        print "[ERRO] Make sure the input file path is written/escaped correctly"
        sys.exit(1)
    return text_file

def get_text_file_object(text_file_path, text_format, parameters, logger):
    if text_format == "list":
        text_file = _get_text_from_string(text_file_path, logger)
    else:
        _check_format(text_format)
        text_file = _get_text_from_file(text_file_path, text_format, parameters, logger)
    return text_file



