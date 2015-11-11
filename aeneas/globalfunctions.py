#!/usr/bin/env python
# coding=utf-8

"""
Global common functions.
"""

from lxml import etree
import math
import os
import re
import shutil
import sys

import aeneas.globalconstants as gc

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.3.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

HHMMSS_MMM_PATTERN = re.compile(r"([0-9]*):([0-9]*):([0-9]*)\.([0-9]*)")
HHMMSS_MMM_PATTERN_COMMA = re.compile(r"([0-9]*):([0-9]*):([0-9]*),([0-9]*)")

def custom_tmp_dir():
    """
    Return the path of the temporary directory to use.

    On Linux and OS X, return the value of
    :class:`aeneas.globalconstants.TMP_PATH`
    (e.g., ``/tmp/``).

    On Windows, return ``None``, so that ``tempfile``
    will use the environment directory.

    :rtype: string (path)
    """
    if sys.platform in ["linux", "linux2", "darwin"]:
        return gc.TMP_PATH
    return None

def file_extension(path):
    """
    Return the file extension.

    Examples: ::

        /foo/bar.baz => baz
        None         => None

    :param path: the file path
    :type  path: string (path)
    :rtype: string (path)
    """
    if path is None:
        return None
    ext = os.path.splitext(os.path.basename(path))[1]
    if ext.startswith("."):
        ext = ext[1:]
    return ext

def file_name_without_extension(path):
    """
    Return the file name without extension.

    Examples: ::

        /foo/bar.baz => bar
        /foo/bar     => bar
        None         => None

    :param path: the file path
    :type  path: string (path)
    :rtype: string (path)
    """
    if path is None:
        return None
    return os.path.splitext(os.path.basename(path))[0]

def safe_unicode(string):
    """
    Safely decode a UTF-8 string into a unicode object.

    On error return ``None``.

    :param string: string value to be converted
    :type  string: string
    """
    if string is None:
        return None
    unic = None
    try:
        unic = string.decode("utf-8")
    except UnicodeDecodeError:
        pass
    return unic

def safe_float(string, default=None):
    """
    Safely parse a string into a float.

    On error return the ``default`` value.

    :param string: string value to be converted
    :type  string: string
    :param default: default value to be used in case of failure
    :type  default: float
    :rtype: float
    """
    value = default
    try:
        value = float(string)
    except TypeError:
        pass
    except ValueError:
        pass
    return value

def safe_int(string, default=None):
    """
    Safely parse a string into an int.

    On error return the ``default`` value.

    :param string: string value to be converted
    :type  string: string
    :param default: default value to be used in case of failure
    :type  default: int
    :rtype: int
    """
    value = safe_float(string, default)
    if value is not None:
        value = int(value)
    return value

def remove_bom(string):
    """
    Remove the BOM character (if any) from the given string.

    :param string: a string, possibly with BOM
    :type  string: string
    :rtype: string
    """
    tmp = None
    try:
        tmp = string.decode('utf-8-sig')
        tmp = tmp.encode('utf-8')
    except UnicodeError:
        pass
    return tmp

def norm_join(prefix, suffix):
    """
    Join ``prefix`` and ``suffix`` paths
    and return the resulting path, normalized.

    :param prefix: the prefix path
    :type  prefix: string (path)
    :param suffix: the suffix path
    :type  suffix: string (path)
    :rtype: string (path)
    """
    if (prefix is None) and (suffix is None):
        return "."
    if prefix is None:
        return os.path.normpath(suffix)
    if suffix is None:
        return os.path.normpath(prefix)
    return os.path.normpath(os.path.join(prefix, suffix))

def config_txt_to_string(string):
    """
    Convert the contents of a TXT config file
    into the corresponding configuration string ::

        key_1=value_1|key_2=value_2|...|key_n=value_n

    :param string: the contents of a TXT config file
    :type  string: string
    :rtype: string
    """
    if string is None:
        return None
    pairs = [l for l in string.splitlines() if len(l) > 0]
    return gc.CONFIG_STRING_SEPARATOR_SYMBOL.join(pairs)

def config_string_to_dict(string, result=None):
    """
    Convert a given configuration string ::

        key_1=value_1|key_2=value_2|...|key_n=value_n

    into the corresponding dictionary ::

        dictionary[key_1] = value_1
        dictionary[key_2] = value_2
        ...
        dictionary[key_n] = value_n

    :param string: the configuration string
    :type  string: string
    :rtype: dict
    """
    if string is None:
        return dict()
    pairs = string.split(gc.CONFIG_STRING_SEPARATOR_SYMBOL)
    return pairs_to_dict(pairs, result)

def config_xml_to_dict(contents, result, parse_job=True):
    """
    Convert the contents of a XML config file
    into the corresponding dictionary ::

        dictionary[key_1] = value_1
        dictionary[key_2] = value_2
        ...
        dictionary[key_n] = value_n

    :param contents: the XML configuration contents
    :type  contents: string
    :param parse_job: if ``True``, parse the job properties;
                      if ``False``, parse the tasks properties
    :type  parse_job: bool
    :rtype: dict (``parse_job=True``) or list of dict (``parse_job=False``)
    """
    try:
        root = etree.fromstring(contents)
        pairs = []
        if parse_job:
            # parse job
            for elem in root:
                if (elem.tag != gc.CONFIG_XML_TASKS_TAG) and (elem.text is not None):
                    pairs.append("%s%s%s" % (
                        elem.tag,
                        gc.CONFIG_STRING_ASSIGNMENT_SYMBOL,
                        elem.text.strip()
                    ))
            return pairs_to_dict(pairs)
        else:
            # parse tasks
            output_list = []
            for task in root.find(gc.CONFIG_XML_TASKS_TAG):
                if task.tag == gc.CONFIG_XML_TASK_TAG:
                    pairs = []
                    for elem in task:
                        if elem.text is not None:
                            pairs.append("%s%s%s" % (
                                elem.tag,
                                gc.CONFIG_STRING_ASSIGNMENT_SYMBOL,
                                elem.text.strip()
                            ))
                    output_list.append(pairs_to_dict(pairs))
            return output_list
    except:
        result.passed = False
        result.add_error("An error occurred while parsing XML file")
        return dict()

def config_dict_to_string(dictionary):
    """
    Convert a given config dictionary ::

        dictionary[key_1] = value_1
        dictionary[key_2] = value_2
        ...
        dictionary[key_n] = value_n

    into the corresponding string ::

        key_1=value_1|key_2=value_2|...|key_n=value_n

    :param dictionary: the config dictionary
    :type  dictionary: dict
    :rtype: string
    """
    parameters = []
    for key in dictionary:
        parameters.append("%s%s%s" % (
            key,
            gc.CONFIG_STRING_ASSIGNMENT_SYMBOL,
            dictionary[key]
        ))
    return gc.CONFIG_STRING_SEPARATOR_SYMBOL.join(parameters)

def pairs_to_dict(pairs, result=None):
    """
    Convert a given list of ``key=value`` strings ::

        key_1=value_1|key_2=value_2|...|key_n=value_n

    into the corresponding dictionary ::

        dictionary[key_1] = value_1
        dictionary[key_2] = value_2
        ...
        dictionary[key_n] = value_n

    :param pairs: the list of key=value strings
    :type  pairs: list of strings
    :rtype: dict
    """
    dictionary = dict()
    for pair in pairs:
        if len(pair) > 0:
            tokens = pair.split(gc.CONFIG_STRING_ASSIGNMENT_SYMBOL)
            if ((len(tokens) == 2) and
                    (len(tokens[0])) > 0 and
                    (len(tokens[1]) > 0)):
                dictionary[tokens[0]] = tokens[1]
            elif result is not None:
                result.add_warning("Invalid key=value string: '%s'" % pair)
    return dictionary

def copytree(source_directory, destination_directory, ignore=None):
    """
    Recursively copy the contents of a source directory
    into a destination directory.
    Both directories must exist.

    NOTE: this function does not copy the root directory ``source_directory``
    into ``destination_directory``.

    NOTE: ``shutil.copytree(src, dst)`` requires ``dst`` not to exist,
    so we cannot use for our purposes.

    NOTE: code adapted from http://stackoverflow.com/a/12686557

    :param source_directory: the source directory, already existing
    :type  source_directory: string (path)
    :param destination_directory: the destination directory, already existing
    :type  destination_directory: string (path)
    """
    if os.path.isdir(source_directory):
        if not os.path.isdir(destination_directory):
            os.makedirs(destination_directory)
        files = os.listdir(source_directory)
        if ignore is not None:
            ignored = ignore(source_directory, files)
        else:
            ignored = set()
        for f in files:
            if f not in ignored:
                copytree(
                    os.path.join(source_directory, f),
                    os.path.join(destination_directory, f),
                    ignore
                )
    else:
        shutil.copyfile(source_directory, destination_directory)

def ensure_parent_directory(path, get_parent=True):
    """
    Ensures the parent directory exists.

    :param path: the path of the file
    :type  path: string (path)
    :param get_parent: if True, ensure the parent directory of ``path`` exists;
                       if False, ensure ``path`` exists
    :type  get_paerent: bool
    :raise IOError: if the path cannot be created
    """
    parent_directory = os.path.abspath(path)
    if get_parent:
        parent_directory = os.path.dirname(parent_directory)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)

def time_from_ttml(string):
    """
    Parse the given ``SS.mmms`` string
    (TTML values have an "s" suffix, e.g. ``1.234s``)
    and return a float time value.

    :param string: the string to be parsed
    :type  string: unicode
    :rtype: float
    """
    if (string is None) or (len(string) < 2):
        return 0
    # strips "s" at the end
    string = string[:-1]
    return time_from_ssmmm(string)

def time_to_ttml(time_value):
    """
    Format the given time value into a ``SS.mmms`` string
    (TTML values have an "s" suffix, e.g. ``1.234s``).

    Examples: ::

        12        => 12.000s
        12.345    => 12.345s
        12.345432 => 12.345s
        12.345678 => 12.346s

    :param time_value: a time value, in seconds
    :type  time_value: float
    :rtype: string
    """
    if time_value is None:
        time_value = 0
    return "%ss" % time_to_ssmmm(time_value)

def time_from_ssmmm(string):
    """
    Parse the given ``SS.mmm`` string and return a float time value.

    :param string: the string to be parsed
    :type  string: unicode
    :rtype: float
    """
    if (string is None) or (len(string) < 1):
        return 0.000
    return float(string)

def time_to_ssmmm(time_value):
    """
    Format the given time value into a ``SS.mmm`` string.

    Examples: ::

        12        => 12.000
        12.345    => 12.345
        12.345432 => 12.345
        12.345678 => 12.346

    :param time_value: a time value, in seconds
    :type  time_value: float
    :rtype: string
    """
    if time_value is None:
        time_value = 0
    return "%.3f" % (time_value)

def time_from_hhmmssmmm(string, decimal_separator="."):
    """
    Parse the given ``HH:MM:SS.mmm`` string and return a float time value.

    :param string: the string to be parsed
    :type  string: unicode
    :rtype: float
    """
    v_length = 0.000
    try:
        if decimal_separator == ",":
            match = HHMMSS_MMM_PATTERN_COMMA.search(string)
        else:
            match = HHMMSS_MMM_PATTERN.search(string)
        if match is not None:
            v_h = int(match.group(1))
            v_m = int(match.group(2))
            v_s = int(match.group(3))
            v_f = float("0." + match.group(4))
            v_length = v_h * 3600 + v_m * 60 + v_s + v_f
    except:
        pass
    return v_length

def time_to_hhmmssmmm(time_value, decimal_separator="."):
    """
    Format the given time value into a ``HH:MM:SS.mmm`` string.

    Examples: ::

        12        => 00:00:12.000
        12.345    => 00:00:12.345
        12.345432 => 00:00:12.345
        12.345678 => 00:00:12.346
        83        => 00:01:23.000
        83.456    => 00:01:23.456
        83.456789 => 00:01:23.456
        3600      => 01:00:00.000
        3612.345  => 01:00:12.345

    :param time_value: a time value, in seconds
    :type  time_value: float
    :param decimal_separator: the decimal separator, default ``.``
    :type  decimal_separator: char
    :rtype: string
    """
    if time_value is None:
        time_value = 0
    tmp = time_value
    hours = int(math.floor(tmp / 3600))
    tmp -= (hours * 3600)
    minutes = int(math.floor(tmp / 60))
    tmp -= minutes * 60
    seconds = int(math.floor(tmp))
    tmp -= seconds
    milliseconds = int(math.floor(tmp * 1000))
    return "%02d:%02d:%02d%s%03d" % (
        hours,
        minutes,
        seconds,
        decimal_separator,
        milliseconds
    )

def time_to_srt(time_value):
    """
    Format the given time value into a ``HH:MM:SS,mmm`` string,
    as used in the SRT format.

    Examples: ::

        12        => 00:00:12,000
        12.345    => 00:00:12,345
        12.345432 => 00:00:12,345
        12.345678 => 00:00:12,346
        83        => 00:01:23,000
        83.456    => 00:01:23,456
        83.456789 => 00:01:23,456
        3600      => 01:00:00,000
        3612.345  => 01:00:12,345

    :param time_value: a time value, in seconds
    :type  time_value: float
    :rtype: string
    """
    return time_to_hhmmssmmm(time_value, ",")

def split_url(url):
    """
    Split the given URL base#anchor into [base, anchor],
    or [base, None] if no anchor is present.

    :param url: the url
    :type  url: str
    :rtype: list of str
    """
    if url is None:
        return [None, None]
    array = url.split("#")
    if len(array) == 1:
        array.append(None)
    elif len(array) > 2:
        # TODO raise an exception?
        array = array[0:2]
    return array

def can_run_c_extension(name=None):
    """
    Determine whether the given Python C extension loads correctly.

    If ``name`` is ``None``, tests all Python C extensions,
    and return ``True`` if and only if all load correctly.

    :param name: the name of the Python C extension to test
    :type  name: string
    :rtype: bool
    """
    def can_run_cdtw():
        """ Python C extension for computing DTW """
        try:
            import aeneas.cdtw
            return True
        except ImportError:
            return False
    def can_run_cmfcc():
        """ Python C extension for computing MFCC """
        try:
            import aeneas.cmfcc
            return True
        except ImportError:
            return False
    def can_run_cew():
        """ Python C extension for synthesizing with espeak """
        try:
            import aeneas.cew
            return True
        except ImportError:
            return False

    if name == "cdtw":
        return can_run_cdtw()
    elif name == "cmfcc":
        return can_run_cmfcc()
    elif name == "cew":
        return can_run_cew()
    else:
        if (os.name == "posix") and (os.uname()[0] == "Linux"):
            # Linux
            return can_run_cdtw() and can_run_cmfcc() and can_run_cew()
        else:
            # no cew for other OSes
            return can_run_cdtw() and can_run_cmfcc()

def file_can_be_written(path):
    """
    Return ``True`` if a file can be written at the given ``path``.

    IMPORTANT: this function will attempt to open the given ``path``
    in write mode, possibly destroying the file previously existing there.

    :param path: the file path
    :type  path: string (path)
    :rtype: bool
    """
    if path is None:
        return False
    try:
        test_file = open(path, "wb")
        test_file.close()
        delete_file(None, path)
        return True
    except IOError:
        return False

def directory_exists(path):
    """
    Return ``True`` if the given ``path`` string
    points to an existing directory.

    :param path: the file path
    :type  path: string (path)
    :rtype: bool
    """
    if (path is None) or (not os.path.isdir(path)):
        return False
    return True

def file_exists(path):
    """
    Return ``True`` if the given ``path`` string
    points to an existing file.

    :param path: the file path
    :type  path: string (path)
    :rtype: bool
    """
    if (path is None) or (not os.path.isfile(path)):
        return False
    return True

def delete_directory(path):
    """
    Safely delete a directory.

    :param path: the file path
    :type  path: string (path)
    """
    if path is not None:
        try:
            shutil.rmtree(path)
        except:
            pass

def delete_file(handler, path):
    """
    Safely delete file.

    :param handler: the file handler (as returned by tempfile)
    :type  handler: obj
    :param path: the file path
    :type  path: string (path)
    """
    if handler is not None:
        try:
            os.close(handler)
        except:
            pass
    if path is not None:
        try:
            os.remove(path)
        except:
            pass

def get_rel_path(path, from_path=None, absolute=False):
    """
    Get a path relative to the CWD or ``from_path``.

    :param path: the file path
    :type  path: string (path)
    :param from_path: the current directory; if None, use CWD
    :type  from_path: string (path)
    :param absolute: if True, output an absolute path
    :type  absolute: bool
    :rtype: string (path)
    """
    if path is None:
        return None
    if from_path is None:
        current_directory = os.path.dirname(os.path.realpath(sys.argv[0]))
    else:
        current_directory = from_path
    target = os.path.join(current_directory, path)
    rel_path = os.path.relpath(target)
    if absolute:
        return os.path.abspath(rel_path)
    else:
        return rel_path

def get_abs_path(path, from_file):
    """
    Get a path relative to the parent directory of ``from_file``,
    and return it as an absolute path.

    This method is intented to be called with ``__file__``
    as second argument.

    :param path: the file path
    :type  path: string (path)
    :param from_file: the reference file
    :type  from_file: string (path)
    :rtype: string (path)
    """
    return get_rel_path(path, os.path.dirname(from_file), True)

def human_readable_number(number, suffix=""):
    """
    Format the given number into a human-readable string.

    :param number: the number
    :type  number: int or float
    :param suffix: the unit of the number
    :type  suffix: str

    Code adapted from http://stackoverflow.com/a/1094933
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(number) < 1024.0:
            return "%3.1f%s%s" % (number, unit, suffix)
        number /= 1024.0
    return "%.1f%s%s" % (number, "Y", suffix)



