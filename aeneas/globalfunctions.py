#!/usr/bin/env python
# coding=utf-8

"""
Global common functions.
"""

from __future__ import absolute_import
from __future__ import print_function
from lxml import etree
import math
import io
import os
import re
import shutil
import sys
import tempfile
import uuid

import aeneas.globalconstants as gc

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

HHMMSS_MMM_PATTERN = re.compile(r"([0-9]*):([0-9]*):([0-9]*)\.([0-9]*)")
HHMMSS_MMM_PATTERN_COMMA = re.compile(r"([0-9]*):([0-9]*):([0-9]*),([0-9]*)")
PY2 = (sys.version_info[0] == 2)

def uuid_string():
    """
    Return a uuid4 as a Unicode string.

    :rtype: Unicode string
    """
    return safe_unicode(str(uuid.uuid4())).lower()

def custom_tmp_dir():
    """
    Return the path of the temporary directory to use.

    On POSIX OSes (Linux and OS X), return the value of
    :class:`aeneas.globalconstants.TMP_PATH`
    (e.g., ``/tmp/``).

    On Windows, return ``None``, so that ``tempfile``
    will use the directory specified by the
    environment/user TMP/TEMP variable.

    :rtype: string (path)
    """
    if is_posix():
        return gc.TMP_PATH
    return None

def tmp_directory():
    """
    Return the path of a temporary directory created by ``tempfile``.

    :rtype: string (path)
    """
    return tempfile.mkdtemp(dir=custom_tmp_dir())

def tmp_file(suffix=u""):
    """
    Return a (handler, path) tuple
    for a temporary file with given suffix created by ``tempfile``.

    :param suffix: the suffix (e.g., the extension) of the file
    :type  suffix: Unicode string
    :rtype: tuple
    """
    return tempfile.mkstemp(suffix=suffix, dir=custom_tmp_dir())

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

def safe_get(dictionary, key, default_value, can_return_none=True):
    """
    Safely perform a dictionary get,
    returning the default value if the key is not found.

    :param dictionary: the dictionary
    :type  dictionary: dict
    :param key: the key
    :type  key: Unicode string
    :param default_value: the default value to be returned
    :type  default_value: variant
    :param can_return_none: if ``True``, the function can return ``None``;
                            otherwise, return ``default_value`` even if the
                            dictionary lookup succeeded
    :type  can_return_none: bool
    :rtype: variant
    """
    return_value = default_value
    try:
        return_value = dictionary[key]
        if (return_value is None) and (not can_return_none):
            return_value = default_value
    except (KeyError, TypeError):
        # KeyError if key is not present in dictionary
        # TypeError if dictionary is None
        pass
    return return_value

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

    Leading and trailing blank characters will be stripped
    and empty lines (after stripping) will be ignored.

    :param string: the contents of a TXT config file
    :type  string: Unicode string
    :rtype: Unicode string
    """
    if string is None:
        return None
    pairs = [l.strip() for l in string.splitlines() if len(l.strip()) > 0]
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
        return {}
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
    :type  contents: bytes
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
                    pairs.append(u"%s%s%s" % (
                        safe_unicode(elem.tag),
                        gc.CONFIG_STRING_ASSIGNMENT_SYMBOL,
                        safe_unicode(elem.text.strip())
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
                            pairs.append(u"%s%s%s" % (
                                safe_unicode(elem.tag),
                                gc.CONFIG_STRING_ASSIGNMENT_SYMBOL,
                                safe_unicode(elem.text.strip())
                            ))
                    output_list.append(pairs_to_dict(pairs))
            return output_list
    except:
        if result is not None:
            result.passed = False
            result.add_error("An error occurred while parsing XML file")
        if parse_job:
            return {}
        else:
            return []

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
        parameters.append(u"%s%s%s" % (
            key,
            gc.CONFIG_STRING_ASSIGNMENT_SYMBOL,
            dictionary[key]
        ))
    return gc.CONFIG_STRING_SEPARATOR_SYMBOL.join(parameters)

def pairs_to_dict(pairs, result=None):
    """
    Convert a given list of ``key=value`` strings ::

        ["key_1=value_1", "key_2=value_2", ..., "key_n=value_n"]

    into the corresponding dictionary ::

        dictionary[key_1] = value_1
        dictionary[key_2] = value_2
        ...
        dictionary[key_n] = value_n

    :param pairs: the list of key=value strings
    :type  pairs: list of strings
    :rtype: dict
    """
    dictionary = {}
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

def ensure_parent_directory(path, ensure_parent=True):
    """
    Ensures the parent directory exists.

    :param path: the path of the file
    :type  path: string (path)
    :param ensure_parent: if True, ensure the parent directory of ``path`` exists;
                       if False, ensure ``path`` exists
    :type  ensure_parent: bool
    :raise OSError: if the path cannot be created
    """
    parent_directory = os.path.abspath(path)
    if ensure_parent:
        parent_directory = os.path.dirname(parent_directory)
    if not os.path.exists(parent_directory):
        try:
            os.makedirs(parent_directory)
        except (IOError, OSError):
            raise OSError("Directory '%s' cannot be created" % parent_directory)

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
    :param decimal_separator: the decimal separator to be used
    :type  decimal_separator: string
    :rtype: float
    """
    if decimal_separator == ",":
        pattern = HHMMSS_MMM_PATTERN_COMMA
    else:
        pattern = HHMMSS_MMM_PATTERN
    v_length = 0.000
    try:
        match = pattern.search(string)
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
    Split the given URL base#anchor into (base, anchor),
    or (base, None) if no anchor is present.

    :param url: the url
    :type  url: str
    :rtype: list of str
    """
    if url is None:
        return (None, None)
    array = url.split("#")
    if len(array) == 1:
        array.append(None)
    elif len(array) > 2:
        # TODO raise an exception?
        array = array[0:2]
    return tuple(array)

def is_posix():
    """
    Return ``True`` if running on a POSIX OS.

    :rtype: bool
    """
    # from https://docs.python.org/2/library/os.html#os.name
    # the registered values of os.name are:
    # "posix", "nt", "os2", "ce", "java", "riscos"
    return os.name == "posix"

def is_linux():
    """
    Return ``True`` if running on Linux.

    :rtype: bool
    """
    return (is_posix()) and (os.uname()[0] == "Linux")

def is_osx():
    """
    Return ``True`` if running on Mac OS X (Darwin).

    :rtype: bool
    """
    return (is_posix()) and (os.uname()[0] == "Darwin")

def is_windows():
    """
    Return ``True`` if running on Windows.

    :rtype: bool
    """
    return os.name == "nt"

def fix_slash(path):
    """
    On non-POSIX OSes, change the slashes in ``path``
    for loading in the browser.

    Example: ::

        c:\\abc\\def => c:/abc/def

    :param path: the path
    :type  path: Unicode string (path)
    :rtype: Unicode string
    """
    if not is_posix():
        # TODO is there a better way to do this?
        path = path.replace("\\", "/")
    return path

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
        if is_linux():
            # Linux
            return can_run_cdtw() and can_run_cmfcc() and can_run_cew()
        else:
            # no cew for other OSes
            return can_run_cdtw() and can_run_cmfcc()

def run_c_extension_with_fallback(
        log_function,
        extension,
        c_function,
        py_function,
        args,
        force_pure_python=False
):
    """
    Run a function calling a C extension, falling back
    to a pure Python function if the former does not succeed.

    :param log_function: a logger function (see ``_log()`` in many classes)
    :type  log_function: function
    :param extension: the name of the extension
    :type  extension: string
    :param c_function: the (Python) function calling the C extension
    :type  c_function: function
    :param py_function: the (Python) function providing the fallback
    :type  py_function: function
    :param force_pure_python: if ``True``, force running the pure Python code
    :type  force_pure_python: bool
    :rtype: depends on the extension being called

    :raise RuntimeError: if both the C extension and
                         the pure Python code did not succeed.

    .. versionadded:: 1.4.0
    """
    computed = False
    if force_pure_python:
        log_function(u"Force using pure Python code")
    else:
        if gc.USE_C_EXTENSIONS:
            log_function(u"C extensions enabled in gc")
            if can_run_c_extension(extension):
                log_function([u"C extensions enabled in gc and %s can be loaded", extension])
                computed, result = c_function(*args)
            else:
                log_function([u"C extensions enabled in gc but %s cannot be loaded", extension])
        else:
            log_function(u"C extensions disabled in gc")
    if not computed:
        log_function(u"Running the pure Python code")
        computed, result = py_function(*args)
    if not computed:
        raise RuntimeError("Both the C extension and the pure Python code did not succeed.")
    return result

def file_can_be_read(path):
    """
    Return ``True`` if the file at the given ``path`` can be read.

    :param path: the file path
    :type  path: string (path)
    :rtype: bool

    .. versionadded:: 1.4.0
    """
    if path is None:
        return False
    try:
        # TODO is doing this with os attributes preferable?
        test_file = io.open(path, "rb")
        test_file.close()
        return True
    except (IOError, OSError):
        pass
    return False

def file_can_be_written(path):
    """
    Return ``True`` if a file can be written at the given ``path``.

    IMPORTANT: this function will attempt to open the given ``path``
    in write mode, possibly destroying the file previously existing there.

    :param path: the file path
    :type  path: string (path)
    :rtype: bool

    .. versionadded:: 1.4.0
    """
    if path is None:
        return False
    try:
        # TODO think if it is better doing this with os attributes
        test_file = io.open(path, "wb")
        test_file.close()
        delete_file(None, path)
        return True
    except (IOError, OSError):
        pass
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

def file_size(path):
    """
    Return the size, in bytes, of the file at the given ``path``.
    Return ``-1`` if the file does not exist or cannot be read.

    :param path: the file path
    :type  path: string (path)
    :rtype: int
    """
    try:
        return os.path.getsize(path)
    except OSError:
        return -1

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

def relative_path(path, from_file):
    """
    Return the relative path of a file or directory, specified
    as ``path`` relative to (the parent directory of) ``from file``.

    The returned path is relative to the current working directory.

    Example: ::

        path="res/foo.bar"
        from_file="/root/abc/def/ghi.py"
        cwd="/root"
        => "abc/def/res/foo.bar"

    :param path: the file path
    :type  path: string (path)
    :param from_file: the reference file
    :type  from_file: string (path)
    :rtype: string (path)
    """
    if path is None:
        return None
    abs_path_target = absolute_path(path, from_file)
    abs_path_cwd = os.getcwd()
    return os.path.relpath(abs_path_target, start=abs_path_cwd)

def absolute_path(path, from_file):
    """
    Return the absolute path of a file or directory, specified
    as ``path`` relative to (the parent directory of) ``from_file``.

    This method is intented to be called with ``__file__``
    as second argument.

    Example: ::

        path="res/foo.bar"
        from_file="/abc/def/ghi.py"
        => "/abc/def/res/foo.bar"

    :param path: the file path
    :type  path: string (path)
    :param from_file: the reference file
    :type  from_file: string (path)
    :rtype: string (path)
    """
    if path is None:
        return None
    current_directory = os.path.dirname(from_file)
    target = os.path.join(current_directory, path)
    return os.path.abspath(target)

def read_file_bytes(input_file_path):
    """
    Read the file at the given file path
    and return its contents as a byte string,
    or ``None`` if an error occurred.

    :param input_file_path: the file path
    :type  input_file_path: string (path)
    :rtype: byte string
    """
    contents = None
    try:
        with io.open(input_file_path, "rb") as input_file:
            contents = input_file.read()
    except:
        pass
    return contents

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

def is_unicode(string):
    """
    Return True if the given string is a sequence of Unicode code points.

    :param string: the string to test
    :type  string: any
    :rtype: bool
    """
    if PY2:
        return isinstance(string, unicode)
    return isinstance(string, str)

def is_bytes(string):
    """
    Return True if the given string is a sequence of bytes.

    :param string: the string to test
    :type  string: any
    :rtype: bool
    """
    if PY2:
        return isinstance(string, str)
    return isinstance(string, bytes)

def is_utf8_encoded(bstring):
    """
    Return True if the given byte string can be decoded
    into a Unicode string using the UTF-8 decoder.

    :param bstring: the string to test
    :type  bstring: byte string
    :rtype: bool
    """
    try:
        bstring.decode("utf-8")
        return True
    except UnicodeDecodeError:
        pass
    return False

def safe_str(string):
    """
    Safely return the given Unicode string
    from a __str__ function: as a byte string
    in Python 2, or as a Unicode string in Python 3.

    :param string: the string to return
    :type  string: Unicode string
    :rtype: byte string or Unicode string
    """
    if string is None:
        return None
    if PY2:
        return string.encode("utf-8")
    return string

def safe_unichr(codepoint):
    """
    Safely return a Unicode string of length one,
    containing the Unicode character with given codepoint.

    :param codepoint: the codepoint
    :type  codepoint: int
    :rtype: Unicode string
    """
    if PY2:
        return unichr(codepoint)
    return chr(codepoint)

def safe_unicode(string):
    """
    Safely convert the given string to a Unicode string.

    :param string: the string to convert
    :type  string: byte string or Unicode string
    :rtype: Unicode string
    """
    if string is None:
        return None
    if is_bytes(string):
        return string.decode("utf-8")
    return string

def safe_bytes(string):
    """
    Safely convert the given string to a bytes string.

    :param string: the string to convert
    :type  string: byte string or Unicode string
    :rtype: byte string
    """
    if string is None:
        return None
    if is_unicode(string):
        return string.encode("utf-8")
    return string

def safe_unicode_stdin(string):
    """
    Safely convert the given string to a Unicode string,
    decoding using ``sys.stdin.encoding`` if needed.

    :param string: the string to convert
    :type  string: byte string or Unicode string
    :rtype: Unicode string
    """
    if string is None:
        return None
    if is_bytes(string):
        try:
            return string.decode(sys.stdin.encoding)
        except UnicodeDecodeError:
            return string.decode(sys.stdin.encoding, "replace")
    return string

def safe_print(string):
    """
    Safely print a given Unicode string to stdout,
    possibly replacing characters non-printable
    in the current stdout encoding.

    :param string: the message string
    :type  string: Unicode string
    """
    try:
        print(string)
    except UnicodeEncodeError:
        try:
            # NOTE encoding and decoding so that in Python 3 no b"..." is printed
            encoded = string.encode(sys.stdout.encoding, "replace")
            decoded = encoded.decode(sys.stdout.encoding, "replace")
            print(decoded)
        except (UnicodeDecodeError, UnicodeEncodeError):
            print(u"[ERRO] An unexpected error happened while printing to stdout.")
            print(u"[ERRO] Please check that your file/string encoding matches the shell encoding.")
            print(u"[ERRO] If possible, set your shell encoding to UTF-8 and convert any files with legacy encodings.")

def object_to_unicode(obj):
    """
    Return a sequence of Unicode code points from the given object.

    :param obj: the object
    :type  obj: object
    :rtype: unicode
    """
    if PY2:
        return unicode(obj)
    return str(obj)

def object_to_bytes(obj):
    """
    Return a sequence of bytes from the given object.

    :param obj: the object
    :type  obj: object
    :rtype: unicode
    """
    if PY2:
        return str(obj)
    return bytes(obj, encoding="utf-8")



