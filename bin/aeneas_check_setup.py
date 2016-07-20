#!/usr/bin/env python
# coding=utf-8

"""
Check whether the setup of aeneas was successful.

Running this script makes sense only
if you git-cloned the original GitHub repository
and/or if you are interested in contributing to the
development of aeneas.
"""

from __future__ import absolute_import
from __future__ import print_function
import os
import sys

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.5.1"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

ANSI_ERROR = u"\033[91m"
ANSI_OK = u"\033[92m"
ANSI_WARNING = u"\033[93m"
ANSI_END = u"\033[0m"

def is_posix():
    return os.name == "posix"

def print_error(msg):
    if is_posix():
        print(u"%s[ERRO] %s%s" % (ANSI_ERROR, msg, ANSI_END))
    else:
        print(u"[ERRO] %s" % (msg))

def print_info(msg):
    print(u"[INFO] %s" % (msg))

def print_success(msg):
    if is_posix():
        print(u"%s[INFO] %s%s" % (ANSI_OK, msg, ANSI_END))
    else:
        print(u"[INFO] %s" % (msg))

def print_warning(msg):
    if is_posix():
        print(u"%s[WARN] %s%s" % (ANSI_WARNING, msg, ANSI_END))
    else:
        print(u"[WARN] %s" % (msg))

def check_import():
    try:
        import aeneas
        print_success(u"aeneas         OK")
        return False 
    except ImportError:
        print_error(u"aeneas         ERROR")
        print_info(u"  Unable to load the aeneas Python package")
        print_info(u"  This error is probably caused by:")
        print_info(u"    A. you did not download/git-clone the aeneas package properly; or")
        print_info(u"    B. you did not install the required Python packages:")
        print_info(u"      1. BeautifulSoup4")
        print_info(u"      2. lxml")
        print_info(u"      3. numpy")
    except Exception as e:
        print_error(e)
    return True

def main():
    # first, check we can import aeneas module
    if check_import():
        sys.exit(1)

    # then, run the built-in diagnostics
    from aeneas.diagnostics import Diagnostics
    errors, warnings, c_ext_warnings = Diagnostics.check_all()
    if errors:
        sys.exit(1)
    if c_ext_warnings:
        print_warning(u"All required dependencies are met but at least one available Python C extension is not compiled")
        print_warning(u"You can still run aeneas but it will be slower")
        print_warning(u"Enjoy running aeneas!")
        sys.exit(2)
    else:
        print_success(u"All required dependencies are met and all available Python C extensions are compiled")
        print_success(u"Enjoy running aeneas!")
        sys.exit(0)



if __name__ == '__main__':
    main()



