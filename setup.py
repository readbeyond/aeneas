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
Set the aeneas package up.
"""

from setuptools import Extension
from setuptools import setup
import io
import os
import shutil
import sys

from setupmeta import PKG_AUTHOR
from setupmeta import PKG_AUTHOR_EMAIL
from setupmeta import PKG_CLASSIFIERS
from setupmeta import PKG_EXTRAS_REQUIRE
from setupmeta import PKG_INSTALL_REQUIRES
from setupmeta import PKG_KEYWORDS
from setupmeta import PKG_LICENSE
from setupmeta import PKG_LONG_DESCRIPTION
from setupmeta import PKG_NAME
from setupmeta import PKG_PACKAGES
from setupmeta import PKG_PACKAGE_DATA
from setupmeta import PKG_SCRIPTS
from setupmeta import PKG_SHORT_DESCRIPTION
from setupmeta import PKG_URL
from setupmeta import PKG_VERSION

__author__ = "Alberto Pettarin"
__email__ = "aeneas@readbeyond.it"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
"""
__license__ = "GNU AGPL 3"
__status__ = "Production"
__version__ = "1.7.1"


##############################################################################
#
# Windows-specific setup for compiling cew
#
##############################################################################

def prepare_cew_for_windows():
    """
    Copy files needed to compile the ``cew`` Python C extension on Windows.

    A glorious day, when Microsoft will offer a decent support
    for Python and shared libraries,
    all this mess will be unnecessary and it should be removed.
    May that day come soon.

    Return ``True`` if successful, ``False`` otherwise.

    :rtype: bool
    """
    try:
        # copy espeak_sapi.dll to C:\Windows\System32\espeak.dll
        espeak_dll_win_path = "C:\\Windows\\System32\\espeak.dll"
        espeak_dll_dst_path = "aeneas\\cew\\espeak.dll"
        espeak_dll_src_paths = [
            "C:\\aeneas\\eSpeak\\espeak_sapi.dll",
            "C:\\sync\\eSpeak\\espeak_sapi.dll",
            "C:\\Program Files\\eSpeak\\espeak_sapi.dll",
            "C:\\Program Files (x86)\\eSpeak\\espeak_sapi.dll",
        ]
        if os.path.exists(espeak_dll_dst_path):
            print("[INFO] Found eSpeak DLL in %s" % espeak_dll_dst_path)
        else:
            found = False
            copied = False
            for src_path in espeak_dll_src_paths:
                if os.path.exists(src_path):
                    found = True
                    print("[INFO] Copying eSpeak DLL from %s into %s" % (src_path, espeak_dll_dst_path))
                    try:
                        shutil.copyfile(src_path, espeak_dll_dst_path)
                        copied = True
                        print("[INFO] Copied eSpeak DLL")
                    except:
                        pass
                    break
            if not found:
                print("[WARN] Unable to find the eSpeak DLL, probably because you installed eSpeak in a non-standard location.")
                print("[WARN] If you want to run aeneas with the C extension cew,")
                print("[WARN] please copy espeak_sapi.dll from your eSpeak directory to %s" % espeak_dll_win_path)
                # print("[WARN] and run the aeneas setup again.")
                # return False
            elif not copied:
                print("[WARN] Unable to copy the eSpeak DLL, probably because you are not running with admin privileges.")
                print("[WARN] If you want to run aeneas with the C extension cew,")
                print("[WARN] please copy espeak_sapi.dll from your eSpeak directory to %s" % espeak_dll_win_path)
                # print("[WARN] and run the aeneas setup again.")
                # return False

        # NOTE: espeak.lib is needed only while compiling the C extension, not when using it
        #       so, we copy it in the current working directory from the included thirdparty\ directory
        # NOTE: PREV: copy thirdparty\espeak.lib to $PYTHON\libs\espeak.lib
        # NOTE: PREV: espeak_lib_dst_path = os.path.join(sys.prefix, "libs", "espeak.lib")
        espeak_lib_src_path = os.path.join(os.path.dirname(__file__), "thirdparty", "espeak.lib")
        espeak_lib_dst_path = os.path.join(os.path.dirname(__file__), "espeak.lib")
        if os.path.exists(espeak_lib_dst_path):
            print("[INFO] Found eSpeak LIB in %s" % espeak_lib_dst_path)
        else:
            try:
                print("[INFO] Copying eSpeak LIB into %s" % espeak_lib_dst_path)
                shutil.copyfile(espeak_lib_src_path, espeak_lib_dst_path)
                print("[INFO] Copied eSpeak LIB")
            except:
                print("[WARN] Unable to copy the eSpeak LIB, probably because you are not running with admin privileges.")
                print("[WARN] If you want to compile the C extension cew,")
                print("[WARN] please copy espeak.lib from the thirdparty directory into %s" % espeak_lib_dst_path)
                print("[WARN] and run the aeneas setup again.")
                return False

        # if here, we have completed the setup, return True
        return True
    except Exception as e:
        print("[WARN] Unexpected exception while preparing cew: %s" % e)
    return False


##############################################################################
#
# find the OS out and read environment variables for options
#
##############################################################################

# get platform
IS_LINUX = (os.name == "posix") and (os.uname()[0] == "Linux")
IS_OSX = (os.name == "posix") and (os.uname()[0] == "Darwin")
IS_WINDOWS = (os.name == "nt")

# define what values of environment variables are considered equal to True
TRUE_VALUES = [
    "TRUE",
    "True",
    "true",
    "YES",
    "Yes",
    "yes",
    "1",
    1
]

# check whether the user set additional parameters using environment variables
# NOTE by the book this should be done by subclassing the setuptools Distribution object
#      but for now using environment variables is good enough
WITHOUT_CDTW = os.getenv("AENEAS_WITH_CDTW", "True") not in TRUE_VALUES
WITHOUT_CMFCC = os.getenv("AENEAS_WITH_CMFCC", "True") not in TRUE_VALUES
WITHOUT_CEW = os.getenv("AENEAS_WITH_CEW", "True") not in TRUE_VALUES
FORCE_CEW = os.getenv("AENEAS_FORCE_CEW", "False") in TRUE_VALUES
FORCE_CFW = os.getenv("AENEAS_FORCE_CFW", "False") in TRUE_VALUES


##############################################################################
#
# actual setup
#
##############################################################################

# try importing numpy: if it fails, warn user and exit
try:
    from numpy import get_include
    from numpy.distutils import misc_util
except ImportError:
    print("[ERRO] You must install numpy before installing aeneas")
    print("[INFO] Try the following command:")
    print("[INFO] $ sudo pip install numpy")
    sys.exit(1)

# to compile cdtw and cmfcc, we need to include the NumPy dirs
INCLUDE_DIRS = [misc_util.get_numpy_include_dirs()]

# scripts to be installed globally
# on Linux and Mac OS X, use the file without extension
# on Windows, use the file with .py extension
if IS_WINDOWS:
    PKG_SCRIPTS = [s + ".py" for s in PKG_SCRIPTS]

# prepare Extension objects
EXTENSION_CDTW = Extension(
    name="aeneas.cdtw.cdtw",
    sources=[
        "aeneas/cdtw/cdtw_py.c",
        "aeneas/cdtw/cdtw_func.c",
        "aeneas/cint/cint.c"
    ],
    include_dirs=[
        get_include()
    ]
)
EXTENSION_CMFCC = Extension(
    name="aeneas.cmfcc.cmfcc",
    sources=[
        "aeneas/cmfcc/cmfcc_py.c",
        "aeneas/cmfcc/cmfcc_func.c",
        "aeneas/cwave/cwave_func.c",
        "aeneas/cint/cint.c"
    ],
    include_dirs=[
        get_include()
    ]
)
EXTENSION_CEW = Extension(
    name="aeneas.cew.cew",
    sources=[
        "aeneas/cew/cew_py.c",
        "aeneas/cew/cew_func.c"
    ],
    libraries=[
        "espeak"
    ]
)
EXTENSION_CFW = Extension(
    name="aeneas.cfw.cfw",
    sources=[
        "aeneas/cfw/cfw_py.cc",
        "aeneas/cfw/cfw_func.cc"
    ],
    include_dirs=[
        "aeneas/cfw/festival",
        "aeneas/cfw/speech_tools"
    ],
    libraries=[
        "Festival",
        "estools",
        "estbase",
        "eststring",
    ]
)
# cwave is ready, but currently not used
# EXTENSION_CWAVE = Extension(
#     name="aeneas.cwave.cwave",
#     sources=[
#         "aeneas/cwave/cwave_py.c",
#         "aeneas/cwave/cwave_func.c"
#     ],
#     include_dirs=[
#         get_include()
#     ]
# )

# append or ignore cew extension as requested
EXTENSIONS = []

if WITHOUT_CDTW:
    print("[INFO] ************************************************************")
    print("[INFO] The user specified AENEAS_WITH_CDTW=False: not building cdtw")
    print("[INFO] ************************************************************")
    print("[INFO] ")
else:
    EXTENSIONS.append(EXTENSION_CDTW)

if WITHOUT_CMFCC:
    print("[INFO] **************************************************************")
    print("[INFO] The user specified AENEAS_WITH_CMFCC=False: not building cmfcc")
    print("[INFO] **************************************************************")
    print("[INFO] ")
else:
    EXTENSIONS.append(EXTENSION_CMFCC)

if WITHOUT_CEW:
    print("[INFO] **********************************************************")
    print("[INFO] The user specified AENEAS_WITH_CEW=False: not building cew")
    print("[INFO] **********************************************************")
    print("[INFO] ")
elif FORCE_CEW:
    print("[INFO] ********************************************************************************")
    print("[INFO] The user specified AENEAS_FORCE_CEW=True: attempting to build cew without checks")
    print("[INFO] ********************************************************************************")
    print("[INFO] ")
    EXTENSIONS.append(EXTENSION_CEW)
else:
    if IS_LINUX:
        EXTENSIONS.append(EXTENSION_CEW)
    elif IS_OSX:
        print("[INFO] *********************************************************************************")
        print("[INFO] Compiling the C extension cew on Mac OS X is experimental.")
        print("[INFO] ")
        print("[INFO] Before installing aeneas with cew, you must run:")
        print("[INFO] $ brew update && brew upgrade --cleanup espeak")
        print("[INFO] to run the new brew formula installing libespeak, the library version of espeak.")
        print("[INFO] ")
        print("[INFO] If you experience problems, disable cew compilation by specifying")
        print("[INFO] the environment variable AENEAS_WITH_CEW=False .")
        print("[INFO] Please see the aeneas installation documentation for details.")
        print("[INFO] ********************************************************************************")
        print("[INFO] ")
        EXTENSIONS.append(EXTENSION_CEW)
    elif IS_WINDOWS:
        print("[INFO] *****************************************************************")
        print("[INFO] Compiling the C extension cew on Windows is experimental.")
        print("[INFO] ")
        print("[INFO] If you experience problems, disable cew compilation by specifying")
        print("[INFO] the environment variable AENEAS_WITH_CEW=False .")
        print("[INFO] Please see the aeneas installation documentation for details.")
        print("[INFO] *****************************************************************")
        print("[INFO] ")
        if prepare_cew_for_windows():
            EXTENSIONS.append(EXTENSION_CEW)
        else:
            print("[WARN] Unable to complete the setup for C extension cew, not building it.")
    else:
        print("[INFO] The C extension cew is not available for your OS.")

if FORCE_CFW:
    print("[INFO] ********************************************************************************")
    print("[INFO] The user specified AENEAS_FORCE_CFW=True: attempting to build cfw without checks")
    print("[INFO] ********************************************************************************")
    print("[INFO] ")
    EXTENSIONS.append(EXTENSION_CFW)

# now we are ready to call setup()
setup(
    name=PKG_NAME,
    version=PKG_VERSION,
    packages=PKG_PACKAGES,
    package_data=PKG_PACKAGE_DATA,
    description=PKG_SHORT_DESCRIPTION,
    long_description=PKG_LONG_DESCRIPTION,
    author=PKG_AUTHOR,
    author_email=PKG_AUTHOR_EMAIL,
    url=PKG_URL,
    license=PKG_LICENSE,
    keywords=PKG_KEYWORDS,
    classifiers=PKG_CLASSIFIERS,
    install_requires=PKG_INSTALL_REQUIRES,
    extras_require=PKG_EXTRAS_REQUIRE,
    scripts=PKG_SCRIPTS,
    include_dirs=INCLUDE_DIRS,
    ext_modules=EXTENSIONS
)
