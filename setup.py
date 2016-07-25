#!/usr/bin/env python
# coding=utf-8

"""
Set aeneas package up
"""

from setuptools import setup, Extension
import io
import os
import shutil
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
        espeak_dll_dst_path = "C:\\Windows\\System32\\espeak.dll"
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
                print("[WARN] If you want to compile the C extension cew,")
                print("[WARN] please copy espeak_sapi.dll from your eSpeak directory into %s" % espeak_dll_dst_path)
                print("[WARN] and run the aeneas setup again.")
                return False
            elif not copied:
                print("[WARN] Unable to copy the eSpeak DLL, probably because you are not running with admin privileges.")
                print("[WARN] If you want to compile the C extension cew,")
                print("[WARN] please copy espeak_sapi.dll from your eSpeak directory into %s" % espeak_dll_dst_path)
                print("[WARN] and run the aeneas setup again.")
                return False

        # copy thirdparty/espeak.lib to $PYTHON\libs\espeak.lib
        espeak_lib_src_path = os.path.join(os.path.dirname(__file__), "thirdparty", "espeak.lib")
        espeak_lib_dst_path = os.path.join(sys.prefix, "libs", "espeak.lib")
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

# try importing numpy: if it fails, warn user and exit
try:
    from numpy import get_include
    from numpy.distutils import misc_util
except ImportError:
    print("[ERRO] You must install numpy before installing aeneas")
    print("[INFO] Try the following command:")
    print("[INFO] $ sudo pip install numpy")
    sys.exit(1)

# package version, must be unique for each PyPI upload
PACKAGE_VERSION = "1.5.1.0"

# check whether the user set additional parameters using environment variables
# NOTE by the book this should be done by subclassing the setuptools Distribution object
#      but for now using environment variables is good enough
WITHOUT_CEW = os.getenv("AENEAS_WITH_CEW", "True") not in ["TRUE", "True", "true", "YES", "Yes", "yes", "1", 1]
FORCE_CEW = os.getenv("AENEAS_FORCE_CEW", "False") in ["TRUE", "True", "true", "YES", "Yes", "yes", "1", 1]

# get platform
IS_LINUX = (os.name == "posix") and (os.uname()[0] == "Linux")
IS_OSX = (os.name == "posix") and (os.uname()[0] == "Darwin")
IS_WINDOWS = (os.name == "nt")

# get human-readable descriptions
SHORT_DESCRIPTION = "aeneas is a Python/C library and a set of tools to automagically synchronize audio and text (aka forced alignment)"
try:
    LONG_DESCRIPTION = io.open("README.rst", "r", encoding="utf-8").read()
except:
    LONG_DESCRIPTION = SHORT_DESCRIPTION

# to compile cdtw and cmfcc, we need to include the NumPy dirs
INCLUDE_DIRS = [misc_util.get_numpy_include_dirs()]

# scripts to be installed globally
# on Linux and Mac OS X, use the file without extension
# on Windows, use the file with .py extension
SCRIPTS = [
    "bin/aeneas_check_setup",
    "bin/aeneas_convert_syncmap",
    "bin/aeneas_download",
    "bin/aeneas_execute_job",
    "bin/aeneas_execute_task",
    "bin/aeneas_plot_waveform",
    "bin/aeneas_synthesize_text",
    "bin/aeneas_validate",
]
if IS_WINDOWS:
    SCRIPTS = [s + ".py" for s in SCRIPTS]

# prepare Extension objects
EXTENSION_CDTW = Extension(
    "aeneas.cdtw.cdtw",
    ["aeneas/cdtw/cdtw_py.c", "aeneas/cdtw/cdtw_func.c", "aeneas/cint/cint.c"],
    include_dirs=[get_include()]
)
EXTENSION_CEW = Extension(
    "aeneas.cew.cew",
    ["aeneas/cew/cew_py.c", "aeneas/cew/cew_func.c"],
    libraries=["espeak"]
)
EXTENSION_CMFCC = Extension(
    "aeneas.cmfcc.cmfcc",
    ["aeneas/cmfcc/cmfcc_py.c", "aeneas/cmfcc/cmfcc_func.c", "aeneas/cwave/cwave_func.c", "aeneas/cint/cint.c"],
    include_dirs=[get_include()]
)
# cwave is ready, but currently not used
#EXTENSION_CWAVE = Extension(
#    "aeneas.cwave.cwave",
#    ["aeneas/cwave/cwave_py.c", "aeneas/cwave/cwave_func.c"],
#    include_dirs=[get_include()]
#)
#EXTENSIONS = [EXTENSION_CDTW, EXTENSION_CMFCC, EXTENSION_CWAVE]

# append or ignore cew extension as requested
EXTENSIONS = [EXTENSION_CDTW, EXTENSION_CMFCC]
if WITHOUT_CEW:
    print("[INFO] **********************************************************")
    print("[INFO] The user specified AENEAS_WITH_CEW=False: not building cew")
    print("[INFO] **********************************************************")
elif FORCE_CEW:
    print("[INFO] *******************************************************************************************")
    print("[INFO] The user specified AENEAS_FORCE_CEW=True: attempting to build cew without performing checks")
    print("[INFO] *******************************************************************************************")
    EXTENSIONS.append(EXTENSION_CEW)
else:
    if IS_LINUX:
        EXTENSIONS.append(EXTENSION_CEW)
    elif IS_OSX:
        print("[INFO] *************************************************************************************")
        print("[INFO] Compiling C extension cew on Mac OS X is experimental.")
        print("[INFO] ")
        print("[INFO] Before installing aeneas with cew, you must run:")
        print("[INFO] $ brew update && brew upgrade --cleanup espeak")
        print("[INFO] to run the new brew formula installing libespeak, the library version of espeak.")
        print("[INFO] ")
        print("[INFO] If you experience problems, disable cew compilation specifying AENEAS_WITH_CEW=False.")
        print("[INFO] Please see the aeneas installation documentation for details.")
        print("[INFO] *************************************************************************************")
        EXTENSIONS.append(EXTENSION_CEW)
    elif IS_WINDOWS:
        print("[INFO] *************************************************************************************")
        print("[INFO] Compiling C extension cew on Windows is experimental.")
        print("[INFO] ")
        print("[INFO] If you experience problems, disable cew compilation specifying AENEAS_WITH_CEW=False.")
        print("[INFO] Please see the aeneas installation documentation for details.")
        print("[INFO] *************************************************************************************")
        if prepare_cew_for_windows():
            EXTENSIONS.append(EXTENSION_CEW)
        else:
            print("[WARN] Unable to complete C extension cew setup, not compiling it.")
    else:
        print("[INFO] Extension cew is not available for your OS")

# finally set the aeneas module up
setup(
    name="aeneas",
    packages=[
        "aeneas",
        "aeneas.cdtw",
        "aeneas.cew",
        "aeneas.cmfcc",
        "aeneas.cwave",
        "aeneas.extra",
        "aeneas.tools"
    ],
    package_data={
        "aeneas": ["res/*", "*.md"],
        "aeneas.cdtw": ["*.c", "*.h", "*.md"],
        "aeneas.cew": ["*.c", "*.h", "*.md"],
        "aeneas.cmfcc": ["*.c", "*.h", "*.md"],
        "aeneas.cwave": ["*.c", "*.h", "*.md"],
        "aeneas.extra": ["*.md"],
        "aeneas.tools": ["res/*", "*.md"]
    },
    version=PACKAGE_VERSION,
    description=SHORT_DESCRIPTION,
    author="Alberto Pettarin",
    author_email="alberto@albertopettarin.it",
    url="https://github.com/readbeyond/aeneas",
    license="GNU Affero General Public License v3 (AGPL v3)",
    long_description=LONG_DESCRIPTION,
    install_requires=[
        "BeautifulSoup4==4.4.1",
        "lxml==3.6.0",
        "numpy>=1.9"
    ],
    extras_require={
        "full": ["pafy>=0.3.74", "Pillow>=3.1.1", "requests>=2.9.1", "youtube-dl>=2015.7.21"],
        "nopillow" : ["pafy>=0.3.74", "requests>=2.9.1", "youtube-dl>=2015.7.21"],
        "pafy": ["pafy>=0.3.74", "youtube-dl>=2015.7.21"],
        "pillow": ["Pillow>=3.1.1"],
        "requests": ["requests>=2.9.1"],
    },
    scripts=SCRIPTS,
    keywords=[
        "AUD",
        "CSV",
        "DTW",
        "EAF",
        "ELAN",
        "EPUB 3 Media Overlay",
        "EPUB 3",
        "EPUB",
        "JSON",
        "MFCC",
        "Mel-frequency cepstral coefficients",
        "ReadBeyond Sync",
        "ReadBeyond",
        "SBV",
        "SMIL",
        "SRT",
        "SSV",
        "SUB",
        "TSV",
        "TTML",
        "VTT",
        "XML",
        "aeneas",
        "audio/text alignment",
        "dynamic time warping",
        "espeak",
        "ffmpeg",
        "ffprobe",
        "forced alignment",
        "media overlay",
        "rb_smil_emulator",
        "subtitles",
        "sync",
        "synchronization",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Topic :: Education",
        "Topic :: Multimedia",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Printing",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Text Processing :: Markup",
        "Topic :: Text Processing :: Markup :: HTML",
        "Topic :: Text Processing :: Markup :: XML",
        "Topic :: Utilities"
    ],
    ext_modules=EXTENSIONS,
    include_dirs=INCLUDE_DIRS,
)
