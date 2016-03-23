#!/usr/bin/env python
# coding=utf-8

"""
Set aeneas package up
"""

from setuptools import setup, Extension
import io
import os
import sys

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.5.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

try:
    from numpy import get_include
    from numpy.distutils import misc_util
except ImportError:
    print("[ERRO] You must install numpy before installing aeneas")
    print("[INFO] Try the following command:")
    print("[INFO] $ sudo pip install numpy")
    sys.exit(1)

IS_LINUX = (os.name == "posix") and (os.uname()[0] == "Linux")

SHORT_DESCRIPTION = "aeneas is a Python/C library and a set of tools to automagically synchronize audio and text (aka forced alignment)"

LONG_DESCRIPTION = io.open("README.rst", "r", encoding="utf-8").read()

INCLUDE_DIRS = [misc_util.get_numpy_include_dirs()]

EXTENSION_CDTW = Extension(
    "aeneas.cdtw.cdtw",
    ["aeneas/cdtw/cdtw_py.c", "aeneas/cdtw/cdtw_func.c", "aeneas/cdtw/cint.c"],
    include_dirs=[get_include()]
)
EXTENSION_CEW = Extension(
    "aeneas.cew.cew",
    ["aeneas/cew/cew_py.c", "aeneas/cew/cew_func.c"],
    libraries=["espeak"]
)
EXTENSION_CMFCC = Extension(
    "aeneas.cmfcc.cmfcc",
    ["aeneas/cmfcc/cmfcc_py.c", "aeneas/cmfcc/cmfcc_func.c", "aeneas/cmfcc/cwave_func.c", "aeneas/cmfcc/cint.c"],
    include_dirs=[get_include()]
)
# cwave is ready, but currently not used
#EXTENSION_CWAVE = Extension(
#    "aeneas.cwave.cwave",
#    ["aeneas/cwave/cwave_py.c", "aeneas/cwave/cwave_func.c"],
#    include_dirs=[get_include()]
#)
#EXTENSIONS = [EXTENSION_CDTW, EXTENSION_CMFCC, EXTENSION_CWAVE]

EXTENSIONS = [EXTENSION_CDTW, EXTENSION_CMFCC]
if IS_LINUX:
    # cew is available only for Linux at the moment
    EXTENSIONS.append(EXTENSION_CEW)

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
    version="1.5.0.0",
    description=SHORT_DESCRIPTION,
    author="Alberto Pettarin",
    author_email="alberto@albertopettarin.it",
    url="https://github.com/readbeyond/aeneas",
    license="GNU Affero General Public License v3 (AGPL v3)",
    long_description=LONG_DESCRIPTION,
    install_requires=[
        "BeautifulSoup4>=4.4",
        "lxml>=3.0",
        "numpy>=1.9"
    ],
    extras_require={
        "full": ["pafy>=0.3.74", "Pillow>=3.1.1", "requests>=2.9.1", "youtube-dl>=2015.7.21"],
        "nopillow" : ["pafy>=0.3.74", "requests>=2.9.1", "youtube-dl>=2015.7.21"],
        "pafy": ["pafy>=0.3.74", "youtube-dl>=2015.7.21"],
        "pillow": ["Pillow>=3.1.1"],
        "requests": ["requests>=2.9.1"],
    },
    scripts=[
        "bin/aeneas_check_setup",
        "bin/aeneas_convert_syncmap",
        "bin/aeneas_download",
        "bin/aeneas_execute_job",
        "bin/aeneas_execute_task",
        "bin/aeneas_plot_waveform",
        "bin/aeneas_synthesize_text",
        "bin/aeneas_validate",
    ],
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
        "SMIL",
        "SRT",
        "SSV",
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
