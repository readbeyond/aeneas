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
Metadata for setting the aeneas package up
"""

import io

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
# you might need to edit the information below this line
#
##############################################################################

# package version
# NOTE: generate a new one for each PyPI upload, otherwise it will fail
PKG_VERSION = "1.7.1.0"

# required packages to install
# NOTE: always use exact version numbers
# NOTE: this list should be the same as requirements.txt
PKG_INSTALL_REQUIRES = [
    "BeautifulSoup4>=4.5.1",
    "lxml>=3.6.0",
    "numpy>=1.9"
]

# required packages to install extra tools
PKG_EXTRAS_REQUIRE = {
    "full": [
        "boto3>=1.4.2",
        "pafy>=0.5.2",
        "Pillow>=3.1.1",
        "requests>=2.9.1",
        "tgt>=1.4.2",
        "youtube-dl>=2016.9.27",
    ],
    "nopillow": [
        "boto3>=1.4.2",
        "pafy>=0.5.2",
        "requests>=2.9.1",
        "tgt>=1.4.2",
        "youtube-dl>=2016.9.27",
    ],
    "boto3": [
        "boto3>=1.4.2",
    ],
    "pafy": [
        "pafy>=0.5.2",
        "youtube-dl>=2016.9.27",
    ],
    "pillow": [
        "Pillow>=3.1.1",
    ],
    "requests": [
        "requests>=2.9.1",
    ],
    "tgt": [
        "tgt>=1.4.2",
    ]
}

# packages to be distributed
# NOTE: not including the aeneas.test package to keep the size small
PKG_PACKAGES = [
    "aeneas",
    "aeneas.cdtw",
    "aeneas.cew",
    "aeneas.cfw",
    "aeneas.cmfcc",
    "aeneas.cwave",
    "aeneas.extra",
    "aeneas.syncmap",
    "aeneas.tools",
    "aeneas.ttswrappers"
]

# data files to be distributed
# NOTE: .py files will be added automatically
PKG_PACKAGE_DATA = {
    "aeneas": [
        "res/*",
        "*.md"
    ],
    "aeneas.cdtw": [
        "*.c",
        "*.h",
        "*.md"
    ],
    "aeneas.cew": [
        "*.c",
        "*.h",
        "*.md",
        "*.dll"
    ],
    "aeneas.cew": [
        "*.cc",
        "*.h",
        "*.md"
    ],
    "aeneas.cmfcc": [
        "*.c",
        "*.h",
        "*.md"
    ],
    "aeneas.cwave": [
        "*.c",
        "*.h",
        "*.md"
    ],
    "aeneas.extra": [
        "*.md"
    ],
    "aeneas.syncmap": [
        "*.md"
    ],
    "aeneas.tools": [
        "res/*",
        "*.md"
    ],
    "aeneas.ttswrappers": [
        "*.md"
    ]
}

# scripts to be installed globally
# on Linux and Mac OS X, use the file without extension
# on Windows, use the file with .py extension
PKG_SCRIPTS = [
    "bin/aeneas_check_setup",
    "bin/aeneas_convert_syncmap",
    "bin/aeneas_download",
    "bin/aeneas_execute_job",
    "bin/aeneas_execute_task",
    "bin/aeneas_plot_waveform",
    "bin/aeneas_synthesize_text",
    "bin/aeneas_validate",
]

##############################################################################
#
# do not edit the metadata below this line
#
##############################################################################

# package name
PKG_NAME = "aeneas"

# package author
PKG_AUTHOR = "Alberto Pettarin"

# package author email
PKG_AUTHOR_EMAIL = "alberto@albertopettarin.it"

# package URL
PKG_URL = "https://github.com/readbeyond/aeneas"

# package license
PKG_LICENSE = "GNU Affero General Public License v3 (AGPL v3)"

# human-readable descriptions
PKG_SHORT_DESCRIPTION = "aeneas is a Python/C library and a set of tools to automagically synchronize audio and text (aka forced alignment)"
try:
    PKG_LONG_DESCRIPTION = io.open("README.rst", "r", encoding="utf-8").read()
except:
    PKG_LONG_DESCRIPTION = PKG_SHORT_DESCRIPTION

# PyPI keywords
PKG_KEYWORDS = [
    "AUD",
    "AWS Polly TTS API",
    "CSV",
    "DTW",
    "EAF",
    "ELAN",
    "EPUB 3 Media Overlay",
    "EPUB 3",
    "EPUB",
    "Festival",
    "JSON",
    "MFCC",
    "Mel-frequency cepstral coefficients",
    "Nuance TTS API",
    "ReadBeyond Sync",
    "ReadBeyond",
    "SBV",
    "SMIL",
    "SRT",
    "SSV",
    "SUB",
    "TGT",
    "TSV",
    "TTML",
    "TTS",
    "TextGrid",
    "VTT",
    "XML",
    "aeneas",
    "audio/text alignment",
    "dynamic time warping",
    "eSpeak",
    "eSpeak-ng",
    "espeak",
    "espeak-ng",
    "festival",
    "ffmpeg",
    "ffprobe",
    "forced alignment",
    "media overlay",
    "rb_smil_emulator",
    "speech to text",
    "subtitles",
    "sync",
    "synchronization",
    "text to speech",
    "text2wave",
    "tts",
]

# PyPI classifiers
PKG_CLASSIFIERS = [
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
]
