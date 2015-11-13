#!/usr/bin/env python
# coding=utf-8

"""
Set aeneas package up
"""

from setuptools import setup, Extension

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

EXTENSIONS = []
INCLUDE_DIRS = []
#try:
from numpy import get_include
from numpy.distutils import misc_util
import os
INCLUDE_DIRS = [misc_util.get_numpy_include_dirs()]
EXTENSION_CDTW = Extension("aeneas.cdtw", ["aeneas/cdtw.c"], include_dirs=[get_include()])
EXTENSION_CEW = Extension("aeneas.cew", ["aeneas/cew.c"], libraries=["espeak"])
EXTENSION_CMFCC = Extension("aeneas.cmfcc", ["aeneas/cmfcc.c"], include_dirs=[get_include()])
EXTENSIONS = [EXTENSION_CDTW, EXTENSION_CMFCC]
if (os.name == "posix") and (os.uname()[0] == "Linux"):
    # cew is available only for Linux at the moment
    EXTENSIONS.append(EXTENSION_CEW)
#except:
#    pass

setup(
    name="aeneas",
    packages=["aeneas", "aeneas.tools"],
    package_data={"aeneas": ["res/*", "speak_lib.h"], "aeneas.tools": ["res/*"]},
    version="1.3.2.8",
    description="aeneas is a Python library and a set of tools to automagically synchronize audio and text",
    author="Alberto Pettarin",
    author_email="alberto@albertopettarin.it",
    url="https://github.com/readbeyond/aeneas",
    license="GNU Affero General Public License v3 (AGPL v3)",
    long_description=open("README.rst", "r").read(),
    setup_requires=["numpy>=1.9"],
    install_requires=["BeautifulSoup>=3.0", "lxml>=3.0", "numpy>=1.9"],
    extras_require={"pafy": ["pafy>=0.3"]},
    keywords=[
        "CSV",
        "DTW",
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
