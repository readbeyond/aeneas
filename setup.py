#!/usr/bin/env python
# coding=utf-8

"""
Set aeneas package up
"""
from numpy.distutils import misc_util
from setuptools import setup, Extension

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.1.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

setup(
    name="aeneas",
    packages=["aeneas", "aeneas.tests", "aeneas.tools"],
    version="1.1.2",
    description="aeneas is a Python library and a set of tools to automagically synchronize audio and text",
    author="Alberto Pettarin",
    author_email="alberto@albertopettarin.it",
    url="https://github.com/readbeyond/aeneas",
    license="GNU Affero General Public License v3 (AGPL v3)",
    long_description=open("README.txt").read(),
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
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    ext_modules=[
        Extension("aeneas.cdtw", ["aeneas/cdtw.c"]),
        Extension("aeneas.cmfcc", ["aeneas/cmfcc.c"])
    ],
    include_dirs=misc_util.get_numpy_include_dirs()
)
