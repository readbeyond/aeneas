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
__version__ = "1.1.1"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

# TODO find out how to compile/install cdtw and cmfcc

setup(
    name='aeneas',
    packages=['aeneas', 'aeneas.tests', 'aeneas.tools'],
    version='1.1.1',
    description='aeneas is a Python library and a set of tools to automagically synchronize audio and text',
    author='Alberto Pettarin',
    author_email='alberto@albertopettarin.it',
    url='https://github.com/readbeyond/aeneas',
    license='GNU Affero General Public License v3 (AGPL v3)',
    long_description=open('README.txt').read(),
    keywords=[
        'aeneas','ReadBeyond','audio/text alignment',
        'forced alignment','sync', 'ReadBeyond Sync',
        'synchronization', 'subtitles',
        'smil', 'srt', 'ttml', 'vtt',
        'ffmpeg', 'ffprobe', 'espeak',
        'dtw', 'mfcc', 'dynamic time warping',
        'Mel-frequency cepstral coefficients',
        'epub 3 media overlay', 'epub', 'epub3', 'media overlay'
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
        Extension('aeneas.cmfcc', ['aeneas/cmfcc.c'])
    ],
    include_dirs=misc_util.get_numpy_include_dirs()
)
