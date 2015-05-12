#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='aeneas',
    packages=['aeneas', 'aeneas.tests', 'aeneas.tools'],
    version='1.0.0',
    description='aeneas is a Python library and a set of tools to automagically synchronize audio and text',
    author='Alberto Pettarin',
    author_email='alberto@albertopettarin.it',
    url='https://github.com/readbeyond/aeneas',
    license='GNU Affero General Public License v3 (AGPL v3)',
    long_description=open('README.txt').read(),
    keywords=['aeneas', 'ReadBeyond', 'audio/text alignment', 'forced alignment', 'sync', 'ReadBeyond Sync', 'synchronization', 'subtitles', 'smil', 'srt', 'ttml', 'vtt', 'ffmpeg', 'ffprobe', 'espeak', 'dtw', 'mfcc', 'dynamic time warping', 'Mel-frequency cepstral coefficients', 'epub 3 media overlay', 'epub', 'epub3', 'media overlay'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)
