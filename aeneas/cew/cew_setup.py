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
Compile the Python C Extension for synthesizing text with eSpeak.

.. versionadded:: 1.3.0
"""

from __future__ import absolute_import
from __future__ import print_function
from setuptools import Extension
from setuptools import setup
import sys


CMODULE = Extension(
    name="cew",
    sources=[
        "cew_py.c",
        "cew_func.c"
    ],
    libraries=[
        "espeak"
    ]
)

setup(
    name="cew",
    version="1.7.1",
    description="Python C Extension for synthesizing text with eSpeak.",
    ext_modules=[CMODULE]
)

print("\n[INFO] Module cew successfully compiled\n")
sys.exit(0)
