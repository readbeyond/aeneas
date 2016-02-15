#!/usr/bin/env python
# coding=utf-8

"""
Compile the Python C Extension for synthesizing text with eSpeak.

.. versionadded:: 1.3.0
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from distutils.core import Extension
from distutils.core import setup

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.5.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

CMODULE = Extension("cew", sources=["cew_py.c", "cew_func.c"], libraries=["espeak"])

setup(
    name="cew",
    version="1.5.0",
    description="""
    Python C Extension for synthesizing text with eSpeak.
    """,
    ext_modules=[CMODULE]
)

print("\n[INFO] Module cew successfully compiled\n")
sys.exit(0)


