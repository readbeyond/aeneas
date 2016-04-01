#!/usr/bin/env python
# coding=utf-8

"""
Compile the Python C Extension for computing the DTW.

.. versionadded:: 1.1.0
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from distutils.core import Extension
from distutils.core import setup
from numpy import get_include
from numpy.distutils import misc_util

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

CMODULE = Extension("cdtw", sources=["cdtw_py.c", "cdtw_func.c", "cint.c"], include_dirs=[get_include()])

setup(
    name="cdtw",
    version="1.5.0",
    description="""
    Python C Extension for computing the DTW as fast as your bare metal allows.
    """,
    ext_modules=[CMODULE],
    include_dirs=[misc_util.get_numpy_include_dirs()]
)

print("\n[INFO] Module cdtw successfully compiled\n")
sys.exit(0)


