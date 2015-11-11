#!/usr/bin/env python
# coding=utf-8

"""
Compile the Python C Extension for synthesizing text with espeak.

.. versionadded:: 1.3.0
"""

import os
import sys

from distutils.core import Extension
from distutils.core import setup

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.3.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

for compiled in ["cew.so", "cew.dylib", "cew.pyd"]:
    if os.path.exists(compiled):
        try:
            os.remove(compiled)
            print "[INFO] Removed file %s\n" % compiled
        except:
            pass

CMODULE = Extension("cew", sources=["cew.c"], libraries=["espeak"])

setup(
    name="cew",
    version="1.3.2",
    description="""
    Python C Extension for synthesizing text with espeak.
    """,
    ext_modules=[CMODULE]
)

print "\n[INFO] Module cew successfully compiled\n"
sys.exit(0)


