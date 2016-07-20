#!/usr/bin/env python
# coding=utf-8

"""
Convert a sync map from a format to another.
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from aeneas.tools.convert_syncmap import ConvertSyncMapCLI

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.5.1"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

def main():
    """
    Convert a sync map from a format to another.
    """
    ConvertSyncMapCLI(invoke="aeneas_convert_syncmap").run(arguments=sys.argv)

if __name__ == '__main__':
    main()



