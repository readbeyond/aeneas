#!/usr/bin/env python
# coding=utf-8

"""
Download an audio file from a YouTube video.
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from aeneas.tools.download import DownloadCLI

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
    Download an audio file from a YouTube video.
    """
    DownloadCLI(invoke="aeneas_download").run(arguments=sys.argv)

if __name__ == '__main__':
    main()



