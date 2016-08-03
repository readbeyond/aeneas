#!/usr/bin/env python
# coding=utf-8

"""
This is the aeneas-cli "hydra" script,
to be compiled by pyinstaller.
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from aeneas.globalfunctions import FROZEN
from aeneas.tools.hydra import HydraCLI

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
    This is the aeneas-cli "hydra" script,
    to be compiled by pyinstaller.
    """
    if FROZEN:
        HydraCLI(invoke="aeneas-cli").run(arguments=sys.argv, show_help=False)
    else:
        HydraCLI(invoke="pyinstaller-aeneas-cli.py").run(arguments=sys.argv, show_help=False)

if __name__ == '__main__':
    main()



