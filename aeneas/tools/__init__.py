#!/usr/bin/env python
# coding=utf-8

"""
aeneas.tools is a collection of modules
that can be run as separate programs by the end user.
"""

import os
import sys

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl (www.readbeyond.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.0.1"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

def get_rel_path(path):
    current_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    target = os.path.join(current_dir, path)
    return os.path.relpath(target)



