#!/usr/bin/env python
# coding=utf-8

"""
aeneas.tests is a collection of (fast) unit tests
to be run to test the main aeneas package.
"""

import os

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.2.1"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

def get_abs_path(rel_path):
    """
    Get absolute path from the path relative to this directory.

    :param rel_path: the relative file path
    :type  rel_path: string (path)
    """
    if rel_path is None:
        return None
    # TODO find a better way
    file_dir = os.path.dirname(__file__)
    return os.path.join(file_dir, rel_path)



