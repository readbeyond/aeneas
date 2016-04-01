#!/usr/bin/env python
# coding=utf-8

"""
This module contains the following classes:

* :class:`~aeneas.hierarchytype.HierarchyType`,
  enumerating the allowed hierarchy types of a :class:`~aeneas.container.Container`.
"""

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

class HierarchyType:
    """
    Enumeration of the allowed hierarchy types of a
    :class:`~aeneas.container.Container`.
    """

    FLAT = "flat"
    """
    Flat hierarchy, that is, all the assets
    are located inside the same root directory
    (possibly, with subdirectories).
    """

    PAGED = "paged"
    """
    Paged hierarchy, that is, assets are divided
    into multiple sibling directories, one corresponding to each page/task.
    A page directory might have subdirectories where the audio/text assets
    are located.
    """

    ALLOWED_VALUES = [FLAT, PAGED]
    """ List of all the allowed values """



