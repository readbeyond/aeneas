#!/usr/bin/env python
# coding=utf-8

"""
Perform validation in one of the following modes:

1. a container
2. a job configuration string
3. a task configuration string
4. a container + configuration string from wizard
5. a job TXT configuration file
6. a job XML configuration file
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from aeneas.tools.validate import ValidateCLI

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
    Perform validation in one of the following modes:

    1. a container
    2. a job configuration string
    3. a task configuration string
    4. a container + configuration string from wizard
    5. a job TXT configuration file
    6. a job XML configuration file
    """
    ValidateCLI(invoke="aeneas_validate").run(arguments=sys.argv)

if __name__ == '__main__':
    main()



