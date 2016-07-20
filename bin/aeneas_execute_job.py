#!/usr/bin/env python
# coding=utf-8

"""
Execute a Job, passed as a container or
as a container and a configuration string
(i.e., from a wizard).
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from aeneas.tools.execute_job import ExecuteJobCLI

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
    Execute a Job, passed as a container or
    as a container and a configuration string
    (i.e., from a wizard).
    """
    ExecuteJobCLI(invoke="aeneas_execute_job").run(arguments=sys.argv)

if __name__ == '__main__':
    main()



