#!/usr/bin/env python
# coding=utf-8

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2016, Alberto Pettarin (www.albertopettarin.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
__email__ = "aeneas@readbeyond.it"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
"""
__license__ = "GNU AGPL 3"
__status__ = "Production"
__version__ = "1.7.1"


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
