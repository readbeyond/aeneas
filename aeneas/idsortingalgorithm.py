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
This module contains the following classes:

* :class:`~aeneas.idsortingalgorithm.IDSortingAlgorithm`,
  enumerating and implementing the available algorithms to sort
  a list of XML ``id`` attributes.

.. warning:: This module is likely to be refactored in a future version
"""

from __future__ import absolute_import
from __future__ import print_function
import re

from aeneas.logger import Loggable


class IDSortingAlgorithm(Loggable):
    """
    Enumeration of the available algorithms to sort
    a list of XML ``id`` attributes.

    :param algorithm: the id sorting algorithm to be used
    :type  algorithm: :class:`~aeneas.idsortingalgorithm.IDSortingAlgorithm`
    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    :raises: ValueError: if the value of ``algorithm`` is not allowed
    """

    LEXICOGRAPHIC = "lexicographic"
    """ Lexicographic sorting
    (e.g., ``f020`` before ``f10`` before ``f2``) """

    NUMERIC = "numeric"
    """ Numeric sorting, ignoring any non-digit characters or leading zeroes
    (e.g., ``f2`` (= ``2``) before ``f10`` (= ``10``) before ``f020`` (= ``20``)) """

    UNSORTED = "unsorted"
    """ Do not sort and return the list of identifiers, in their original order
    (e.g., ``f2`` before ``f020`` before ``f10``,
    assuming this was their order in the XML DOM) """

    ALLOWED_VALUES = [LEXICOGRAPHIC, NUMERIC, UNSORTED]
    """ List of all the allowed values """

    TAG = u"IDSortingAlgorithm"

    def __init__(self, algorithm, rconf=None, logger=None):
        if algorithm not in self.ALLOWED_VALUES:
            raise ValueError(u"Algorithm value not allowed")
        super(IDSortingAlgorithm, self).__init__(rconf=rconf, logger=logger)
        self.algorithm = algorithm

    def sort(self, ids):
        """
        Sort the given list of identifiers,
        returning a new (sorted) list.

        :param list ids: the list of identifiers to be sorted
        :rtype: list
        """
        def extract_int(string):
            """
            Extract an integer from the given string.

            :param string string: the identifier string
            :rtype: int
            """
            return int(re.sub(r"[^0-9]", "", string))

        tmp = list(ids)
        if self.algorithm == IDSortingAlgorithm.UNSORTED:
            self.log(u"Sorting using UNSORTED")
        elif self.algorithm == IDSortingAlgorithm.LEXICOGRAPHIC:
            self.log(u"Sorting using LEXICOGRAPHIC")
            tmp = sorted(ids)
        elif self.algorithm == IDSortingAlgorithm.NUMERIC:
            self.log(u"Sorting using NUMERIC")
            tmp = ids
            try:
                tmp = sorted(tmp, key=extract_int)
            except (ValueError, TypeError) as exc:
                self.log_exc(u"Not all id values contain a numeric part. Returning the id list unchanged.", exc, False, None)
        return tmp
