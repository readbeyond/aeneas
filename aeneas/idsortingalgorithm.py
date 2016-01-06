#!/usr/bin/env python
# coding=utf-8

"""
Enumeration of the available algorithms to sort IDs.
"""

from __future__ import absolute_import
from __future__ import print_function
import re

from aeneas.logger import Logger

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.4.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class IDSortingAlgorithm(object):
    """
    Enumeration of the available algorithms to sort
    a list of XML ``id`` attributes.

    :param algorithm: the id sorting algorithm to be used
    :type  algorithm: :class:`aeneas.idsortingalgorithm.IDSortingAlgorithm`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`

    :raise ValueError: if the value of ``algorithm`` is not allowed
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

    def __init__(self, algorithm, logger=None):
        if algorithm not in self.ALLOWED_VALUES:
            raise ValueError("Algorithm value not allowed")
        self.algorithm = algorithm
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def sort(self, ids):
        """
        Sort the given list of identifiers,
        returning a new (sorted) list.

        :param ids: the list of identifiers to be sorted
        :type ids: list of Unicode strings
        :rtype: list of Unicode strings
        """
        def extract_int(string):
            """
            Extract an integer from the given string.

            :param string: the identifier string
            :type  string: string
            :rtype: int
            """
            return int(re.sub(r"[^0-9]", "", string))

        tmp = list(ids)
        if self.algorithm == IDSortingAlgorithm.UNSORTED:
            self._log(u"Sorting using UNSORTED")
        elif self.algorithm == IDSortingAlgorithm.LEXICOGRAPHIC:
            self._log(u"Sorting using LEXICOGRAPHIC")
            tmp = sorted(ids)
        elif self.algorithm == IDSortingAlgorithm.NUMERIC:
            self._log(u"Sorting using NUMERIC")
            tmp = ids
            try:
                tmp = sorted(tmp, key=extract_int)
            except (ValueError, TypeError) as exc:
                self._log(u"Not all id values contain a numeric part:", Logger.WARNING)
                self._log([u"%s", exc], Logger.WARNING)
                self._log(u"Returning the id list unchanged (unsorted)", Logger.WARNING)
        return tmp



