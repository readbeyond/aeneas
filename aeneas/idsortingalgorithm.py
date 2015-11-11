#!/usr/bin/env python
# coding=utf-8

"""
Enumeration of the available algorithms to sort IDs.
"""

import re

from aeneas.logger import Logger

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.3.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class IDSortingAlgorithm(object):
    """
    Enumeration of the available algorithms to sort
    a list of XML ``id`` attributes.

    :param algorithm: the id sorting algorithm to be used
    :type  algorithm: string (from :class:`aeneas.idsortingalgorithm.IDSortingAlgorithm` enumeration)
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    LEXICOGRAPHIC = "lexicographic"
    """ Lexicographic sorting
    (e.g., ``f020`` before ``f10`` before ``f2``) """

    NUMERIC = "numeric"
    """ Numeric sorting, ignoring any non-digit characters or leading zeroes
    (e.g., ``f2`` (= ``2``) before ``f10`` (= ``10``)
    before ``f020`` (= ``20``)) """

    UNSORTED = "unsorted"
    """ Do not sort and return the list of identifiers, in their original order
    (e.g., ``f2`` before ``f020`` before ``f10``, assuming this was
    their order in the XML DOM) """

    ALLOWED_VALUES = [LEXICOGRAPHIC, NUMERIC, UNSORTED]
    """ List of all the allowed values """

    TAG = "IDSortingAlgorithm"

    def __init__(self, algorithm, logger=None):
        self.algorithm = algorithm
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def sort(self, ids):
        """
        Sort the given list of identifiers.

        :param ids: a list of identifiers to be sorted
        :type ids: list of strings
        :rtype: list of strings
        """
        tmp = ids
        if self.algorithm == IDSortingAlgorithm.UNSORTED:
            # nothing to do
            self._log("Using UNSORTED")
        if self.algorithm == IDSortingAlgorithm.LEXICOGRAPHIC:
            # sort lexicographically
            self._log("Using LEXICOGRAPHIC")
            tmp = sorted(ids)
        if self.algorithm == IDSortingAlgorithm.NUMERIC:
            # sort numerically
            self._log("Using NUMERIC")
            tmp = ids
            try:
                tmp = [[int(re.sub(r"[^0-9]", "", i)), i] for i in ids]
                tmp = sorted(tmp)
                tmp = [t[1] for t in tmp]
            except (ValueError, TypeError):
                self._log("Failed", Logger.WARNING)
        return tmp



