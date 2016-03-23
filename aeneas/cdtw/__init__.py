#!/usr/bin/env python
# coding=utf-8

"""
aeneas.cdtw is a Python C extension for computing the DTW.

.. function:: cdtw.compute_best_path(mfcc1, mfcc2, delta)

    Compute the DTW (approximated) best path
    for the two audio waves, represented by their MFCCs.

    This function implements the Sakoe-Chiba heuristic,
    that is, it explores only a band of width ``2 * delta``
    around the main diagonal of the cost matrix.

    The computation is done in-memory, and it might fail
    if there is not enough memory to allocate the cost matrix
    or the list to be returned.

    The returned list contains tuples ``(i, j)``,
    representing the best path from ``(0, 0)`` to ``(n-1, m-1)``,
    where ``n`` is the length of ``mfcc1``, and
    ``m`` is the length of ``mfcc2``.
    The returned list has length between ``min(n, m)`` and ``n + m``
    (it can be less than ``n + m`` if diagonal steps
    are selected in the best path).

    :param mfcc1: the MFCCs of the first wave ``(n, mfcc_size)``
    :type  mfcc1: :class:`numpy.ndarray`
    :param mfcc2: the MFCCs of the second wave ``(m, mfcc_size)``
    :type  mfcc2: :class:`numpy.ndarray`
    :param int delta: the margin parameter
    :rtype: list of tuples

.. function:: cdtw.compute_cost_matrix_step(mfcc1, mfcc2, delta)

    Compute the DTW (approximated) cost matrix
    for the two audio waves, represented by their MFCCs.

    This function implements the Sakoe-Chiba heuristic,
    that is, it explores only a band of width ``2 * delta``
    around the main diagonal of the cost matrix.

    The computation is done in-memory, and it might fail
    if there is not enough memory to allocate the cost matrix.

    The returned tuple ``(cost_matrix, centers)``
    contains the cost matrix (NumPy 2D array of shape (n, delta))
    and the row centers (NumPy 1D array of size n).

    :param mfcc1: the MFCCs of the first wave ``(n, mfcc_size)``
    :type  mfcc1: :class:`numpy.ndarray`
    :param mfcc2: the MFCCs of the second wave ``(m, mfcc_size)``
    :type  mfcc2: :class:`numpy.ndarray`
    :param int delta: the margin parameter
    :rtype: tuple

.. function:: cdtw.compute_accumulated_cost_matrix_step(cost_matrix, centers)

    Compute the DTW (approximated) accumulated cost matrix
    from the cost matrix and the row centers.

    This function implements the Sakoe-Chiba heuristic,
    that is, it explores only a band of width ``2 * delta``
    around the main diagonal of the cost matrix.

    The computation is done in-memory,
    and the accumulated cost matrix is computed in place,
    that is, the original cost matrix is destroyed
    and its allocated memory used to store
    the accumulated cost matrix.
    Hence, this call should not fail for memory reasons.

    The returned NumPy 2D array of shape ``(n, delta)``
    contains the accumulated cost matrix.

    :param cost_matrix: the cost matrix ``(n, delta)``
    :type  cost_matrix: :class:`numpy.ndarray`
    :param centers: the row centers ``(n,)``
    :type  centers: :class:`numpy.ndarray`
    :rtype: :class:`numpy.ndarray`

.. function:: cdtw.compute_best_path_step(accumulated_cost_matrix, centers)

    Compute the DTW (approximated) best path
    from the accumulated cost matrix and the row centers.

    This function implements the Sakoe-Chiba heuristic,
    that is, it explores only a band of width ``2 * delta``
    around the main diagonal of the cost matrix.

    The computation is done in-memory, and it might fail
    if there is not enough memory to allocate the list to be returned.

    The returned list contains tuples ``(i, j)``,
    representing the best path from ``(0, 0)`` to ``(n-1, m-1)``,
    where ``n`` is the length of ``mfcc1``, and
    ``m`` is the length of ``mfcc2``.
    The returned list has length between ``min(n, m)`` and ``n + m``
    (it can be less than ``n + m`` if diagonal steps
    are selected in the best path).

    :param cost_matrix: the accumulated cost matrix ``(n, delta)``
    :type  cost_matrix: :class:`numpy.ndarray`
    :param centers: the row centers ``(n, )``
    :type  centers: :class:`numpy.ndarray`
    :rtype: list of tuples
"""

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.5.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"



