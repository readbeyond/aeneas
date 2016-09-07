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
