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
This module contains the implementation
of dynamic time warping (DTW) algorithms
to align two audio waves, represented by their
Mel-frequency cepstral coefficients (MFCCs).

This module contains the following classes:

* :class:`~aeneas.dtw.DTWAlgorithm`,
  an enumeration of the available algorithms;
* :class:`~aeneas.dtw.DTWAligner`,
  the actual wave aligner;
* :class:`~aeneas.dtw.DTWExact`,
  a DTW aligner implementing the exact (full) DTW algorithm;
* :class:`~aeneas.dtw.DTWStripe`,
  a DTW aligner implementing the Sachoe-Chiba band heuristic.

To align two wave files:

1. build an :class:`~aeneas.dtw.DTWAligner` object,
   passing in the constructor
   the paths of the two wave files
   or their MFCC representations;
2. call :func:`~aeneas.dtw.DTWAligner.compute_path`
   to compute the min cost path between
   the MFCC representations of the two wave files.

.. warning:: This module might be refactored in a future version
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy

from aeneas.audiofilemfcc import AudioFileMFCC
from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration
import aeneas.globalfunctions as gf


class DTWAlgorithm(object):
    """
    Enumeration of the DTW algorithms that can be used
    for the alignment of two audio waves.
    """

    EXACT = "exact"
    """ Classical (exact) DTW algorithm.

    This implementation has ``O(nm)`` time and space complexity,
    where ``n`` (respectively, ``m``) is the number of MFCC window shifts (vectors)
    of the real (respectively, synthesized) wave. """

    STRIPE = "stripe"
    """ DTW algorithm restricted to a stripe around the main diagonal
    (Sakoe-Chiba Band), for reducing memory usage and run time.

    Note that this is an heuristic approximation of the optimal (exact) path.

    This implementation has ``O(nd)`` time and space complexity,
    where ``n`` is the number of MFCC window shifts (vectors)
    of the real wave,
    and ``d`` is the number of MFCC window shifts
    corresponding to the margin. """

    ALLOWED_VALUES = [EXACT, STRIPE]
    """ List of all the allowed values """


class DTWAlignerNotInitialized(Exception):
    """
    Error raised when trying to compute
    using an DTWAligner object whose real and/or synt waves
    are not initialized yet.
    """
    pass


class DTWAligner(Loggable):
    """
    The audio wave aligner.

    The two waves, henceforth named real and synthesized,
    can be passed as :class:`~aeneas.audiofilemfcc.AudioFileMFCC` objects
    or as file paths.
    In the latter case, MFCCs will be extracted upon object creation.

    :param real_wave_mfcc: the real audio file
    :type  real_wave_mfcc: :class:`~aeneas.audiofilemfcc.AudioFileMFCC`
    :param synt_wave_mfcc: the synthesized audio file
    :type  synt_wave_mfcc: :class:`~aeneas.audiofilemfcc.AudioFileMFCC`
    :param string real_wave_path: the path to the real audio file
    :param string synt_wave_path: the path to the synthesized audio file
    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    :raises: ValueError: if ``real_wave_mfcc`` or ``synt_wave_mfcc`` is not ``None``
                         but not of type :class:`~aeneas.audiofilemfcc.AudioFileMFCC`
    :raises: ValueError: if ``real_wave_path`` or ``synt_wave_path`` is not ``None``
                         but it cannot be read
    """

    TAG = u"DTWAligner"

    def __init__(
        self,
        real_wave_mfcc=None,
        synt_wave_mfcc=None,
        real_wave_path=None,
        synt_wave_path=None,
        rconf=None,
        logger=None
    ):
        if (real_wave_mfcc is not None) and (type(real_wave_mfcc) is not AudioFileMFCC):
            raise ValueError(u"Real wave mfcc must be None or of type AudioFileMFCC")
        if (synt_wave_mfcc is not None) and (type(synt_wave_mfcc) is not AudioFileMFCC):
            raise ValueError(u"Synt wave mfcc must be None or of type AudioFileMFCC")
        if (real_wave_path is not None) and (not gf.file_can_be_read(real_wave_path)):
            raise ValueError(u"Real wave cannot be read")
        if (synt_wave_path is not None) and (not gf.file_can_be_read(synt_wave_path)):
            raise ValueError(u"Synt wave cannot be read")
        if (rconf is not None) and (rconf[RuntimeConfiguration.DTW_ALGORITHM] not in DTWAlgorithm.ALLOWED_VALUES):
            raise ValueError(u"Algorithm value not allowed")
        super(DTWAligner, self).__init__(rconf=rconf, logger=logger)
        self.real_wave_mfcc = real_wave_mfcc
        self.synt_wave_mfcc = synt_wave_mfcc
        self.real_wave_path = real_wave_path
        self.synt_wave_path = synt_wave_path
        if (self.real_wave_mfcc is None) and (self.real_wave_path is not None):
            self.real_wave_mfcc = AudioFileMFCC(self.real_wave_path, rconf=self.rconf, logger=self.logger)
        if (self.synt_wave_mfcc is None) and (self.synt_wave_path is not None):
            self.synt_wave_mfcc = AudioFileMFCC(self.synt_wave_path, rconf=self.rconf, logger=self.logger)

    def compute_accumulated_cost_matrix(self):
        """
        Compute the accumulated cost matrix, and return it.

        :rtype: :class:`numpy.ndarray` (2D)
        :raises: RuntimeError: if both the C extension and
                               the pure Python code did not succeed.

        .. versionadded:: 1.2.0
        """
        dtw = self._setup_dtw()
        self.log(u"Returning accumulated cost matrix")
        return dtw.compute_accumulated_cost_matrix()

    def compute_path(self):
        """
        Compute the min cost path between the two waves, and return it.

        Return the computed path as a tuple with two elements,
        each being a :class:`numpy.ndarray` (1D) of ``int`` indices: ::

        ([r_1, r_2, ..., r_k], [s_1, s_2, ..., s_k])

        where ``r_i`` are the indices in the real wave
        and ``s_i`` are the indices in the synthesized wave,
        and ``k`` is the length of the min cost path.

        :rtype: tuple (see above)
        :raises: RuntimeError: if both the C extension and
                               the pure Python code did not succeed.
        """
        dtw = self._setup_dtw()
        self.log(u"Computing path...")
        wave_path = dtw.compute_path()
        self.log(u"Computing path... done")
        self.log(u"Translating path to full wave indices...")
        real_indices = numpy.array([t[0] for t in wave_path])
        synt_indices = numpy.array([t[1] for t in wave_path])
        if self.rconf.mmn:
            self.log(u"Translating real indices with masked_middle_map...")
            real_indices = self.real_wave_mfcc.masked_middle_map[real_indices]
            real_indices[0] = self.real_wave_mfcc.head_length
            self.log(u"Translating real indices with masked_middle_map... done")
            self.log(u"Translating synt indices with masked_middle_map...")
            synt_indices = self.synt_wave_mfcc.masked_middle_map[synt_indices]
            self.log(u"Translating synt indices with masked_middle_map... done")
        else:
            self.log(u"Translating real indices by adding head_length...")
            real_indices += self.real_wave_mfcc.head_length
            self.log(u"Translating real indices by adding head_length... done")
            self.log(u"Nothing to do with synt indices")
        self.log(u"Translating path to full wave indices... done")
        return (real_indices, synt_indices)

    def compute_boundaries(self, synt_anchors):
        """
        Compute the min cost path between the two waves,
        and return a list of boundary points,
        representing the argmin values with respect to
        the provided ``synt_anchors`` timings.

        If ``synt_anchors`` has ``k`` elements,
        the returned array will have ``k+1`` elements,
        accounting for the tail fragment.

        :param synt_anchors: the anchor time values (in seconds) of the synthesized fragments,
                             each representing the begin time in the synthesized wave
                             of the corresponding fragment
        :type  synt_anchors: list of :class:`~aeneas.exacttiming.TimeValue`

        Return the list of boundary indices.

        :rtype: :class:`numpy.ndarray` (1D)
        """
        self.log(u"Computing path...")
        real_indices, synt_indices = self.compute_path()
        self.log(u"Computing path... done")

        self.log(u"Computing boundary indices...")
        # both real_indices and synt_indices are w.r.t. the full wave
        self.log([u"Fragments:        %d", len(synt_anchors)])
        self.log([u"Path length:      %d", len(real_indices)])
        # synt_anchors as in seconds, convert them in MFCC indices
        # see also issue #102
        mws = self.rconf.mws
        sample_rate = self.rconf.sample_rate
        samples_per_mws = mws * sample_rate
        if samples_per_mws.is_integer:
            anchor_indices = numpy.array([int(a[0] / mws) for a in synt_anchors])
        else:
            #
            # NOTE this is not elegant, but it saves the day for the user
            #
            self.log_warn(u"The number of samples in each window shift is not an integer, time drift might occur.")
            anchor_indices = numpy.array([(int(a[0] * sample_rate / mws) / sample_rate) for a in synt_anchors])
        #
        # right side sets the split point at the very beginning of "next" fragment
        #
        # NOTE clip() is needed since searchsorted() with side="right" might return
        #      an index == len(synt_indices) == len(real_indices)
        #      when the insertion point is past the last element of synt_indices
        #      causing the fancy indexing real_indices[...] below might fail
        begin_indices = numpy.clip(numpy.searchsorted(synt_indices, anchor_indices, side="right"), 0, len(synt_indices) - 1)
        # first split must occur at zero
        begin_indices[0] = 0
        #
        # map onto real indices, obtaining "default" boundary indices
        #
        # NOTE since len(synt_indices) == len(real_indices)
        #      and because the numpy.clip() above, the fancy indexing is always valid
        #
        boundary_indices = numpy.append(real_indices[begin_indices], self.real_wave_mfcc.tail_begin)
        self.log([u"Boundary indices: %d", len(boundary_indices)])
        self.log(u"Computing boundary indices... done")
        return boundary_indices

    def _setup_dtw(self):
        """
        Set the DTW object up.
        """
        # check we have the AudioFileMFCC objects
        if (self.real_wave_mfcc is None) or (self.real_wave_mfcc.middle_mfcc is None):
            self.log_exc(u"The real wave MFCCs are not initialized", None, True, DTWAlignerNotInitialized)
        if (self.synt_wave_mfcc is None) or (self.synt_wave_mfcc.middle_mfcc is None):
            self.log_exc(u"The synt wave MFCCs are not initialized", None, True, DTWAlignerNotInitialized)

        # setup
        algorithm = self.rconf[RuntimeConfiguration.DTW_ALGORITHM]
        delta = int(2 * self.rconf.dtw_margin / self.rconf[RuntimeConfiguration.MFCC_WINDOW_SHIFT])
        mfcc2_length = self.synt_wave_mfcc.middle_length
        self.log([u"Requested algorithm: '%s'", algorithm])
        self.log([u"delta = %d", delta])
        self.log([u"m = %d", mfcc2_length])
        # check if delta is >= length of synt wave
        if mfcc2_length <= delta:
            self.log(u"We have mfcc2_length <= delta")
            if (self.rconf[RuntimeConfiguration.C_EXTENSIONS]) and (gf.can_run_c_extension()):
                # the C code can be run: since it is still faster, do not run EXACT
                self.log(u"C extensions enabled and loaded: not selecting EXACT algorithm")
            else:
                self.log(u"Selecting EXACT algorithm")
                algorithm = DTWAlgorithm.EXACT

        # select mask here
        if self.rconf.mmn:
            self.log(u"Using masked MFCC")
            real_mfcc = self.real_wave_mfcc.masked_middle_mfcc
            synt_mfcc = self.synt_wave_mfcc.masked_middle_mfcc
        else:
            self.log(u"Using unmasked MFCC")
            real_mfcc = self.real_wave_mfcc.middle_mfcc
            synt_mfcc = self.synt_wave_mfcc.middle_mfcc

        # execute the selected algorithm
        if algorithm == DTWAlgorithm.EXACT:
            self.log(u"Computing with EXACT algo")
            dtw = DTWExact(
                m1=real_mfcc,
                m2=synt_mfcc,
                rconf=self.rconf,
                logger=self.logger
            )
        else:
            self.log(u"Computing with STRIPE algo")
            dtw = DTWStripe(
                m1=real_mfcc,
                m2=synt_mfcc,
                delta=delta,
                rconf=self.rconf,
                logger=self.logger
            )
        return dtw


class DTWStripe(Loggable):

    TAG = u"DTWStripe"

    def __init__(self, m1, m2, delta, rconf=None, logger=None):
        super(DTWStripe, self).__init__(rconf=rconf, logger=logger)
        self.m1 = m1
        self.m2 = m2
        self.delta = delta

    def compute_accumulated_cost_matrix(self):
        return gf.run_c_extension_with_fallback(
            self.log,
            "cdtw",
            self._compute_acm_c_extension,
            self._compute_acm_pure_python,
            (),
            rconf=self.rconf
        )

    def _compute_acm_c_extension(self):
        self.log(u"Computing acm using C extension...")
        try:
            self.log(u"Importing cdtw...")
            import aeneas.cdtw.cdtw
            self.log(u"Importing cdtw... done")
            # discard first MFCC component
            mfcc1 = self.m1[1:, :]
            mfcc2 = self.m2[1:, :]
            n = mfcc1.shape[1]
            m = mfcc2.shape[1]
            delta = self.delta
            self.log([u"n m delta: %d %d %d", n, m, delta])
            if delta > m:
                self.log(u"Limiting delta to m")
                delta = m
            cost_matrix, centers = aeneas.cdtw.cdtw.compute_cost_matrix_step(mfcc1, mfcc2, delta)
            accumulated_cost_matrix = aeneas.cdtw.cdtw.compute_accumulated_cost_matrix_step(cost_matrix, centers)
            self.log(u"Computing acm using C extension... done")
            return (True, accumulated_cost_matrix)
        except Exception as exc:
            self.log_exc(u"An unexpected error occurred while running cdtw", exc, False, None)
        return (False, None)

    def _compute_acm_pure_python(self):
        self.log(u"Computing acm using pure Python code...")
        try:
            cost_matrix, centers = self._compute_cost_matrix()
            accumulated_cost_matrix = self._compute_accumulated_cost_matrix(cost_matrix, centers)
            self.log(u"Computing acm using pure Python code... done")
            return (True, accumulated_cost_matrix)
        except Exception as exc:
            self.log_exc(u"An unexpected error occurred while running pure Python code", exc, False, None)
        return (False, None)

    def compute_path(self):
        return gf.run_c_extension_with_fallback(
            self.log,
            "cdtw",
            self._compute_path_c_extension,
            self._compute_path_pure_python,
            (),
            rconf=self.rconf
        )

    def _compute_path_c_extension(self):
        self.log(u"Computing path using C extension...")
        try:
            self.log(u"Importing cdtw...")
            import aeneas.cdtw.cdtw
            self.log(u"Importing cdtw... done")
            # discard first MFCC component
            mfcc1 = self.m1[1:, :]
            mfcc2 = self.m2[1:, :]
            n = mfcc1.shape[1]
            m = mfcc2.shape[1]
            delta = self.delta
            self.log([u"n m delta: %d %d %d", n, m, delta])
            if delta > m:
                self.log(u"Limiting delta to m")
                delta = m
            best_path = aeneas.cdtw.cdtw.compute_best_path(
                mfcc1,
                mfcc2,
                delta
            )
            self.log(u"Computing path using C extension... done")
            return (True, best_path)
        except Exception as exc:
            self.log_exc(u"An unexpected error occurred while running cdtw", exc, False, None)
        return (False, None)

    def _compute_path_pure_python(self):
        self.log(u"Computing path using pure Python code...")
        try:
            cost_matrix, centers = self._compute_cost_matrix()
            accumulated_cost_matrix = self._compute_accumulated_cost_matrix(cost_matrix, centers)
            best_path = self._compute_best_path(accumulated_cost_matrix, centers)
            self.log(u"Computing path using pure Python code... done")
            return (True, best_path)
        except Exception as exc:
            self.log_exc(u"An unexpected error occurred while running pure Python code", exc, False, None)
        return (False, None)

    def _compute_cost_matrix(self):
        self.log(u"Computing cost matrix...")
        # discard first MFCC component
        mfcc1 = self.m1[1:, :]
        mfcc2 = self.m2[1:, :]
        norm2_1 = numpy.sqrt(numpy.sum(mfcc1 ** 2, 0))
        norm2_2 = numpy.sqrt(numpy.sum(mfcc2 ** 2, 0))
        n = mfcc1.shape[1]
        m = mfcc2.shape[1]
        delta = self.delta
        self.log([u"n m delta: %d %d %d", n, m, delta])
        if delta > m:
            self.log(u"Limiting delta to m")
            delta = m
        cost_matrix = numpy.zeros((n, delta))
        centers = numpy.zeros(n, dtype=int)
        for i in range(n):
            # center j at row i
            center_j = (m * i) // n
            # COMMENTED self.log([u"Center at row %d is %d", i, center_j])
            range_start = max(0, center_j - (delta // 2))
            range_end = range_start + delta
            if range_end > m:
                range_end = m
                range_start = range_end - delta
            centers[i] = range_start
            # COMMENTED self.log([u"Range at row %d is %d %d", i, range_start, range_end])
            for j in range(range_start, range_end):
                tmp = mfcc1[:, i].transpose().dot(mfcc2[:, j])
                tmp /= norm2_1[i] * norm2_2[j]
                cost_matrix[i][j - range_start] = 1 - tmp
        self.log(u"Computing cost matrix... done")
        return (cost_matrix, centers)

    def _compute_accumulated_cost_matrix(self, cost_matrix, centers):
        # create accumulated cost matrix
        #
        # a[i][j] = c[i][j] + min(c[i-1][j-1], c[i-1][j], c[i][j-1])
        #
        return self._compute_acm_in_place(cost_matrix, centers)

    def _compute_acm_in_place(self, cost_matrix, centers):
        self.log(u"Computing the acm with the in-place algorithm...")
        n, delta = cost_matrix.shape
        self.log([u"n delta: %d %d", n, delta])
        current_row = numpy.copy(cost_matrix[0, :])
        # COMMENTED cost_matrix[0][0] = current_row[0]
        for j in range(1, delta):
            cost_matrix[0][j] = current_row[j] + cost_matrix[0][j - 1]
        # fill table
        for i in range(1, n):
            current_row = numpy.copy(cost_matrix[i, :])
            offset = centers[i] - centers[i - 1]
            for j in range(delta):
                cost0 = numpy.inf
                if (j + offset) < delta:
                    cost0 = cost_matrix[i - 1][j + offset]
                cost1 = numpy.inf
                if j > 0:
                    cost1 = cost_matrix[i][j - 1]
                cost2 = numpy.inf
                if ((j + offset - 1) < delta) and ((j + offset - 1) >= 0):
                    cost2 = cost_matrix[i - 1][j + offset - 1]
                cost_matrix[i][j] = current_row[j] + min(cost0, cost1, cost2)
        self.log(u"Computing the acm with the in-place algorithm... done")
        return cost_matrix

    # DISABLED
    # def _compute_acm_not_in_place(self, cost_matrix, centers):
    #    self.log(u"Computing the acm with the not-in-place algorithm...")
    #    acc_matrix = numpy.zeros(cost_matrix.shape)
    #    n, delta = acc_matrix.shape
    #    self.log([u"n delta: %d %d", n, delta])
    #    # first row
    #    acc_matrix[0][0] = cost_matrix[0][0]
    #    for j in range(1, delta):
    #        acc_matrix[0][j] = acc_matrix[0][j-1] + cost_matrix[0][j]
    #    # fill table
    #    for i in range(1, n):
    #        offset = centers[i] - centers[i-1]
    #        for j in range(delta):
    #            cost0 = numpy.inf
    #            if (j+offset) < delta:
    #                cost0 = acc_matrix[i-1][j+offset]
    #            cost1 = numpy.inf
    #            if j > 0:
    #                cost1 = acc_matrix[i][j-1]
    #            cost2 = numpy.inf
    #            if ((j+offset-1) < delta) and ((j+offset-1) >= 0):
    #                cost2 = acc_matrix[i-1][j+offset-1]
    #            acc_matrix[i][j] = cost_matrix[i][j] + min(cost0, cost1, cost2)
    #    self.log(u"Computing the acm with the not-in-place algorithm... done")
    #    return acc_matrix

    def _compute_best_path(self, acc_matrix, centers):
        self.log(u"Computing best path...")
        # get dimensions
        n, delta = acc_matrix.shape
        self.log([u"n delta: %d %d", n, delta])
        i = n - 1
        j = delta - 1 + centers[i]
        path = [(i, j)]
        # compute best (min cost) path
        while (i > 0) or (j > 0):
            if i == 0:
                path.append((0, j - 1))
                j -= 1
            elif j == 0:
                path.append((i - 1, 0))
                i -= 1
            else:
                offset = centers[i] - centers[i - 1]
                r_j = j - centers[i]
                cost0 = numpy.inf
                if (r_j + offset) < delta:
                    cost0 = acc_matrix[i - 1][r_j + offset]
                cost1 = numpy.inf
                if r_j > 0:
                    cost1 = acc_matrix[i][r_j - 1]
                cost2 = numpy.inf
                if (r_j > 0) and ((r_j + offset - 1) < delta) and ((r_j + offset - 1) >= 0):
                    cost2 = acc_matrix[i - 1][r_j + offset - 1]
                costs = [
                    cost0,
                    cost1,
                    cost2
                ]
                moves = [
                    (i - 1, j),
                    (i, j - 1),
                    (i - 1, j - 1)
                ]
                min_cost = numpy.argmin(costs)
                # COMMENTED self.log([u"Selected min cost move %d", min_cost])
                min_move = moves[min_cost]
                path.append(min_move)
                i, j = min_move
        # reverse path and return
        path.reverse()
        self.log(u"Computing best path... done")
        return path


class DTWExact(Loggable):

    TAG = u"DTWExact"

    def __init__(self, m1, m2, rconf=None, logger=None):
        super(DTWExact, self).__init__(rconf=rconf, logger=logger)
        self.m1 = m1
        self.m2 = m2

    def compute_accumulated_cost_matrix(self):
        self.log(u"Computing acm using pure Python code...")
        cost_matrix = self._compute_cost_matrix()
        accumulated_cost_matrix = self._compute_accumulated_cost_matrix(cost_matrix)
        self.log(u"Computing acm using pure Python code... done")
        return accumulated_cost_matrix

    def compute_path(self):
        self.log(u"Computing path using pure Python code...")
        accumulated_cost_matrix = self.compute_accumulated_cost_matrix()
        best_path = self._compute_best_path(accumulated_cost_matrix)
        self.log(u"Computing path using pure Python code... done")
        return best_path

    def _compute_cost_matrix(self):
        self.log(u"Computing cost matrix...")
        # discard first MFCC component
        mfcc1 = self.m1[1:, :]
        mfcc2 = self.m2[1:, :]
        norm2_1 = numpy.sqrt(numpy.sum(mfcc1 ** 2, 0))
        norm2_2 = numpy.sqrt(numpy.sum(mfcc2 ** 2, 0))
        # compute dot product
        self.log(u"Computing matrix with transpose+dot...")
        cost_matrix = mfcc1.transpose().dot(mfcc2)
        self.log(u"Computing matrix with transpose+dot... done")
        # normalize
        self.log(u"Normalizing matrix...")
        norm_matrix = numpy.outer(norm2_1, norm2_2)
        cost_matrix = 1 - (cost_matrix / norm_matrix)
        self.log(u"Normalizing matrix... done")
        self.log(u"Computing cost matrix... done")
        return cost_matrix

    def _compute_accumulated_cost_matrix(self, cost_matrix):
        # create accumulated cost matrix
        #
        # a[i][j] = c[i][j] + min(c[i-1][j-1], c[i-1][j], c[i][j-1])
        #
        return self._compute_acm_in_place(cost_matrix)

    def _compute_acm_in_place(self, cost_matrix):
        self.log(u"Computing the acm with the in-place algorithm...")
        n, m = cost_matrix.shape
        self.log([u"n m: %d %d", n, m])
        current_row = numpy.copy(cost_matrix[0, :])
        # COMMENTED cost_matrix[0][0] = current_row[0]
        for j in range(1, m):
            cost_matrix[0][j] = current_row[j] + cost_matrix[0][j - 1]
        for i in range(1, n):
            current_row = numpy.copy(cost_matrix[i, :])
            cost_matrix[i][0] = cost_matrix[i - 1][0] + current_row[0]
            for j in range(1, m):
                cost_matrix[i][j] = current_row[j] + min(
                    cost_matrix[i - 1][j],
                    cost_matrix[i][j - 1],
                    cost_matrix[i - 1][j - 1]
                )
        self.log(u"Computing the acm with the in-place algorithm... done")
        return cost_matrix

    # DISABLED
    # def _compute_acm_not_in_place(self, cost_matrix):
    #    self.log(u"Computing the acm with the not-in-place algorithm...")
    #    acc_matrix = numpy.zeros(cost_matrix.shape)
    #    n, m = acc_matrix.shape
    #    self.log([u"n m: %d %d", n, m])
    #    acc_matrix[0][0] = cost_matrix[0][0]
    #    for j in range(1, m):
    #        acc_matrix[0][j] = acc_matrix[0][j-1] + cost_matrix[0][j]
    #    for i in range(1, n):
    #        acc_matrix[i][0] = acc_matrix[i-1][0] + cost_matrix[i][0]
    #    for i in range(1, n):
    #        for j in range(1, m):
    #            acc_matrix[i][j] = cost_matrix[i][j] + min(
    #                acc_matrix[i-1][j],
    #                acc_matrix[i][j-1],
    #                acc_matrix[i-1][j-1]
    #            )
    #    self.log(u"Computing the acm with the not-in-place algorithm... done")
    #    return acc_matrix

    def _compute_best_path(self, acc_matrix):
        self.log(u"Computing best path...")
        # get dimensions
        n, m = acc_matrix.shape
        self.log([u"n m: %d %d", n, m])
        i = n - 1
        j = m - 1
        path = [(i, j)]
        # compute best (min cost) path
        while (i > 0) or (j > 0):
            if i == 0:
                path.append((0, j - 1))
                j -= 1
            elif j == 0:
                path.append((i - 1, 0))
                i -= 1
            else:
                costs = [
                    acc_matrix[i - 1][j],
                    acc_matrix[i][j - 1],
                    acc_matrix[i - 1][j - 1]
                ]
                moves = [
                    (i - 1, j),
                    (i, j - 1),
                    (i - 1, j - 1)
                ]
                min_cost = numpy.argmin(costs)
                # COMMENTED self.log([u"Selected min cost move %d", min_cost])
                min_move = moves[min_cost]
                path.append(min_move)
                i, j = min_move
        # reverse path and return
        path.reverse()
        self.log(u"Computing best path... done")
        return path
