#!/usr/bin/env python
# coding=utf-8

"""
This module contains the implementation
of dynamic time warping (DTW) algorithms
to align two audio waves, represented by their
Mel-frequency cepstral coefficients (MFCCs).

The two classes provided by this module are:

1. :class:`aeneas.dtw.DTWAlgorithm`
   is an enumeration of the available algorithms.
2. :class:`aeneas.dtw.DTWAligner`
   is the actual feature extractor and aligner.

To align two wave files:

1. build an :class:`aeneas.dtw.DTWAligner` object
   passing the paths of the two wave files
   in the constructor, possibly with custom arguments
   to fine-tune the alignment;
2. call ``compute_mfcc`` to extract the MFCCs of the two wave files;
3. call ``compute_path`` to compute the min cost path between
   the MFCC representations of the two wave files;
4. obtain the map between the two wave files by reading the
   ``computed_map`` property.
"""

import numpy
import os

from aeneas.audiofile import AudioFileMonoWAV
from aeneas.logger import Logger
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf

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

class DTWAlgorithm(object):
    """
    Enumeration of the DTW algorithms that can be used
    for the alignment of two audio waves.
    """

    EXACT = "exact"
    """ Classical (exact) DTW algorithm.

    This implementation has ``O(nm)`` time and space complexity,
    where ``n`` (respectively, ``m``) is the number of MFCCs
    of the real (respectively, synthesized) wave. """

    STRIPE = "stripe"
    """ DTW algorithm restricted to a stripe around the main diagonal
    (Sakoe-Chiba Band), for optimized memory usage and processing.

    Note that this is an heuristic approximation of the optimal (exact) path.

    This implementation has ``O(nd)`` time and space complexity,
    where ``n`` is the number of MFCCs of the real wave,
    and ``d`` is the number of MFCCs
    corresponding to the margin. """

    ALLOWED_VALUES = [EXACT, STRIPE]
    """ List of all the allowed values """



class DTWAligner(object):
    """
    The MFCC extractor and wave aligner.

    :param real_wave_path: the path to the real wav file (must be mono!)
    :type  real_wave_path: string (path)
    :param synt_wave_path: the path to the synthesized wav file (must be mono!)
    :type  synt_wave_path: string (path)
    :param frame_rate: the MFCC frame rate, in frames per second. Default:
                       :class:`aeneas.globalconstants.MFCC_FRAME_RATE`
    :type  frame_rate: int
    :param margin: the margin to be used in DTW stripe algorithms, in seconds.
                   Default: :class:`aeneas.globalconstants.ALIGNER_MARGIN`
    :type  margin: int
    :param algorithm: the DTW algorithm to be used when aligning the waves
    :type  algorithm: :class:`aeneas.dtw.DTWAlgorithm`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`

    :raise ValueError: if ``real_wave_path`` or ``synt_wave_path`` is ``None`` or it does not exist, or if ``algorithm`` is not an allowed value
    """

    TAG = "DTWAligner"

    def __init__(
            self,
            real_wave_path=None,
            synt_wave_path=None,
            frame_rate=gc.MFCC_FRAME_RATE,
            margin=gc.ALIGNER_MARGIN,
            algorithm=DTWAlgorithm.STRIPE,
            logger=None
        ):
        if (real_wave_path is not None) and (not gf.file_exists(real_wave_path)):
            raise ValueError("Real wave path does not exist")
        if (synt_wave_path is not None) and (not gf.file_exists(synt_wave_path)):
            raise ValueError("Synt wave path does not exist")
        if algorithm not in DTWAlgorithm.ALLOWED_VALUES:
            raise ValueError("Algorithm value not allowed")
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()
        self.real_wave_path = real_wave_path
        self.synt_wave_path = synt_wave_path
        self.frame_rate = frame_rate
        self.margin = margin
        self.algorithm = algorithm
        self.real_wave_full_mfcc = None
        self.synt_wave_full_mfcc = None
        self.real_wave_length = None
        self.synt_wave_length = None
        self.computed_path = None

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    @property
    def real_wave_full_mfcc(self):
        """
        MFCCs of the real wave, including the 0-th.

        :rtype: numpy 2D array

        .. versionadded:: 1.1.0
        """
        return self.__real_wave_full_mfcc

    @real_wave_full_mfcc.setter
    def real_wave_full_mfcc(self, real_wave_full_mfcc):
        self.__real_wave_full_mfcc = real_wave_full_mfcc

    @property
    def real_wave_length(self):
        """
        The length, in seconds, of the real wave.

        :rtype: float

        .. versionadded:: 1.1.0
        """
        return self.__real_wave_length

    @real_wave_length.setter
    def real_wave_length(self, real_wave_length):
        self.__real_wave_length = real_wave_length

    @property
    def synt_wave_full_mfcc(self):
        """
        MFCCs of the synthesized wave, including the 0-th.

        :rtype: numpy 2D array

        .. versionadded:: 1.1.0
        """
        return self.__synt_wave_full_mfcc

    @synt_wave_full_mfcc.setter
    def synt_wave_full_mfcc(self, synt_wave_full_mfcc):
        self.__synt_wave_full_mfcc = synt_wave_full_mfcc

    @property
    def synt_wave_length(self):
        """
        The length, in seconds, of the synthesized wave.

        :rtype: float

        .. versionadded:: 1.1.0
        """
        return self.__synt_wave_length

    @synt_wave_length.setter
    def synt_wave_length(self, synt_wave_length):
        self.__synt_wave_length = synt_wave_length

    def compute_mfcc(self):
        """
        Compute the MFCCs of the two waves,
        and store them internally.

        :raise IOError: if the real or synt wave file cannot be read
        """
        if not gf.file_exists(self.real_wave_path):
            raise IOError("Real wave path is None or it does not exist")

        if not gf.file_exists(self.synt_wave_path):
            raise IOError("Synt wave path is None or it does not exist")

        self._log("Computing MFCCs for real wave...")
        wave = AudioFileMonoWAV(self.real_wave_path, logger=self.logger)
        wave.extract_mfcc(self.frame_rate)
        self.real_wave_full_mfcc = wave.audio_mfcc
        self.real_wave_length = wave.audio_length
        self._log("Computing MFCCs for real wave... done")

        self._log("Computing MFCCs for synt wave...")
        wave = AudioFileMonoWAV(self.synt_wave_path, logger=self.logger)
        wave.extract_mfcc(self.frame_rate)
        self.synt_wave_full_mfcc = wave.audio_mfcc
        self.synt_wave_length = wave.audio_length
        self._log("Computing MFCCs for synt wave... done")

    def compute_path(self):
        """
        Compute the min cost path between the two waves,
        and store it interally.
        """
        dtw = self._setup_dtw()
        self._log("Computing path...")
        self.computed_path = dtw.compute_path()
        self._log("Computing path... done")

    def compute_accumulated_cost_matrix(self):
        """
        Compute the accumulated cost matrix,
        and return it.

        :rtype: numpy 2D array

        .. versionadded:: 1.2.0
        """
        dtw = self._setup_dtw()
        self._log("Returning accumulated cost matrix")
        return dtw.compute_accumulated_cost_matrix()

    def _setup_dtw(self):
        """ Setup DTW object """
        # setup
        dtw = None
        algorithm = self.algorithm
        delta = self.frame_rate * (self.margin * 2)
        mfcc2_size = self.synt_wave_full_mfcc.shape[1]
        self._log(["Requested algorithm: '%s'", algorithm])
        self._log(["delta = %d", delta])
        self._log(["m = %d", mfcc2_size])
        # check if delta is >= length of synt wave
        if mfcc2_size <= delta:
            self._log("We have mfcc2_size <= delta")
            if gc.USE_C_EXTENSIONS and gf.can_run_c_extension():
                # the C code can be run: since it is still faster, do not run EXACT
                self._log("C extensions enabled and loaded: not selecting EXACT algorithm")
            elif gc.ALIGNER_USE_EXACT_ALGORITHM_WHEN_MARGIN_TOO_LARGE:
                self._log("Selecting EXACT algorithm")
                algorithm = DTWAlgorithm.EXACT
            else:
                self._log("Selecting EXACT algorithm disabled in gc")

        # execute the selected algorithm
        if algorithm == DTWAlgorithm.STRIPE:
            self._log("Computing with STRIPE algo")
            dtw = DTWStripe(
                self.real_wave_full_mfcc,
                self.synt_wave_full_mfcc,
                delta,
                self.logger
            )
        if algorithm == DTWAlgorithm.EXACT:
            self._log("Computing with EXACT algo")
            dtw = DTWExact(
                self.real_wave_full_mfcc,
                self.synt_wave_full_mfcc,
                self.logger
            )
        return dtw

    @property
    def computed_map(self):
        """
        Return the computed map between the two waves,
        as a list of lists, each being a pair of floats: ::

        [[r_1, s_1], [r_2, s_2], ..., [r_k, s_k]]

        where ``r_i`` are the time instants in the real wave
        and ``s_i`` are the time instants in the synthesized wave,
        and ``k = n + m`` (or ``k = n + d``)
        is the length of the min cost path.

        :rtype: list of pairs of floats (see above)
        """
        result = []
        for i in range(len(self.computed_path)):
            real_time = 1.0 * self.computed_path[i][0] / self.frame_rate
            synt_time = 1.0 * self.computed_path[i][1] / self.frame_rate
            result.append([real_time, synt_time])
        return result



class DTWStripe(object):

    TAG = "DTWStripe"

    def __init__(self, m1, m2, delta, logger):
        self.m1 = m1
        self.m2 = m2
        self.delta = delta
        self.logger = logger

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def compute_accumulated_cost_matrix(self):
        if gc.USE_C_EXTENSIONS:
            self._log("C extensions enabled in gc")
            if gf.can_run_c_extension("cdtw"):
                self._log("C extensions enabled in gc and cdtw can be loaded")
                try:
                    return self._compute_acm_c_extension()
                except:
                    self._log(
                        "An error occurred running cdtw",
                         severity=Logger.WARNING
                    )
            else:
                self._log("C extensions enabled in gc, but cdtw cannot be loaded")
        else:
            self._log("C extensions disabled in gc")
        self._log("Running the pure Python code")
        return self._compute_acm_pure_python()

    def _compute_acm_c_extension(self):
        self._log("Computing acm using C extension...")
        self._log("Importing cdtw...")
        import aeneas.cdtw
        self._log("Importing cdtw... done")
        # discard first MFCC component
        mfcc1 = self.m1[1:, :]
        mfcc2 = self.m2[1:, :]
        norm2_1 = numpy.sqrt(numpy.sum(mfcc1 ** 2, 0))
        norm2_2 = numpy.sqrt(numpy.sum(mfcc2 ** 2, 0))
        n = mfcc1.shape[1]
        m = mfcc2.shape[1]
        delta = self.delta
        self._log(["n m delta: %d %d %d", n, m, delta])
        if delta > m:
            self._log("Limiting delta to m")
            delta = m
        cost_matrix, centers = aeneas.cdtw.cdtw_compute_cost_matrix_step(mfcc1, mfcc2, norm2_1, norm2_2, delta)
        accumulated_cost_matrix = aeneas.cdtw.cdtw_compute_accumulated_cost_matrix_step(cost_matrix, centers)
        self._log("Computing acm using C extension... done")
        return accumulated_cost_matrix

    def _compute_acm_pure_python(self):
        self._log("Computing acm using pure Python code...")
        self._log("Computing cost matrix...")
        cost_matrix, centers = self._compute_cost_matrix()
        self._log("Computing cost matrix... done")
        self._log("Computing accumulated cost matrix...")
        accumulated_cost_matrix = self._compute_accumulated_cost_matrix(cost_matrix, centers)
        self._log("Computing accumulated cost matrix... done")
        self._log("Computing acm using pure Python code... done")
        return accumulated_cost_matrix

    def compute_path(self):
        if gc.USE_C_EXTENSIONS:
            self._log("C extensions enabled in gc")
            if gf.can_run_c_extension("cdtw"):
                self._log("C extensions enabled in gc and cdtw can be loaded")
                try:
                    return self._compute_path_c_extension()
                except:
                    self._log(
                        "An error occurred running cdtw",
                         severity=Logger.WARNING
                    )
            else:
                self._log("C extensions enabled in gc, but cdtw cannot be loaded")
        else:
            self._log("C extensions disabled in gc")
        self._log("Running the pure Python code")
        return self._compute_path_pure_python()

    def _compute_path_c_extension(self):
        self._log("Computing path using C extension...")
        self._log("Importing cdtw...")
        import aeneas.cdtw
        self._log("Importing cdtw... done")
        # discard first MFCC component
        mfcc1 = self.m1[1:, :]
        mfcc2 = self.m2[1:, :]
        norm2_1 = numpy.sqrt(numpy.sum(mfcc1 ** 2, 0))
        norm2_2 = numpy.sqrt(numpy.sum(mfcc2 ** 2, 0))
        n = mfcc1.shape[1]
        m = mfcc2.shape[1]
        delta = self.delta
        self._log(["n m delta: %d %d %d", n, m, delta])
        if delta > m:
            self._log("Limiting delta to m")
            delta = m
        best_path = aeneas.cdtw.cdtw_compute_best_path(
            mfcc1,
            mfcc2,
            norm2_1,
            norm2_2,
            delta
        )
        self._log("Computing path using C extension... done")
        return best_path

    def _compute_path_pure_python(self):
        self._log("Computing path using pure Python code...")
        self._log("Computing cost matrix...")
        cost_matrix, centers = self._compute_cost_matrix()
        self._log("Computing cost matrix... done")
        self._log("Computing accumulated cost matrix...")
        accumulated_cost_matrix = self._compute_accumulated_cost_matrix(cost_matrix, centers)
        self._log("Computing accumulated cost matrix... done")
        self._log("Computing best path...")
        best_path = self._compute_best_path(accumulated_cost_matrix, centers)
        self._log("Computing best path... done")
        self._log("Computing path using pure Python code... done")
        return best_path

    def _compute_cost_matrix(self):
        # discard first MFCC component
        mfcc1 = self.m1[1:, :]
        mfcc2 = self.m2[1:, :]
        norm2_1 = numpy.sqrt(numpy.sum(mfcc1 ** 2, 0))
        norm2_2 = numpy.sqrt(numpy.sum(mfcc2 ** 2, 0))
        n = mfcc1.shape[1]
        m = mfcc2.shape[1]
        delta = self.delta
        self._log(["n m delta: %d %d %d", n, m, delta])
        if delta > m:
            self._log("Limiting delta to m")
            delta = m
        cost_matrix = numpy.zeros((n, delta))
        centers = numpy.zeros(n)
        for i in range(n):
            # center j at row i
            center_j = (m * i) / n
            #self._log(["Center at row %d is %d", i, center_j])
            range_start = max(0, center_j - (delta / 2))
            range_end = range_start + delta
            if range_end > m:
                range_end = m
                range_start = range_end - delta
            centers[i] = range_start
            #self._log(["Range at row %d is %d %d", i, range_start, range_end])
            for j in range(range_start, range_end):
                tmp = mfcc1[:, i].transpose().dot(mfcc2[:, j])
                tmp /= norm2_1[i] * norm2_2[j]
                cost_matrix[i][j - range_start] = 1 - tmp
        return (cost_matrix, centers)

    def _compute_accumulated_cost_matrix(self, cost_matrix, centers):
        # create accumulated cost matrix
        #
        # a[i][j] = c[i][j] + min(c[i-1][j-1], c[i-1][j], c[i][j-1])
        #
        if gc.ALIGNER_USE_IN_PLACE_ALGORITHMS:
            return self._compute_acm_in_place(cost_matrix, centers)
        else:
            return self._compute_acm_not_in_place(cost_matrix, centers)

    def _compute_acm_in_place(self, cost_matrix, centers):
        self._log("Using the in-place algorithm for computing the acm")
        n, delta = cost_matrix.shape
        self._log(["n delta: %d %d", n, delta])
        current_row = numpy.copy(cost_matrix[0, :])
        #cost_matrix[0][0] = current_row[0]
        for j in range(1, delta):
            cost_matrix[0][j] = current_row[j] + cost_matrix[0][j-1]
        # fill table
        for i in range(1, n):
            current_row = numpy.copy(cost_matrix[i, :])
            offset = centers[i] - centers[i-1]
            for j in range(delta):
                cost0 = numpy.inf
                if (j+offset) < delta:
                    cost0 = cost_matrix[i-1][j+offset]
                cost1 = numpy.inf
                if j > 0:
                    cost1 = cost_matrix[i][j-1]
                cost2 = numpy.inf
                if ((j+offset-1) < delta) and ((j+offset-1) >= 0):
                    cost2 = cost_matrix[i-1][j+offset-1]
                cost_matrix[i][j] = current_row[j] + min(cost0, cost1, cost2)
        return cost_matrix

    def _compute_acm_not_in_place(self, cost_matrix, centers):
        self._log("Using the not-in-place algorithm for computing the acm")
        acc_matrix = numpy.zeros(cost_matrix.shape)
        n, delta = acc_matrix.shape
        self._log(["n delta: %d %d", n, delta])
        # first row
        acc_matrix[0][0] = cost_matrix[0][0]
        for j in range(1, delta):
            acc_matrix[0][j] = acc_matrix[0][j-1] + cost_matrix[0][j]
        # fill table
        for i in range(1, n):
            offset = centers[i] - centers[i-1]
            for j in range(delta):
                cost0 = numpy.inf
                if (j+offset) < delta:
                    cost0 = acc_matrix[i-1][j+offset]
                cost1 = numpy.inf
                if j > 0:
                    cost1 = acc_matrix[i][j-1]
                cost2 = numpy.inf
                if ((j+offset-1) < delta) and ((j+offset-1) >= 0):
                    cost2 = acc_matrix[i-1][j+offset-1]
                acc_matrix[i][j] = cost_matrix[i][j] + min(cost0, cost1, cost2)
        return acc_matrix

    def _compute_best_path(self, acc_matrix, centers):
        # get dimensions
        n, delta = acc_matrix.shape
        self._log(["n delta: %d %d", n, delta])
        i = n - 1
        j = delta - 1 + centers[i]
        path = [(i, j)]
        # compute best (min cost) path
        while (i > 0) or (j > 0):
            if i == 0:
                path.append((0, j-1))
                j -= 1
            elif j == 0:
                path.append((i-1, 0))
                i -= 1
            else:
                offset = centers[i] - centers[i-1]
                r_j = j - centers[i]
                cost0 = numpy.inf
                if (r_j+offset) < delta:
                    cost0 = acc_matrix[i-1][r_j+offset]
                cost1 = numpy.inf
                if r_j > 0:
                    cost1 = acc_matrix[i][r_j-1]
                cost2 = numpy.inf
                if (r_j > 0) and ((r_j+offset-1) < delta) and ((r_j+offset-1) >= 0):
                    cost2 = acc_matrix[i-1][r_j+offset-1]
                costs = [
                    cost0,
                    cost1,
                    cost2
                ]
                moves = [
                    (i-1, j),
                    (i, j-1),
                    (i-1, j-1)
                ]
                min_cost = numpy.argmin(costs)
                #self._log(["Selected min cost move %d", min_cost])
                min_move = moves[min_cost]
                path.append(min_move)
                i, j = min_move
        # reverse path and return
        path.reverse()
        return path



class DTWExact(object):

    TAG = "DTWExact"

    def __init__(self, m1, m2, logger):
        self.m1 = m1
        self.m2 = m2
        self.logger = logger

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def compute_accumulated_cost_matrix(self):
        self._log("Computing path using pure Python code...")
        self._log("Computing cost matrix...")
        cost_matrix = self._compute_cost_matrix()
        self._log("Computing cost matrix... done")
        self._log("Computing accumulated cost matrix...")
        accumulated_cost_matrix = self._compute_accumulated_cost_matrix(cost_matrix)
        self._log("Computing accumulated cost matrix... done")
        self._log("Computing path using pure Python code... done")
        return accumulated_cost_matrix

    def compute_path(self):
        self._log("Computing path using pure Python code...")
        self._log("Computing cost matrix...")
        cost_matrix = self._compute_cost_matrix()
        self._log("Computing cost matrix... done")
        self._log("Computing accumulated cost matrix...")
        accumulated_cost_matrix = self._compute_accumulated_cost_matrix(cost_matrix)
        self._log("Computing accumulated cost matrix... done")
        self._log("Computing best path...")
        best_path = self._compute_best_path(accumulated_cost_matrix)
        self._log("Computing best path... done")
        self._log("Computing path using pure Python code... done")
        return best_path

    def _compute_cost_matrix(self):
        # discard first MFCC component
        mfcc1 = self.m1[1:, :]
        mfcc2 = self.m2[1:, :]
        norm2_1 = numpy.sqrt(numpy.sum(mfcc1 ** 2, 0))
        norm2_2 = numpy.sqrt(numpy.sum(mfcc2 ** 2, 0))
        # compute dot product
        self._log("Computing matrix with transpose+dot...")
        cost_matrix = mfcc1.transpose().dot(mfcc2)
        self._log("Computing matrix with transpose+dot... done")
        # normalize
        self._log("Normalizing matrix...")

        #
        # slower version
        #
        #for i in range(len(norm2_1)):
        #    for j in range(len(norm2_2)):
        #        cost_matrix[i][j] /= norm2_1[i] * norm2_2[j]
        #        cost_matrix[i][j] = 1 - cost_matrix[i][j]

        #
        # faster version
        #
        norm_matrix = numpy.outer(norm2_1, norm2_2)
        cost_matrix = 1 - (cost_matrix / norm_matrix)

        self._log("Normalizing matrix... done")
        return cost_matrix

    def _compute_accumulated_cost_matrix(self, cost_matrix):
        # create accumulated cost matrix
        #
        # a[i][j] = c[i][j] + min(c[i-1][j-1], c[i-1][j], c[i][j-1])
        #
        if gc.ALIGNER_USE_IN_PLACE_ALGORITHMS:
            return self._compute_acm_in_place(cost_matrix)
        else:
            return self._compute_acm_not_in_place(cost_matrix)

    def _compute_acm_in_place(self, cost_matrix):
        self._log("Using the in-place algorithm for computing the acm")
        n, m = cost_matrix.shape
        self._log(["n m: %d %d", n, m])
        current_row = numpy.copy(cost_matrix[0, :])
        #cost_matrix[0][0] = current_row[0]
        for j in range(1, m):
            cost_matrix[0][j] = current_row[j] + cost_matrix[0][j-1]
        for i in range(1, n):
            current_row = numpy.copy(cost_matrix[i, :])
            cost_matrix[i][0] = cost_matrix[i-1][0] + current_row[0]
            for j in range(1, m):
                cost_matrix[i][j] = current_row[j] + min(
                    cost_matrix[i-1][j],
                    cost_matrix[i][j-1],
                    cost_matrix[i-1][j-1]
                )
        return cost_matrix

    def _compute_acm_not_in_place(self, cost_matrix):
        self._log("Using the not-in-place algorithm for computing the acm")
        acc_matrix = numpy.zeros(cost_matrix.shape)
        n, m = acc_matrix.shape
        self._log(["n m: %d %d", n, m])
        acc_matrix[0][0] = cost_matrix[0][0]
        for j in range(1, m):
            acc_matrix[0][j] = acc_matrix[0][j-1] + cost_matrix[0][j]
        for i in range(1, n):
            acc_matrix[i][0] = acc_matrix[i-1][0] + cost_matrix[i][0]
        for i in range(1, n):
            for j in range(1, m):
                acc_matrix[i][j] = cost_matrix[i][j] + min(
                    acc_matrix[i-1][j],
                    acc_matrix[i][j-1],
                    acc_matrix[i-1][j-1]
                )
        return acc_matrix

    def _compute_best_path(self, acc_matrix):
        # get dimensions
        n, m = acc_matrix.shape
        self._log(["n m: %d %d", n, m])
        i = n - 1
        j = m - 1
        path = [(i, j)]
        # compute best (min cost) path
        while (i > 0) or (j > 0):
            if i == 0:
                path.append((0, j-1))
                j -= 1
            elif j == 0:
                path.append((i-1, 0))
                i -= 1
            else:
                costs = [
                    acc_matrix[i-1][j],
                    acc_matrix[i][j-1],
                    acc_matrix[i-1][j-1]
                ]
                moves = [
                    (i-1, j),
                    (i, j-1),
                    (i-1, j-1)
                ]
                min_cost = numpy.argmin(costs)
                #self._log(["Selected min cost move %d", min_cost])
                min_move = moves[min_cost]
                path.append(min_move)
                i, j = min_move
        # reverse path and return
        path.reverse()
        return path



