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

* :class:`~aeneas.audiofilemfcc.AudioFileMFCC`,
  representing a mono WAVE audio file as a matrix of
  Mel-frequency ceptral coefficients (MFCC).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy

from aeneas.audiofile import AudioFile
from aeneas.exacttiming import TimeInterval
from aeneas.exacttiming import TimeValue
from aeneas.logger import Loggable
from aeneas.mfcc import MFCC
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.vad import VAD
import aeneas.globalfunctions as gf


class AudioFileMFCC(Loggable):
    """
    A monoaural (single channel) WAVE audio file,
    represented as a NumPy 2D matrix of
    Mel-frequency ceptral coefficients (MFCC).

    The matrix is "fat", that is,
    its number of rows is equal to the number of MFCC coefficients
    and its number of columns is equal to the number of window shifts
    in the audio file.
    The number of MFCC coefficients and the MFCC window shift can
    be modified via the
    :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.MFCC_SIZE`
    and
    :data:`~aeneas.runtimeconfiguration.RuntimeConfiguration.MFCC_WINDOW_SHIFT`
    keys in the ``rconf`` object.

    If ``mfcc_matrix`` is not ``None``,
    it will be used as the MFCC matrix.

    If ``file_path`` or ``audio_file`` is not ``None``,
    the MFCCs will be computed upon creation of the object,
    possibly converting to PCM16 Mono WAVE and/or
    loading audio data in memory.

    The MFCCs for the entire wave
    are divided into three
    contiguous intervals (possibly, zero-length)::

        HEAD   = [:middle_begin[
        MIDDLE = [middle_begin:middle_end[
        TAIL   = [middle_end:[

    The usual NumPy convention of including the left/start index
    and excluding the right/end index is adopted.

    For alignment purposes, only the ``MIDDLE`` portion of the wave
    is taken into account; the ``HEAD`` and ``TAIL`` intervals are ignored.

    This class heavily uses NumPy views and in-place operations
    to avoid creating temporary data or copying data around.

    :param string file_path: the path of the PCM16 mono WAVE file, or ``None``
    :param tuple file_format: the format of the audio file, if known in advance: ``(codec, channels, rate)`` or ``None``
    :param mfcc_matrix: the MFCC matrix to be set, or ``None``
    :type  mfcc_matrix: :class:`numpy.ndarray`
    :param audio_file: an audio file, or ``None``
    :type  audio_file: :class:`~aeneas.audiofile.AudioFile`
    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    :raises: ValueError: if ``file_path``, ``audio_file``, and ``mfcc_matrix`` are all ``None``

    .. versionadded:: 1.5.0
    """

    TAG = u"AudioFileMFCC"

    def __init__(
            self,
            file_path=None,
            file_format=None,
            mfcc_matrix=None,
            audio_file=None,
            rconf=None,
            logger=None
    ):
        if (file_path is None) and (audio_file is None) and (mfcc_matrix is None):
            raise ValueError(u"You must initialize with at least one of: file_path, audio_file, or mfcc_matrix")
        super(AudioFileMFCC, self).__init__(rconf=rconf, logger=logger)
        self.file_path = file_path
        self.audio_file = audio_file
        self.is_reversed = False
        self.__mfcc = None
        self.__mfcc_mask = None
        self.__mfcc_mask_map = None
        self.__speech_intervals = None
        self.__nonspeech_intervals = None
        self.log(u"Initializing MFCCs...")
        if mfcc_matrix is not None:
            self.__mfcc = mfcc_matrix
            self.audio_length = self.all_length * self.rconf.mws
        elif (self.file_path is not None) or (self.audio_file is not None):
            audio_file_was_none = False
            if self.audio_file is None:
                audio_file_was_none = True
                self.audio_file = AudioFile(
                    file_path=self.file_path,
                    file_format=file_format,
                    rconf=self.rconf,
                    logger=self.logger
                )
                # NOTE load audio samples into memory, if not present already
                self.audio_file.audio_samples
            gf.run_c_extension_with_fallback(
                self.log,
                "cmfcc",
                self._compute_mfcc_c_extension,
                self._compute_mfcc_pure_python,
                (),
                rconf=self.rconf
            )
            self.audio_length = self.audio_file.audio_length
            if audio_file_was_none:
                self.log(u"Clearing the audio data...")
                self.audio_file.clear_data()
                self.audio_file = None
                self.log(u"Clearing the audio data... done")
        self.__middle_begin = 0
        self.__middle_end = self.__mfcc.shape[1]
        self.log(u"Initializing MFCCs... done")

    def __unicode__(self):
        msg = [
            u"File path:        %s" % self.file_path,
            u"Audio length (s): %s" % gf.safe_float(self.audio_length),
        ]
        return u"\n".join(msg)

    def __str__(self):
        return gf.safe_str(self.__unicode__())

    @property
    def all_mfcc(self):
        """
        The MFCCs of the entire audio file,
        that is, HEAD + MIDDLE + TAIL.

        :rtype: :class:`numpy.ndarray` (2D)
        """
        return self.__mfcc

    @property
    def all_length(self):
        """
        The length, in MFCC coefficients,
        of the entire audio file,
        that is, HEAD + MIDDLE + TAIL.

        :rtype: int
        """
        return self.__mfcc.shape[1]

    @property
    def middle_mfcc(self):
        """
        The MFCCs of the middle part of the audio file,
        that is, without HEAD and TAIL.

        :rtype: :class:`numpy.ndarray` (2D)
        """
        return self.__mfcc[:, self.__middle_begin:self.__middle_end]

    @property
    def middle_length(self):
        """
        The length, in MFCC coefficients,
        of the middle part of the audio file,
        that is, without HEAD and TAIL.

        :rtype: int
        """
        return self.__middle_end - self.__middle_begin

    @property
    def middle_map(self):
        """
        Return the map
        from the MFCC frame indices
        in the MIDDLE portion of the wave
        to the MFCC FULL frame indices,
        that is, an ``numpy.arange(self.middle_begin, self.middle_end)``.

        NOTE: to translate indices of MIDDLE,
        instead of using fancy indexing with the
        result of this function, you might want to simply
        add ``self.head_length``.
        This function is provided mostly for consistency
        with the MASKED case.

        :rtype: :class:`numpy.ndarray` (1D)
        """
        return numpy.arange(self.__middle_begin, self.__middle_end)

    @property
    def head_length(self):
        """
        The length, in MFCC coefficients,
        of the HEAD of the audio file.

        :rtype: int
        """
        return self.__middle_begin

    @property
    def tail_length(self):
        """
        The length, in MFCC coefficients,
        of the TAIL of the audio file.

        :rtype: int
        """
        return self.all_length - self.__middle_end

    @property
    def tail_begin(self):
        """
        The index, in MFCC coefficients,
        where the TAIL of the audio file starts.

        :rtype: int
        """
        return self.__middle_end

    @property
    def audio_length(self):
        """
        The length, in seconds, of the audio file.

        This value is the actual length of the audio file,
        computed as ``number of samples / sample_rate``,
        hence it might differ than ``len(self.__mfcc) * mfcc_window_shift``.

        :rtype: :class:`~aeneas.exacttiming.TimeValue`
        """
        return self.__audio_length

    @audio_length.setter
    def audio_length(self, audio_length):
        self.__audio_length = audio_length

    @property
    def is_reversed(self):
        """
        Return ``True`` if currently reversed.

        :rtype: bool
        """
        return self.__is_reversed

    @is_reversed.setter
    def is_reversed(self, is_reversed):
        self.__is_reversed = is_reversed

    @property
    def masked_mfcc(self):
        """
        Return the MFCC speech frames
        in the FULL wave.

        :rtype: :class:`numpy.ndarray` (2D)
        """
        self._ensure_mfcc_mask()
        return self.__mfcc[:, self.__mfcc_mask]

    @property
    def masked_length(self):
        """
        Return the number of MFCC speech frames
        in the FULL wave.

        :rtype: int
        """
        self._ensure_mfcc_mask()
        return len(self.__mfcc_mask_map)

    @property
    def masked_map(self):
        """
        Return the map
        from the MFCC speech frame indices
        to the MFCC FULL frame indices.

        :rtype: :class:`numpy.ndarray` (1D)
        """
        self._ensure_mfcc_mask()
        return self.__mfcc_mask_map

    @property
    def masked_middle_mfcc(self):
        """
        Return the MFCC speech frames
        in the MIDDLE portion of the wave.

        :rtype: :class:`numpy.ndarray` (2D)
        """
        begin, end = self._masked_middle_begin_end()
        return (self.masked_mfcc)[:, begin:end]

    @property
    def masked_middle_length(self):
        """
        Return the number of MFCC speech frames
        in the MIDDLE portion of the wave.

        :rtype: int
        """
        begin, end = self._masked_middle_begin_end()
        return end - begin

    @property
    def masked_middle_map(self):
        """
        Return the map
        from the MFCC speech frame indices
        in the MIDDLE portion of the wave
        to the MFCC FULL frame indices.

        :rtype: :class:`numpy.ndarray` (1D)
        """
        begin, end = self._masked_middle_begin_end()
        return self.__mfcc_mask_map[begin:end]

    def _masked_middle_begin_end(self):
        """
        Return the begin and end indices w.r.t. ``self.__mfcc_mask_map``,
        corresponding to indices in the MIDDLE portion of the wave,
        that is, which fall between ``self.__middle_begin`` and
        ``self.__middle_end`` in ``self.__mfcc``.

        :rtype: (int, int)
        """
        self._ensure_mfcc_mask()
        begin = numpy.searchsorted(self.__mfcc_mask_map, self.__middle_begin, side="left")
        end = numpy.searchsorted(self.__mfcc_mask_map, self.__middle_end, side="right")
        return (begin, end)

    def intervals(self, speech=True, time=True):
        """
        Return a list of intervals::

        [(b_1, e_1), (b_2, e_2), ..., (b_k, e_k)]

        where ``b_i`` is the time when the ``i``-th interval begins,
        and ``e_i`` is the time when it ends.

        :param bool speech: if ``True``, return speech intervals,
                            otherwise return nonspeech intervals
        :param bool time: if ``True``, return :class:`~aeneas.exacttiming.TimeInterval` objects,
                          otherwise return indices (int)
        :rtype: list of pairs (see above)
        """
        self._ensure_mfcc_mask()
        if speech:
            self.log(u"Converting speech runs to intervals...")
            intervals = self.__speech_intervals
        else:
            self.log(u"Converting nonspeech runs to intervals...")
            intervals = self.__nonspeech_intervals
        if time:
            mws = self.rconf.mws
            intervals = [TimeInterval(
                begin=(b * mws),
                end=((e + 1) * mws)
            ) for b, e in intervals]
        self.log(u"Converting... done")
        return intervals

    def inside_nonspeech(self, index):
        """
        If ``index`` is contained in a nonspeech interval,
        return a pair ``(interval_begin, interval_end)``
        such that ``interval_begin <= index < interval_end``,
        i.e., ``interval_end`` is assumed not to be included.

        Otherwise, return ``None``.

        :rtype: ``None`` or tuple
        """
        self._ensure_mfcc_mask()
        if (index < 0) or (index >= self.all_length) or (self.__mfcc_mask[index]):
            return None
        return self._binary_search_intervals(self.__nonspeech_intervals, index)

    @classmethod
    def _binary_search_intervals(cls, intervals, index):
        """
        Binary search for the interval containing index,
        assuming there is such an interval.
        This function should never return ``None``.
        """
        start = 0
        end = len(intervals) - 1
        while start <= end:
            middle_index = start + ((end - start) // 2)
            middle = intervals[middle_index]
            if (middle[0] <= index) and (index < middle[1]):
                return middle
            elif middle[0] > index:
                end = middle_index - 1
            else:
                start = middle_index + 1
        return None

    @property
    def middle_begin(self):
        """
        Return the index where MIDDLE starts.

        :rtype: int
        """
        return self.__middle_begin

    @middle_begin.setter
    def middle_begin(self, index):
        """
        Set the index where MIDDLE starts.

        :param int index: the new index for MIDDLE begin
        """
        if (index < 0) or (index > self.all_length):
            raise ValueError(u"The given index is not valid")
        self.__middle_begin = index

    @property
    def middle_begin_seconds(self):
        """
        Return the time instant, in seconds, where MIDDLE starts.

        :rtype: :class:`~aeneas.exacttiming.TimeValue`
        """
        return TimeValue(self.__middle_begin) * self.rconf.mws

    @property
    def middle_end(self):
        """
        Return the index (+1) where MIDDLE ends.

        :rtype: int
        """
        return self.__middle_end

    @middle_end.setter
    def middle_end(self, index):
        """
        Set the index (+1) where MIDDLE ends.

        :param int index: the new index for MIDDLE end
        """
        if (index < 0) or (index > self.all_length):
            raise ValueError(u"The given index is not valid")
        self.__middle_end = index

    @property
    def middle_end_seconds(self):
        """
        Return the time instant, in seconds, where MIDDLE ends.

        :rtype: :class:`~aeneas.exacttiming.TimeValue`
        """
        return TimeValue(self.__middle_end) * self.rconf.mws

    def _ensure_mfcc_mask(self):
        """
        Ensure that ``run_vad()`` has already been called,
        and hence ``self.__mfcc_mask`` has a meaningful value.
        """
        if self.__mfcc_mask is None:
            self.log(u"VAD was not run: running it now")
            self.run_vad()

    def _compute_mfcc_c_extension(self):
        """
        Compute MFCCs using the Python C extension cmfcc.
        """
        self.log(u"Computing MFCCs using C extension...")
        try:
            self.log(u"Importing cmfcc...")
            import aeneas.cmfcc.cmfcc
            self.log(u"Importing cmfcc... done")
            self.__mfcc = (aeneas.cmfcc.cmfcc.compute_from_data(
                self.audio_file.audio_samples,
                self.audio_file.audio_sample_rate,
                self.rconf[RuntimeConfiguration.MFCC_FILTERS],
                self.rconf[RuntimeConfiguration.MFCC_SIZE],
                self.rconf[RuntimeConfiguration.MFCC_FFT_ORDER],
                self.rconf[RuntimeConfiguration.MFCC_LOWER_FREQUENCY],
                self.rconf[RuntimeConfiguration.MFCC_UPPER_FREQUENCY],
                self.rconf[RuntimeConfiguration.MFCC_EMPHASIS_FACTOR],
                self.rconf[RuntimeConfiguration.MFCC_WINDOW_LENGTH],
                self.rconf[RuntimeConfiguration.MFCC_WINDOW_SHIFT]
            )[0]).transpose()
            self.log(u"Computing MFCCs using C extension... done")
            return (True, None)
        except Exception as exc:
            self.log_exc(u"An unexpected error occurred while running cmfcc", exc, False, None)
        return (False, None)

    def _compute_mfcc_pure_python(self):
        """
        Compute MFCCs using the pure Python code.
        """
        self.log(u"Computing MFCCs using pure Python code...")
        try:
            self.__mfcc = MFCC(
                rconf=self.rconf,
                logger=self.logger
            ).compute_from_data(
                self.audio_file.audio_samples,
                self.audio_file.audio_sample_rate
            ).transpose()
            self.log(u"Computing MFCCs using pure Python code... done")
            return (True, None)
        except Exception as exc:
            self.log_exc(u"An unexpected error occurred while running pure Python code", exc, False, None)
        return (False, None)

    def reverse(self):
        """
        Reverse the audio file.

        The reversing is done efficiently using NumPy views inplace
        instead of swapping values.

        Only speech and nonspeech intervals are actually recomputed
        as Python lists.
        """
        self.log(u"Reversing...")
        all_length = self.all_length
        self.__mfcc = self.__mfcc[:, ::-1]
        tmp = self.__middle_end
        self.__middle_end = all_length - self.__middle_begin
        self.__middle_begin = all_length - tmp
        if self.__mfcc_mask is not None:
            self.__mfcc_mask = self.__mfcc_mask[::-1]
            # equivalent to
            # self.__mfcc_mask_map = ((all_length - 1) - self.__mfcc_mask_map)[::-1]
            # but done in place using NumPy view
            self.__mfcc_mask_map *= -1
            self.__mfcc_mask_map += all_length - 1
            self.__mfcc_mask_map = self.__mfcc_mask_map[::-1]
            self.__speech_intervals = [(all_length - i[1], all_length - i[0]) for i in self.__speech_intervals[::-1]]
            self.__nonspeech_intervals = [(all_length - i[1], all_length - i[0]) for i in self.__nonspeech_intervals[::-1]]
        self.is_reversed = not self.is_reversed
        self.log(u"Reversing...done")

    def run_vad(
        self,
        log_energy_threshold=None,
        min_nonspeech_length=None,
        extend_before=None,
        extend_after=None
    ):
        """
        Determine which frames contain speech and nonspeech,
        and store the resulting boolean mask internally.

        The four parameters might be ``None``:
        in this case, the corresponding RuntimeConfiguration values
        are applied.

        :param float log_energy_threshold: the minimum log energy threshold to consider a frame as speech
        :param int min_nonspeech_length: the minimum length, in frames, of a nonspeech interval
        :param int extend_before: extend each speech interval by this number of frames to the left (before)
        :param int extend_after: extend each speech interval by this number of frames to the right (after)
        """
        def _compute_runs(array):
            """
            Compute runs as a list of arrays,
            each containing the indices of a contiguous run.

            :param array: the data array
            :type  array: :class:`numpy.ndarray` (1D)
            :rtype: list of :class:`numpy.ndarray` (1D)
            """
            if len(array) < 1:
                return []
            return numpy.split(array, numpy.where(numpy.diff(array) != 1)[0] + 1)
        self.log(u"Creating VAD object")
        vad = VAD(rconf=self.rconf, logger=self.logger)
        self.log(u"Running VAD...")
        self.__mfcc_mask = vad.run_vad(
            wave_energy=self.__mfcc[0],
            log_energy_threshold=log_energy_threshold,
            min_nonspeech_length=min_nonspeech_length,
            extend_before=extend_before,
            extend_after=extend_after
        )
        self.__mfcc_mask_map = (numpy.where(self.__mfcc_mask))[0]
        self.log(u"Running VAD... done")
        self.log(u"Storing speech and nonspeech intervals...")
        # where( == True) already computed, reusing
        # COMMENTED runs = _compute_runs((numpy.where(self.__mfcc_mask))[0])
        runs = _compute_runs(self.__mfcc_mask_map)
        self.__speech_intervals = [(r[0], r[-1]) for r in runs]
        # where( == False) not already computed, computing now
        runs = _compute_runs((numpy.where(~self.__mfcc_mask))[0])
        self.__nonspeech_intervals = [(r[0], r[-1]) for r in runs]
        self.log(u"Storing speech and nonspeech intervals... done")

    def set_head_middle_tail(self, head_length=None, middle_length=None, tail_length=None):
        """
        Set the HEAD, MIDDLE, TAIL explicitly.

        If a parameter is ``None``, it will be ignored.
        If both ``middle_length`` and ``tail_length`` are specified,
        only ``middle_length`` will be applied.

        :param head_length: the length of HEAD, in seconds
        :type  head_length: :class:`~aeneas.exacttiming.TimeValue`
        :param middle_length: the length of MIDDLE, in seconds
        :type  middle_length: :class:`~aeneas.exacttiming.TimeValue`
        :param tail_length: the length of TAIL, in seconds
        :type  tail_length: :class:`~aeneas.exacttiming.TimeValue`
        :raises: TypeError: if one of the arguments is not ``None``
                            or :class:`~aeneas.exacttiming.TimeValue`
        :raises: ValueError: if one of the arguments is greater
                             than the length of the audio file
        """
        for variable, name in [
            (head_length, "head_length"),
            (middle_length, "middle_length"),
            (tail_length, "tail_length")
        ]:
            if (variable is not None) and (not isinstance(variable, TimeValue)):
                raise TypeError(u"%s is not None or TimeValue" % name)
            if (variable is not None) and (variable > self.audio_length):
                raise ValueError(u"%s is greater than the length of the audio file" % name)
        self.log(u"Setting head middle tail...")
        mws = self.rconf.mws
        self.log([u"Before: 0 %d %d %d", self.middle_begin, self.middle_end, self.all_length])
        if head_length is not None:
            self.middle_begin = int(head_length / mws)
        if middle_length is not None:
            self.middle_end = self.middle_begin + int(middle_length / mws)
        elif tail_length is not None:
            self.middle_end = self.all_length - int(tail_length / mws)
        self.log([u"After:  0 %d %d %d", self.middle_begin, self.middle_end, self.all_length])
        self.log(u"Setting head middle tail... done")
