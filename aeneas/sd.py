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

* :class:`~aeneas.sd.SD`, for detecting the audio head and tail of a given audio file.

.. warning:: This module is likely to be refactored in a future version

.. versionadded:: 1.2.0
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from aeneas.audiofilemfcc import AudioFileMFCC
from aeneas.dtw import DTWAligner
from aeneas.exacttiming import Decimal
from aeneas.exacttiming import InvalidOperation
from aeneas.exacttiming import TimeValue
from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.synthesizer import Synthesizer
import aeneas.globalfunctions as gf


class SD(Loggable):
    """
    The SD ("start detector").

    Given an audio file and a text, detects the audio head and/or tail,
    using a voice activity detector (via :class:`~aeneas.vad.VAD`) and
    performing an alignment with a partial portion of the text
    (via :class:`~aeneas.dtw.DTWAligner`).

    This implementation relies on the following heuristic:

    1. synthesize text until
       ``max_head_length`` times :data:`aeneas.sd.SD.QUERY_FACTOR`
       seconds are reached;
    2. consider only the first
       ``max_head_length`` times :data:`aeneas.sd.SD.AUDIO_FACTOR`
       seconds of the audio file;
    3. compute the best partial alignment of 1. with 2., and return
       the corresponding time value.

    (Similarly for the audio tail.)

    :param real_wave_mfcc: the audio file
    :type  real_wave_mfcc: :class:`~aeneas.audiofile.AudioFileMFCC`
    :param text_file: the text file
    :type  text_file: :class:`~aeneas.textfile.TextFile`
    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    QUERY_FACTOR = Decimal("1.0")
    """
    Multiply the max head/tail length by this factor
    to get the minimum query length to be synthesized.
    Default: ``1.0``.

    .. versionadded:: 1.5.0
    """

    AUDIO_FACTOR = Decimal("2.5")
    """
    Multiply the max head/tail length by this factor
    to get the minimum length in the audio that will be searched
    for.
    Set it to be at least ``1.0 + QUERY_FACTOR * 1.5``.
    Default: ``2.5``.

    .. versionadded:: 1.5.0
    """

    MAX_LENGTH = TimeValue("10.000")
    """
    Try detecting audio head or tail up to this many seconds.
    Default: ``10.000``.

    .. versionadded:: 1.2.0
    """

    MIN_LENGTH = TimeValue("0.000")
    """
    Try detecting audio head or tail of at least this many seconds.
    Default: ``0.000``.

    .. versionadded:: 1.2.0
    """

    TAG = u"SD"

    def __init__(self, real_wave_mfcc, text_file, rconf=None, logger=None):
        super(SD, self).__init__(rconf=rconf, logger=logger)
        self.real_wave_mfcc = real_wave_mfcc
        self.text_file = text_file

    def detect_interval(
            self,
            min_head_length=None,
            max_head_length=None,
            min_tail_length=None,
            max_tail_length=None
    ):
        """
        Detect the interval of the audio file
        containing the fragments in the text file.

        Return the audio interval as a tuple of two
        :class:`~aeneas.exacttiming.TimeValue` objects,
        representing the begin and end time, in seconds,
        with respect to the full wave duration.

        If one of the parameters is ``None``, the default value
        (``0.0`` for min, ``10.0`` for max) will be used.

        :param min_head_length: estimated minimum head length
        :type  min_head_length: :class:`~aeneas.exacttiming.TimeValue`
        :param max_head_length: estimated maximum head length
        :type  max_head_length: :class:`~aeneas.exacttiming.TimeValue`
        :param min_tail_length: estimated minimum tail length
        :type  min_tail_length: :class:`~aeneas.exacttiming.TimeValue`
        :param max_tail_length: estimated maximum tail length
        :type  max_tail_length: :class:`~aeneas.exacttiming.TimeValue`
        :rtype: (:class:`~aeneas.exacttiming.TimeValue`, :class:`~aeneas.exacttiming.TimeValue`)
        :raises: TypeError: if one of the parameters is not ``None`` or a number
        :raises: ValueError: if one of the parameters is negative
        """
        head = self.detect_head(min_head_length, max_head_length)
        tail = self.detect_tail(min_tail_length, max_tail_length)
        begin = head
        end = self.real_wave_mfcc.audio_length - tail
        self.log([u"Audio length: %.3f", self.real_wave_mfcc.audio_length])
        self.log([u"Head length:  %.3f", head])
        self.log([u"Tail length:  %.3f", tail])
        self.log([u"Begin:        %.3f", begin])
        self.log([u"End:          %.3f", end])
        if (begin >= TimeValue("0.000")) and (end > begin):
            self.log([u"Returning %.3f %.3f", begin, end])
            return (begin, end)
        self.log(u"Returning (0.000, 0.000)")
        return (TimeValue("0.000"), TimeValue("0.000"))

    def detect_head(self, min_head_length=None, max_head_length=None):
        """
        Detect the audio head, returning its duration, in seconds.

        :param min_head_length: estimated minimum head length
        :type  min_head_length: :class:`~aeneas.exacttiming.TimeValue`
        :param max_head_length: estimated maximum head length
        :type  max_head_length: :class:`~aeneas.exacttiming.TimeValue`
        :rtype: :class:`~aeneas.exacttiming.TimeValue`
        :raises: TypeError: if one of the parameters is not ``None`` or a number
        :raises: ValueError: if one of the parameters is negative
        """
        return self._detect(min_head_length, max_head_length, tail=False)

    def detect_tail(self, min_tail_length=None, max_tail_length=None):
        """
        Detect the audio tail, returning its duration, in seconds.

        :param min_tail_length: estimated minimum tail length
        :type  min_tail_length: :class:`~aeneas.exacttiming.TimeValue`
        :param max_tail_length: estimated maximum tail length
        :type  max_tail_length: :class:`~aeneas.exacttiming.TimeValue`
        :rtype: :class:`~aeneas.exacttiming.TimeValue`
        :raises: TypeError: if one of the parameters is not ``None`` or a number
        :raises: ValueError: if one of the parameters is negative
        """
        return self._detect(min_tail_length, max_tail_length, tail=True)

    def _detect(self, min_length, max_length, tail=False):
        """
        Detect the head or tail within ``min_length`` and ``max_length`` duration.

        If detecting the tail, the real wave MFCC and the query are reversed
        so that the tail detection problem reduces to a head detection problem.

        Return the duration of the head or tail, in seconds.

        :param min_length: estimated minimum length
        :type  min_length: :class:`~aeneas.exacttiming.TimeValue`
        :param max_length: estimated maximum length
        :type  max_length: :class:`~aeneas.exacttiming.TimeValue`
        :rtype: :class:`~aeneas.exacttiming.TimeValue`
        :raises: TypeError: if one of the parameters is not ``None`` or a number
        :raises: ValueError: if one of the parameters is negative
        """
        def _sanitize(value, default, name):
            if value is None:
                value = default
            try:
                value = TimeValue(value)
            except (TypeError, ValueError, InvalidOperation) as exc:
                self.log_exc(u"The value of %s is not a number" % (name), exc, True, TypeError)
            if value < 0:
                self.log_exc(u"The value of %s is negative" % (name), None, True, ValueError)
            return value

        min_length = _sanitize(min_length, self.MIN_LENGTH, "min_length")
        max_length = _sanitize(max_length, self.MAX_LENGTH, "max_length")
        mws = self.rconf.mws
        min_length_frames = int(min_length / mws)
        max_length_frames = int(max_length / mws)
        self.log([u"MFCC window shift s:     %.3f", mws])
        self.log([u"Min start length s:      %.3f", min_length])
        self.log([u"Min start length frames: %d", min_length_frames])
        self.log([u"Max start length s:      %.3f", max_length])
        self.log([u"Max start length frames: %d", max_length_frames])
        self.log([u"Tail?:                   %s", str(tail)])

        self.log(u"Synthesizing query...")
        synt_duration = max_length * self.QUERY_FACTOR
        self.log([u"Synthesizing at least %.3f seconds", synt_duration])
        tmp_handler, tmp_file_path = gf.tmp_file(suffix=u".wav", root=self.rconf[RuntimeConfiguration.TMP_PATH])
        synt = Synthesizer(rconf=self.rconf, logger=self.logger)
        anchors, total_time, synthesized_chars = synt.synthesize(
            self.text_file,
            tmp_file_path,
            quit_after=synt_duration,
            backwards=tail
        )
        self.log(u"Synthesizing query... done")

        self.log(u"Extracting MFCCs for query...")
        query_mfcc = AudioFileMFCC(tmp_file_path, rconf=self.rconf, logger=self.logger)
        self.log(u"Extracting MFCCs for query... done")

        self.log(u"Cleaning up...")
        gf.delete_file(tmp_handler, tmp_file_path)
        self.log(u"Cleaning up... done")

        search_window = max_length * self.AUDIO_FACTOR
        search_window_end = min(int(search_window / mws), self.real_wave_mfcc.all_length)
        self.log([u"Query MFCC length (frames): %d", query_mfcc.all_length])
        self.log([u"Real MFCC length (frames):  %d", self.real_wave_mfcc.all_length])
        self.log([u"Search window end (s):      %.3f", search_window])
        self.log([u"Search window end (frames): %d", search_window_end])

        if tail:
            self.log(u"Tail => reversing real_wave_mfcc and query_mfcc")
            self.real_wave_mfcc.reverse()
            query_mfcc.reverse()

        # NOTE: VAD will be run here, if not done before
        speech_intervals = self.real_wave_mfcc.intervals(speech=True, time=False)
        if len(speech_intervals) < 1:
            self.log(u"No speech intervals, hence no start found")
            if tail:
                self.real_wave_mfcc.reverse()
            return TimeValue("0.000")

        # generate a list of begin indices
        search_end = None
        candidates_begin = []
        for interval in speech_intervals:
            if (interval[0] >= min_length_frames) and (interval[0] <= max_length_frames):
                candidates_begin.append(interval[0])
            search_end = interval[1]
            if search_end >= search_window_end:
                break

        # for each begin index, compute the acm cost
        # to match the query
        # note that we take the min over the last column of the acm
        # meaning that we allow to match the entire query wave
        # against a portion of the real wave
        candidates = []
        for candidate_begin in candidates_begin:
            self.log([u"Candidate interval starting at %d == %.3f", candidate_begin, candidate_begin * mws])
            try:
                rwm = AudioFileMFCC(
                    mfcc_matrix=self.real_wave_mfcc.all_mfcc[:, candidate_begin:search_end],
                    rconf=self.rconf,
                    logger=self.logger
                )
                dtw = DTWAligner(
                    real_wave_mfcc=rwm,
                    synt_wave_mfcc=query_mfcc,
                    rconf=self.rconf,
                    logger=self.logger
                )
                acm = dtw.compute_accumulated_cost_matrix()
                last_column = acm[:, -1]
                min_value = numpy.min(last_column)
                min_index = numpy.argmin(last_column)
                self.log([u"Candidate interval: %d %d == %.3f %.3f", candidate_begin, search_end, candidate_begin * mws, search_end * mws])
                self.log([u"  Min value: %.6f", min_value])
                self.log([u"  Min index: %d == %.3f", min_index, min_index * mws])
                candidates.append((min_value, candidate_begin, min_index))
            except Exception as exc:
                self.log_exc(u"An unexpected error occurred while running _detect", exc, False, None)

        # reverse again the real wave
        if tail:
            self.log(u"Tail => reversing real_wave_mfcc again")
            self.real_wave_mfcc.reverse()

        # return
        if len(candidates) < 1:
            self.log(u"No candidates found")
            return TimeValue("0.000")
        self.log(u"Candidates:")
        for candidate in candidates:
            self.log([u"  Value: %.6f Begin Time: %.3f Min Index: %d", candidate[0], candidate[1] * mws, candidate[2]])
        best = sorted(candidates)[0][1]
        self.log([u"Best candidate: %d == %.3f", best, best * mws])
        return best * mws
