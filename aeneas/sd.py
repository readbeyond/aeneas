#!/usr/bin/env python
# coding=utf-8

"""
This module contains the implementation
of a simple Start Detector (SD),
based on VAD and iterated DTW.

Given a (full) audio file and the corresponding (full) text,
it will compute the time interval
containing the given text,
that is, detect the audio head and the audio length.

.. versionadded:: 1.2.0
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy

from aeneas.audiofile import AudioFileMonoWAVE
from aeneas.audiofilemfcc import AudioFileMFCC
from aeneas.dtw import DTWAligner
from aeneas.logger import Logger
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.synthesizer import Synthesizer
from aeneas.vad import VAD
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.5.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class SD(object):
    """
    The SD extractor.

    :param real_wave_mfcc: the audio file
    :type  real_wave_mfcc: :class:`aeneas.audiofile.AudioFileMFCC`
    :param text_file: the text file
    :type  text_file: :class:`aeneas.textfile.TextFile`
    :param rconf: a runtime configuration. Default: ``None``, meaning that
                  default settings will be used.
    :type  rconf: :class:`aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = u"SD"

    QUERY_FACTOR = 1.0
    """
    Multiply the max head/tail length by this factor
    to get the minimum query length to be synthesized.
    Default: ``1.0``.

    .. versionadded:: 1.5.0
    """

    AUDIO_FACTOR = 2.5
    """
    Multiply the max head/tail length by this factor
    to get the minimum length in the audio that will be searched
    for.
    Set it to be at least ``1.0 + QUERY_FACTOR * 1.5``.
    Default: ``2.5``.

    .. versionadded:: 1.5.0
    """

    MAX_LENGTH = 10.0
    """
    Try detecting audio head or tail up to this many seconds.
    Default: ``10.0``.

    .. versionadded:: 1.2.0
    """

    MIN_LENGTH = 0.0
    """
    Try detecting audio head or tail of at least this many seconds.
    Default: ``0.0``.

    .. versionadded:: 1.2.0
    """

    def __init__(self, real_wave_mfcc, text_file, rconf=None, logger=None):
        self.logger = logger if logger is not None else Logger()
        self.rconf = rconf if rconf is not None else RuntimeConfiguration()
        self.real_wave_mfcc = real_wave_mfcc
        self.text_file = text_file

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

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

        Return the audio interval as a tuple of two float values,
        representing the begin and end time, in seconds,
        with respect to the full wave duration.

        If one of the parameters is ``None``, the default value
        (``0.0`` for min, ``10.0`` for max) will be used.

        :param float min_head_length: estimated minimum head length
        :param float max_head_length: estimated maximum head length
        :param float min_tail_length: estimated minimum tail length
        :param float max_tail_length: estimated maximum tail length
        :rtype: (float, float)
        :raises TypeError: if one of the parameters is not ``None`` or a number
        :raises ValueError: if one of the parameters is negative
        """
        head = self.detect_head(min_head_length, max_head_length)
        tail = self.detect_tail(min_tail_length, max_tail_length)
        begin = head
        end = self.real_wave_mfcc.audio_length - tail
        self._log([u"Audio length: %.3f", self.real_wave_mfcc.audio_length])
        self._log([u"Head length:  %.3f", head])
        self._log([u"Tail length:  %.3f", tail])
        self._log([u"Begin:        %.3f", begin])
        self._log([u"End:          %.3f", end])
        if (begin >= 0.0) and (end > begin):
            self._log([u"Returning %.3f %.3f", begin, end])
            return (begin, end)
        self._log(u"Returning (0.0, 0.0)")
        return (0.0, 0.0)

    def detect_head(self, min_head_length=None, max_head_length=None):
        """
        Detect the audio head.

        :param float min_head_length: estimated minimum head length
        :param float max_head_length: estimated maximum head length
        :rtype: float
        :raises TypeError: if one of the parameters is not ``None`` or a number
        :raises ValueError: if one of the parameters is negative
        """
        return self._detect(min_head_length, max_head_length, False)

    def detect_tail(self, min_tail_length=None, max_tail_length=None):
        """
        Detect the audio tail.

        :param float min_tail_length: estimated minimum tail length
        :param float max_tail_length: estimated maximum tail length
        :rtype: float
        :raises TypeError: if one of the parameters is not ``None`` or a number
        :raises ValueError: if one of the parameters is negative
        """
        return self._detect(min_tail_length, max_tail_length, True)

    def _detect(self, min_length, max_length, tail=False):
        """
        Detect the head or tail within ``min_length`` and ``max_length`` duration.

        If detecting the tail, the real wave MFCC and the query are reversed
        so that the tail detection problem reduces to a head detection problem.

        Return the duration of the head or tail, in seconds.

        :rtype: float
        :raises TypeError: if one of the parameters is not ``None`` or a number
        :raises ValueError: if one of the parameters is negative
        """
        def _sanitize(value, default, name):
            if value is None:
                value = default
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise TypeError(u"The value of %s is not a number" % name)
            if value < 0:
                raise ValueError(u"The value of %s is negative" % name)
            return value
        
        min_length = _sanitize(min_length, self.MIN_LENGTH, "min_length")
        max_length = _sanitize(max_length, self.MAX_LENGTH, "max_length")
        mws = self.rconf["mfcc_window_shift"]
        min_length_frames = int(min_length / mws)
        max_length_frames = int(max_length / mws)
        self._log([u"MFCC window shift s:     %.3f", mws])
        self._log([u"Min start length s:      %.3f", min_length])
        self._log([u"Min start length frames: %d", min_length_frames])
        self._log([u"Max start length s:      %.3f", max_length])
        self._log([u"Max start length frames: %d", max_length_frames])
        self._log([u"Tail?:                   %s", str(tail)])

        self._log(u"Synthesizing query...")
        synt_duration = max_length * self.QUERY_FACTOR
        self._log([u"Synthesizing at least %.3f seconds", synt_duration])
        tmp_handler, tmp_file_path = gf.tmp_file(suffix=u".wav", root=self.rconf["tmp_path"])
        synt = Synthesizer(rconf=self.rconf, logger=self.logger)
        anchors, total_time, synthesized_chars = synt.synthesize(
            self.text_file,
            tmp_file_path,
            quit_after=synt_duration,
            backwards=tail
        )
        self._log(u"Synthesizing query... done")

        self._log(u"Extracting MFCCs for query...")
        query_mfcc = AudioFileMFCC(tmp_file_path, rconf=self.rconf, logger=self.logger)
        self._log(u"Extracting MFCCs for query... done")
        
        self._log(u"Cleaning up...")
        gf.delete_file(tmp_handler, tmp_file_path)
        self._log(u"Cleaning up... done")

        search_window = max_length * self.AUDIO_FACTOR
        search_window_end = min(int(search_window / mws), self.real_wave_mfcc.all_length)
        self._log([u"Query MFCC length (frames): %d", query_mfcc.all_length])
        self._log([u"Real MFCC length (frames):  %d", self.real_wave_mfcc.all_length])
        self._log([u"Search window end (s):      %.3f", search_window])
        self._log([u"Search window end (frames): %d", search_window_end])

        if tail:
            self._log(u"Tail => reversing real_wave_mfcc and query_mfcc")
            self.real_wave_mfcc.reverse()
            query_mfcc.reverse()

        # NOTE: VAD will be run here, if not done before
        speech_intervals = self.real_wave_mfcc.intervals(speech=True, time=False)
        if len(speech_intervals) < 1:
            self._log(u"No speech intervals, hence no start found")
            if tail:
                self.real_wave_mfcc.reverse()
            return 0.0

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
            self._log([u"Candidate interval starting at %d == %.3f", candidate_begin, candidate_begin * mws])
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
                self._log([u"Candidate interval: %d %d == %.3f %.3f", candidate_begin, search_end, candidate_begin * mws, search_end * mws])
                self._log([u"  Min value: %.6f", min_value])
                self._log([u"  Min index: %d == %.3f", min_index, min_index * mws])
                candidates.append((min_value, candidate_begin, min_index))
            except Exception as exc:
                self._log([u"Exception %s", str(exc)], Logger.WARNING)

        # reverse again the real wave
        if tail:
            self._log(u"Tail => reversing real_wave_mfcc again")
            self.real_wave_mfcc.reverse()
        
        # return
        if len(candidates) < 1:
            self._log(u"No candidates found")
            return 0.0
        self._log(u"Candidates:")
        for candidate in candidates:
            self._log([u"  Value: %.6f Begin Time: %.3f Min Index: %d", candidate[0], candidate[1] * mws, candidate[2]]) 
        best = sorted(candidates)[0][1]
        self._log([u"Best candidate: %d == %.3f", best, best * mws])
        return best * mws



