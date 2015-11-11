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

import numpy
import os
import tempfile

from aeneas.audiofile import AudioFileMonoWAV
from aeneas.dtw import DTWAligner
from aeneas.logger import Logger
from aeneas.synthesizer import Synthesizer
from aeneas.vad import VAD
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

class SDMetric(object):
    """
    Enumeration of the SD metrics that can be used
    for selecting the best SD candidate.
    """

    VALUE = "value"
    """ Use the value from the accumulated cost matrix """

    DISTORTION = "distortion"
    """ Use the distortion (value / length) """


class SD(object):
    """
    The SD extractor.

    :param audio_file: the audio file
    :type  audio_file: :class:`aeneas.audiofile.AudioFileMonoWAV`
    :param text_file: the text file
    :type  text_file: :class:`aeneas.textfile.TextFile`
    :param frame_rate: the MFCC frame rate, in frames per second. Default:
                       :class:`aeneas.globalconstants.MFCC_FRAME_RATE`
    :type  frame_rate: int
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = "SD"

    # TODO eliminate these magic numbers
    MAX_RUNS_NO_IMPROVEMENT = 20
    MAX_RUNS_WITH_MIN_LENGTH = 20
    QUERY_FACTOR = 2.0
    AUDIO_FACTOR = 6.0

    def __init__(
            self,
            audio_file,
            text_file,
            frame_rate=gc.MFCC_FRAME_RATE,
            logger=None
        ):
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()
        self.audio_file = audio_file
        self.text_file = text_file
        self.frame_rate = frame_rate
        self.audio_speech = None

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def detect_interval(
            self,
            min_head_length=gc.SD_MIN_HEAD_LENGTH,
            max_head_length=gc.SD_MAX_HEAD_LENGTH,
            min_tail_length=gc.SD_MIN_TAIL_LENGTH,
            max_tail_length=gc.SD_MAX_TAIL_LENGTH,
            metric=SDMetric.VALUE
        ):
        """
        Detect the audio interval.

        :param max_head_length: estimated maximum head length
        :type  max_head_length: float
        :param max_tail_length: estimated maximum tail length
        :type  max_tail_length: float
        :param metric: the metric to be used when comparing candidates
        :type  metric: :class:`aeneas.sd.SDMetric`
        :rtype: (float, float)
        """
        head = self.detect_head(min_head_length, max_head_length, metric)
        tail = self.detect_tail(min_tail_length, max_tail_length, metric)
        begin = head
        end = self.audio_file.audio_length - tail
        self._log(["Audio length: %.3f", self.audio_file.audio_length])
        self._log(["Head length:  %.3f", head])
        self._log(["Tail length:  %.3f", tail])
        self._log(["Begin:        %.3f", begin])
        self._log(["End:          %.3f", end])
        if (begin >= 0) and (end > begin):
            self._log(["Returning %.3f %.3f", begin, end])
            return (begin, end)
        self._log("Returning (0.0, 0.0)")
        return (0.0, 0.0)

    def detect_head(
            self,
            min_head_length=gc.SD_MIN_HEAD_LENGTH,
            max_head_length=gc.SD_MAX_HEAD_LENGTH,
            metric=SDMetric.VALUE
        ):
        """
        Detect the audio head.

        :param min_head_length: estimated minimum head length
        :type  min_head_length: float
        :param max_head_length: estimated maximum head length
        :type  max_head_length: float
        :param metric: the metric to be used when comparing candidates
        :type  metric: :class:`aeneas.sd.SDMetric`
        :rtype: float
        """
        self._extract_mfcc()
        self._extract_speech()
        self.audio_file.clear_data()
        head = 0.0
        try:
            head = self._detect_start(min_head_length, max_head_length, metric, False)
        except Exception as exc:
            self._log("Error while detecting head", Logger.CRITICAL)
            self._log(["  Message: %s", exc], Logger.CRITICAL)
        return head

    def detect_tail(
            self,
            min_tail_length=gc.SD_MIN_TAIL_LENGTH,
            max_tail_length=gc.SD_MAX_TAIL_LENGTH,
            metric=SDMetric.VALUE
        ):
        """
        Detect the audio tail.

        :param min_tail_length: estimated minimum tail length
        :type  min_tail_length: float
        :param max_tail_length: estimated maximum tail length
        :type  max_tail_length: float
        :param metric: the metric to be used when comparing candidates
        :type  metric: :class:`aeneas.sd.SDMetric`
        :rtype: float
        """
        self.audio_file.reverse()
        self._extract_mfcc()
        self._extract_speech()
        self._log("Reversing audio")
        self.audio_file.reverse()
        self.audio_file.clear_data()
        tail = 0.0
        try:
            tail = self._detect_start(min_tail_length, max_tail_length, metric, True)
        except Exception as exc:
            self._log("Error while detecting tail", Logger.CRITICAL)
            self._log(["  Message: %s", exc], Logger.CRITICAL)
        return tail

    def _detect_start(self, min_start_length, max_start_length, metric, backwards=False):
        """ Detect start """

        self._log(["Min start length: %.3f", min_start_length])
        self._log(["Max start length: %.3f", max_start_length])
        self._log(["Metric:           %s", metric])
        self._log(["Backwards:        %s", str(backwards)])

        audio_rate = self.text_file.characters / self.audio_file.audio_length
        self._log(["Audio rate:     %.3f", audio_rate])

        self._log("Synthesizing query...")
        tmp_handler, tmp_file_path = tempfile.mkstemp(
            suffix=".wav",
            dir=gf.custom_tmp_dir()
        )
        synt = Synthesizer(logger=self.logger)
        synt_duration = max_start_length * self.QUERY_FACTOR
        self._log(["Synthesizing %.3f seconds", synt_duration])
        # force_pure_python=True because Python C does not prepend data!!!
        result = synt.synthesize(
            self.text_file,
            tmp_file_path,
            quit_after=synt_duration,
            backwards=backwards,
            force_pure_python=True
        )
        self._log("Synthesizing query... done")

        query_file = AudioFileMonoWAV(tmp_file_path)
        if backwards:
            self._log("Reversing query")
            query_file.reverse()
        self._log("Extracting MFCCs for query...")
        query_file.extract_mfcc(frame_rate=self.frame_rate)
        query_file.clear_data()
        self._log("Extracting MFCCs for query... done")

        self._log("Cleaning up...")
        gf.delete_file(tmp_handler, tmp_file_path)
        self._log("Cleaning up... done")

        query_characters = result[2]
        query_len = query_file.audio_length
        query_mfcc = query_file.audio_mfcc
        query_rate = query_characters / query_len

        stretch_factor = max(1, query_rate / audio_rate)
        self._log(["Audio rate:     %.3f", audio_rate])
        self._log(["Query rate:     %.3f", query_rate])
        self._log(["Stretch factor: %.3f", stretch_factor])

        audio_mfcc = self.audio_file.audio_mfcc
        self._log(["Actual audio has %d frames", audio_mfcc.shape[1]])
        audio_mfcc_end_index = int(max_start_length * self.AUDIO_FACTOR * self.frame_rate)
        self._log(["Limiting audio to first %d frames", audio_mfcc_end_index])
        audio_mfcc_end_index = min(audio_mfcc_end_index, audio_mfcc.shape[1])
        audio_mfcc = audio_mfcc[:, 0:audio_mfcc_end_index]
        self._log(["Limited audio has %d frames", audio_mfcc.shape[1]])

        l, o = audio_mfcc.shape
        l, n = query_mfcc.shape

        # minimum length of a matched interval in the real audio
        stretched_match_minimum_length = int(n * stretch_factor)

        self._log(["Audio has %d frames == %.3f seconds", o, self._i2t(o)])
        self._log(["Query has %d frames == %.3f seconds", n, self._i2t(n)])
        self._log(["Stretch factor:          %.3f", stretch_factor])
        self._log(["Required minimum length: %.3f", stretched_match_minimum_length])
        self._log("Speech intervals:")
        for interval in self.audio_speech:
            self._log(["  %d %d == %.3f %.3f", self._t2i(interval[0]), self._t2i(interval[1]), interval[0], interval[1]])

        admissible_intervals = [x for x in self.audio_speech if ((x[0] >= min_start_length) and (x[0] <= max_start_length))]
        self._log("AdmissibleSpeech intervals:")
        for interval in admissible_intervals:
            self._log(["  %d %d == %.3f %.3f", self._t2i(interval[0]), self._t2i(interval[1]), interval[0], interval[1]])

        candidates = []
        runs_with_min_length = 0
        runs_no_improvement = 0
        runs_min_distortion = numpy.inf
        runs_min_value = numpy.inf

        for interval in admissible_intervals:
            if runs_no_improvement >= self.MAX_RUNS_NO_IMPROVEMENT:
                self._log("  Breaking: too many runs without improvement")
                break

            if runs_with_min_length >= self.MAX_RUNS_WITH_MIN_LENGTH:
                self._log("  Breaking: too many runs with minimum required length")
                break

            start_time = interval[0]
            start_index = self._t2i(start_time)
            self._log(["Evaluating interval starting at %d == %.3f ", start_index, start_time])
            if start_index > o:
                self._log("  Breaking: start index outside audio window")
                break

            req_end_index = start_index + stretched_match_minimum_length
            req_end_time = self._i2t(req_end_index)
            if req_end_index > o:
                self._log("  Breaking: not enough audio left in shifted window")
                break
            end_index = min(start_index + 2 * n, o)
            end_time = self._i2t(end_index)

            self._log(["  Start   %d == %.3f", start_index, start_time])
            self._log(["  Req end %d == %.3f", req_end_index, req_end_time])
            self._log(["  Eff end %d == %.3f", end_index, end_time])

            audio_mfcc_sub = audio_mfcc[:, start_index:end_index]
            l, m = audio_mfcc_sub.shape

            self._log("Computing DTW...")
            aligner = DTWAligner(None, None, frame_rate=self.frame_rate, logger=self.logger)
            aligner.real_wave_full_mfcc = audio_mfcc_sub
            aligner.synt_wave_full_mfcc = query_mfcc
            aligner.real_wave_length = self._i2t(m)
            aligner.synt_wave_length = self._i2t(n)
            acm = aligner.compute_accumulated_cost_matrix()
            # transpose, so we have an n x m accumulated cost matrix
            acm = acm.transpose()
            last_row = acm[-1, :]
            self._log("Computing DTW... done")

            # find the minimum, but its index must be >= stretched_match_minimum_length
            candidate_argmin_index = numpy.argmin(last_row[stretched_match_minimum_length:])
            candidate_length_index = stretched_match_minimum_length + candidate_argmin_index
            candidate_length_time = self._i2t(candidate_length_index)
            candidate_value = last_row[candidate_length_index]
            candidate_end_index = start_index + candidate_length_index
            candidate_end_time = self._i2t(candidate_end_index)
            candidate_distortion = candidate_value / candidate_length_index

            # check if the candidate has minimum length
            if candidate_length_index == stretched_match_minimum_length:
                runs_with_min_length += 1
            else:
                runs_with_min_length = 0

            # check if the candidate improved the global minimum value
            if metric == SDMetric.VALUE:
                if candidate_value < runs_min_value:
                    runs_min_value = candidate_value
                    runs_no_improvement = 0
                else:
                    runs_no_improvement += 1
            if metric == SDMetric.DISTORTION:
                if candidate_distortion < runs_min_distortion:
                    runs_min_distortion = candidate_distortion
                    runs_no_improvement = 0
                else:
                    runs_no_improvement += 1

            # append to the list of candidates
            self._log(["    Interval  start:      %d == %.6f", start_index, start_time])
            self._log(["    Interval  end:        %d == %.6f", end_index, end_time])
            self._log(["    Candidate start:      %d == %.6f", start_index, start_time])
            self._log(["    Candidate end:        %d == %.6f", candidate_end_index, candidate_end_time])
            self._log(["    Candidate length:     %d == %.6f", candidate_length_index, candidate_length_time])
            self._log(["    Candidate value:      %.6f", candidate_value])
            self._log(["    Candidate distortion: %.6f", candidate_distortion])
            candidates.append({
                "start_index": start_index,
                "length": candidate_length_index,
                "value": candidate_value,
                "distortion": candidate_distortion
            })

        # select best candidate and return its start time
        # if we have no best candidate, return 0.0
        best_candidate = self._select_best_candidate(candidates, metric)
        if best_candidate is None:
            return 0.0
        sd_time = self._i2t(max(best_candidate["start_index"], 0))
        self._log(["Returning time %.3f", sd_time])
        return sd_time

    def _extract_mfcc(self):
        """ Extract MFCCs for audio """
        self._log("Extracting MFCCs for audio...")
        self.audio_file.extract_mfcc(frame_rate=self.frame_rate)
        self._log("Extracting MFCCs for audio... done")

    def _extract_speech(self):
        """ Extract speech intervals """
        self._log("Running VAD...")
        vad = VAD(
            self.audio_file.audio_mfcc,
            self.audio_file.audio_length,
            frame_rate=self.frame_rate,
            logger=self.logger
        )
        vad.compute_vad()
        self.audio_speech = vad.speech
        self._log("Running VAD... done")

    def _i2t(self, index):
        """ Frame index to (start) time """
        return index * 1.0 / self.frame_rate

    def _t2i(self, time):
        """ Frame (start) time to index """
        return int(time * self.frame_rate)

    def _select_best_candidate(self, candidates, metric):
        """ Select the best candidate (or None if no one is found) """
        self._log(["Using metric '%s'", metric])
        self._log("Candidates:")
        for candidate in candidates:
            self._log(["  %s", str(candidate)])
        tuples = []
        if metric == SDMetric.VALUE:
            tuples = [(v["value"], v["distortion"], v) for v in candidates]
        if metric == SDMetric.DISTORTION:
            tuples = [(v["distortion"], v["value"], v) for v in candidates]
        if len(tuples) == 0:
            return None
        return min(tuples)[2]



