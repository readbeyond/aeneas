#!/usr/bin/env python
# coding=utf-8

"""
Execute a task, that is, compute the sync map for it.
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy

from aeneas.adjustboundaryalgorithm import AdjustBoundaryAlgorithm
from aeneas.audiofile import AudioFileMonoWAVE
from aeneas.audiofilemfcc import AudioFileMFCC
from aeneas.dtw import DTWAligner
from aeneas.ffmpegwrapper import FFMPEGWrapper
from aeneas.logger import Logger
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.sd import SD
from aeneas.syncmap import SyncMap
from aeneas.syncmap import SyncMapFragment
from aeneas.syncmap import SyncMapHeadTailFormat
from aeneas.synthesizer import Synthesizer
from aeneas.textfile import TextFragment
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

class ExecuteTaskInputError(Exception):
    """
    Error raised when the input parameters of the task are invalid or missing.
    """
    pass



class ExecuteTaskExecutionError(Exception):
    """
    Error raised when the execution of the task fails for internal reasons.
    """
    pass



class ExecuteTask(object):
    """
    Execute a task, that is, compute the sync map for it.

    :param task: the task to be executed
    :type  task: :class:`aeneas.task.Task`
    :param rconf: a runtime configuration. Default: ``None``, meaning that
                  default settings will be used.
    :type  rconf: :class:`aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = u"ExecuteTask"

    def __init__(self, task, rconf=None, logger=None):
        self.task = task
        self.cleanup_info = []
        self.logger = logger if logger is not None else Logger()
        self.rconf = rconf if rconf is not None else RuntimeConfiguration()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def execute(self):
        """
        Execute the task.
        The sync map produced will be stored inside the task object.

        :raise ExecuteTaskInputError: if there is a problem with the input parameters
        :raise ExecuteTaskExecutionError: if there is a problem during the task execution
        """
        self._log(u"Executing task...")

        # check that we have the AudioFile object
        if self.task.audio_file is None:
            self._failed(u"The task does not seem to have its audio file set", False)
        if (
                (self.task.audio_file.audio_length is None) or
                (self.task.audio_file.audio_length <= 0)
            ):
            self._failed(u"The task seems to have an invalid audio file", False)
        if (
                (self.rconf["task_max_a_len"] > 0) and
                (self.task.audio_file.audio_length > self.rconf["task_max_a_len"])
            ):
            self._failed(u"The audio file of the task has length %.3f, more than the maximum allowed (%.3f)." % (
                self.task.audio_file.audio_length,
                self.rconf["task_max_a_len"]
            ), False)

        # check that we have the TextFile object
        if self.task.text_file is None:
            self._failed(u"The task does not seem to have its text file set", False)
        if len(self.task.text_file) == 0:
            self._failed(u"The task text file seems to have no text fragments", False)
        if (
                (self.rconf["task_max_t_len"] > 0) and
                (len(self.task.text_file) > self.rconf["task_max_t_len"])
            ):
            self._failed(u"The text file of the task has %d fragments, more than the maximum allowed (%d)." % (
                len(self.task.text_file),
                self.rconf["task_max_t_len"]
            ), False)
        if self.task.text_file.chars == 0:
            self._failed(u"The task text file seems to have empty text", False)

        self._log(u"Both audio and text input file are present")
        self.cleanup_info = []

        step_index = 1
        try:
            # STEP 1 : convert audio file to real full wave
            self._log(u"STEP %d BEGIN" % (step_index))
            real_full_handler, real_full_path = self._convert()
            self.cleanup_info.append([real_full_handler, real_full_path])
            self._log(u"STEP %d END" % (step_index))
            step_index += 1

            # STEP 2 : extract MFCCs from real wave
            self._log(u"STEP %d BEGIN" % (step_index))
            real_wave_mfcc = self._extract_mfcc(real_full_path)
            self._log(u"STEP %d END" % (step_index))
            step_index += 1

            # STEP 3 : set or detect head and/or tail
            self._log(u"STEP %d BEGIN" % (step_index))
            self._set_head_tail(real_wave_mfcc)
            self._log(u"STEP %d END" % (step_index))
            step_index += 1

            # STEP 4 : synthesize text to wave
            self._log(u"STEP %d BEGIN" % (step_index))
            synt_handler, synt_path, synt_anchors = self._synthesize()
            self.cleanup_info.append([synt_handler, synt_path])
            self._log(u"STEP %d END" % (step_index))
            step_index += 1

            # STEP 5 : extract MFCCs from synt wave
            self._log(u"STEP %d BEGIN" % (step_index))
            synt_wave_mfcc = self._extract_mfcc(synt_path)
            self._log(u"STEP %d END" % (step_index))
            step_index += 1

            # STEP 6 : align waves
            self._log(u"STEP %d BEGIN" % (step_index))
            wave_path = self._align_waves(real_wave_mfcc, synt_wave_mfcc)
            self._log(u"STEP %d END" % (step_index))
            step_index += 1

            # STEP 7 : compute time map, adjusting boundaries as requested 
            self._log(u"STEP %d BEGIN" % (step_index))
            time_map = self._compute_time_map(wave_path, synt_anchors, real_wave_mfcc)
            self._log(u"STEP %d END" % (step_index))
            step_index += 1

            # STEP 8 : create syncmap and add it to task
            self._log(u"STEP %d BEGIN" % (step_index))
            self._create_syncmap(time_map)
            self._log(u"STEP %d END" % (step_index))
            step_index += 1

            # STEP 9 : cleanup
            self._log(u"STEP %d BEGIN" % (step_index))
            self._cleanup()
            self._log(u"STEP %d END" % (step_index))
            step_index += 1

            self._log(u"Executing task... done")
        except Exception as exc:
            self._log(u"STEP %d FAILURE" % step_index, Logger.CRITICAL)
            self._cleanup()
            self._failed("%s" % (exc), True)

    def _failed(self, msg, during_execution=True):
        """ Bubble exception up """
        if during_execution:
            self._log(msg, Logger.CRITICAL)
            raise ExecuteTaskExecutionError(msg)
        else:
            self._log(msg, Logger.CRITICAL)
            raise ExecuteTaskInputError(msg)

    def _cleanup(self):
        """
        Remove all temporary files.
        """
        self._log(u"Cleaning up...")
        for info in self.cleanup_info:
            handler, path = info
            self._log([u"Removing file '%s'", path])
            gf.delete_file(handler, path)
        self.cleanup_info = []
        self._log(u"Cleaning up... done")

    def _convert(self):
        """
        Convert the entire audio file into a ``wav`` file.

        (Head/tail will be cut off later.)

        Return a pair:

        1. handler of the generated wave file
        2. path of the generated wave file
        """
        self._log(u"Converting real audio to WAVE")
        handler = None
        path = None
        self._log(u"Creating output tmp file")
        handler, path = gf.tmp_file(suffix=u".wav", root=self.rconf["tmp_path"])
        self._log(u"Creating FFMPEGWrapper object")
        ffmpeg = FFMPEGWrapper(rconf=self.rconf, logger=self.logger)
        self._log(u"Converting file...")
        ffmpeg.convert(
            input_file_path=self.task.audio_file_path_absolute,
            output_file_path=path
        )
        self._log(u"Converting file... done")
        self._log(u"Converting real audio to WAVE... done")
        return (handler, path)

    def _extract_mfcc(self, audio_file_path):
        """
        Extract the MFCCs of the given mono WAVE file.

        Return an AudioFileMFCC object.
        """
        self._log(u"Extracting MFCCs from full wave...")
        audio_file_mfcc = AudioFileMFCC(audio_file_path, rconf=self.rconf, logger=self.logger)
        self._log(u"Extracting MFCCs from full wave... done")
        return audio_file_mfcc

    def _set_head_tail(self, audio_file_mfcc):
        """
        Set the audio file head or tail,
        suitably cutting the audio file on disk,
        and setting the corresponding parameters in the task configuration.
        """
        self._log(u"Setting head/tail...")
        head_length = self.task.configuration["i_a_head"]
        process_length = self.task.configuration["i_a_process"]
        tail_length = self.task.configuration["i_a_tail"]
        head_max = self.task.configuration["i_a_head_max"]
        head_min = self.task.configuration["i_a_head_min"]
        tail_max = self.task.configuration["i_a_tail_max"]
        tail_min = self.task.configuration["i_a_tail_min"]
        if (
            (head_length is not None) or
            (process_length is not None) or
            (tail_length is not None)
        ):
            self._log(u"Setting explicit head/process/tail...")
            audio_file_mfcc.set_head_middle_tail(head_length, process_length, tail_length)
            self._log(u"Setting explicit head/process/tail... done")
        else:
            self._log(u"Detecting head/tail...")
            sd = SD(audio_file_mfcc, self.task.text_file, rconf=self.rconf, logger=self.logger)
            head_length = 0.0
            tail_length = 0.0
            if (head_min is not None) or (head_max is not None):
                self._log(u"Detecting HEAD...")
                head_length = sd.detect_head(head_min, head_max)
                self._log([u"Detected HEAD: %.3f", head_length])
                self._log(u"Detecting HEAD... done")
            if (tail_min is not None) or (tail_max is not None):
                self._log(u"Detecting TAIL...")
                tail_length = sd.detect_tail(tail_min, tail_max)
                self._log([u"Detected TAIL: %.3f", tail_length])
                self._log(u"Detecting TAIL... done")
            audio_file_mfcc.set_head_middle_tail(head_length=head_length, tail_length=tail_length)
            self._log(u"Detecting head/tail... done")
        self._log(u"Setting head/tail... done")

    def _synthesize(self):
        """
        Synthesize text into a WAVE file.

        Return a triple:

        1. handler of the generated wave file
        2. path of the generated wave file
        3. the list of anchors, that is, a list of floats
           each representing the start time of the corresponding
           text fragment in the generated wave file
           ``[start_1, start_2, ..., start_n]``
        """
        self._log(u"Synthesizing text")
        handler = None
        path = None
        anchors = None
        self._log(u"Creating an output tmp file")
        handler, path = gf.tmp_file(suffix=u".wav", root=self.rconf["tmp_path"])
        self._log(u"Creating Synthesizer object")
        synt = Synthesizer(rconf=self.rconf, logger=self.logger)
        self._log(u"Synthesizing...")
        result = synt.synthesize(self.task.text_file, path)
        anchors = result[0]
        self._log(u"Synthesizing... done")
        self._log(u"Synthesizing text: succeeded")
        return (handler, path, anchors)

    def _align_waves(self, real_wave_mfcc, synt_wave_mfcc):
        """
        Align two AudioFileMFCC objects,
        representing WAVE files.

        Return the computed path, that is,
        a tuple of two numpy 1D arrays of int
        ``(real_indices, synt_indices)``,
        where the indices are relative to the FULL wave,
        for both the real and the synt wave.
        """
        self._log(u"Aligning waves")
        self._log(u"Creating DTWAligner object")
        aligner = DTWAligner(real_wave_mfcc, synt_wave_mfcc, rconf=self.rconf, logger=self.logger)
        self._log(u"Computing path...")
        aligner.compute_path()
        self._log(u"Computing path... done")
        self._log(u"Aligning waves: succeeded")
        return aligner.computed_path

    def _compute_time_map(self, wave_path, synt_anchors, real_wave_mfcc):
        """
        Align the text with the real wave,
        using the ``wave_map`` (containing the mapping
        between real and synt waves) and ``synt_anchors``
        (containing the start times of text fragments
        in the synt wave).

        Return the computed time map, that is,
        a list of pairs ``[start_time, end_time]``,
        of length equal to number of fragments + 2,
        where the two extra elements are for
        the HEAD (first) and TAIL (last).
        """
        self._log(u"Computing time map...")
        self._log([u"Path length: %d", len(wave_path)])
        self._log([u"Fragments:   %d", len(synt_anchors)])

        self._log(u"Obtaining raw alignment...")
        mws = self.rconf["mfcc_window_shift"]
        # unpack wave_path 
        # both real_indices and synt_indices are w.r.t. the full wave
        real_indices, synt_indices = wave_path
        # synt_anchors as in seconds, convert them in MFCC indices
        anchor_indices = numpy.array([int(a[0] / mws) for a in synt_anchors])
        # right side sets the split point at the very beginning of "next" fragment
        begin_indices = numpy.searchsorted(synt_indices, anchor_indices, side="right")
        # first split must occur at zero 
        begin_indices[0] = 0.0
        # map onto real indices, obtaining "default" boundary indices
        boundary_indices = numpy.append(real_indices[begin_indices], real_wave_mfcc.tail_begin)
        self._log(u"Obtaining raw alignment... done")

        self._log(u"Adjusting boundaries...")
        # boundary_indices contains the boundary indices in the all_mfcc of real_wave_mfcc
        # starting with the (head-1st fragment) and ending with (-1th fragment-tail)
        aba_algorithm, aba_parameters = self.task.configuration.aba_parameters()
        time_map = AdjustBoundaryAlgorithm(
            algorithm=aba_algorithm,
            parameters=aba_parameters,
            real_wave_mfcc=real_wave_mfcc,
            boundary_indices=boundary_indices,
            text_file=self.task.text_file,
            rconf=self.rconf,
            logger=self.logger
        ).to_time_map()
        self._log(u"Adjusting boundaries... done")

        self._log(u"Computing time map... done")
        return time_map

    def _create_syncmap(self, time_map):
        """
        Create a sync map out of the provided time map,
        and store it in the task object.

        The time map is
        a list of pairs ``[start_time, end_time]``,
        of length equal to number of fragments + 2,
        where the two extra elements are for
        the HEAD (first) and TAIL (last).
        """
        self._log(u"Creating sync map")
        self._log([u"Fragments in time map (including HEAD/TAIL): %d", len(time_map)])

        # new sync map to be returned
        sync_map = SyncMap()

        # HEAD and TAIL are the first and last elements of time_map
        head = time_map[0]
        tail = time_map[-1]

        # get language for HEAD/TAIL (although the actual value does not matter)
        language = self.task.configuration["language"]

        # get HEAD/TAIL format
        head_tail_format = self.task.configuration["o_h_t_format"]
        self._log([u"Head/tail format: %s", str(head_tail_format)])

        # add head sync map fragment if needed
        if head_tail_format == SyncMapHeadTailFormat.ADD:
            head_frag = TextFragment(u"HEAD", language, [u""])
            sync_map_frag = SyncMapFragment(head_frag, head[0], head[1])
            sync_map.append_fragment(sync_map_frag)
            self._log([u"  Added head (ADD): %.3f %.3f", head[0], head[1]])

        # stretch first and last fragment timings if needed
        if head_tail_format == SyncMapHeadTailFormat.STRETCH:
            self._log([u"  Stretching (STRETCH): %.3f => %.3f (head)", time_map[1][0], head[0]])
            self._log([u"  Stretching (STRETCH): %.3f => %.3f (tail)", time_map[-2][1], tail[1]])
            time_map[1][0] = head[0]
            time_map[-2][1] = tail[1]

        # append fragments
        i = 1
        for fragment in self.task.text_file.fragments:
            start = time_map[i][0]
            end = time_map[i][1]
            sync_map_frag = SyncMapFragment(fragment, start, end)
            sync_map.append_fragment(sync_map_frag)
            self._log([u"  Added fragment %d: %.3f %.3f", i, start, end])
            i += 1

        # add tail sync map fragment if needed
        if head_tail_format == SyncMapHeadTailFormat.ADD:
            tail_frag = TextFragment(u"TAIL", language, [u""])
            sync_map_frag = SyncMapFragment(tail_frag, tail[0], tail[1])
            sync_map.append_fragment(sync_map_frag)
            self._log([u"  Added tail (ADD): %.3f %.3f", tail[0], tail[1]])

        self.task.sync_map = sync_map
        self._log(u"Creating sync map: succeeded")



