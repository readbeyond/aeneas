#!/usr/bin/env python
# coding=utf-8

"""
Execute a task, that is, compute the sync map for it.
"""

import numpy
import os
import tempfile

import aeneas.globalfunctions as gf
from aeneas.adjustboundaryalgorithm import AdjustBoundaryAlgorithm
from aeneas.dtw import DTWAligner
from aeneas.ffmpegwrapper import FFMPEGWrapper
from aeneas.logger import Logger
from aeneas.syncmap import SyncMap, SyncMapFragment
from aeneas.synthesizer import Synthesizer
from aeneas.vad import VAD

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.1.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class ExecuteTask(object):
    """
    Execute a task, that is, compute the sync map for it.

    :param task: the task to be executed
    :type  task: :class:`aeneas.task.Task`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = "ExecuteTask"

    def __init__(self, task, logger=None):
        self.task = task
        self.cleanup_info = []
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def execute(self):
        """
        Execute the task.
        The sync map produced will be stored inside the task object.

        Return ``True`` if the execution succeeded,
        ``False`` if an error occurred.

        :rtype: bool
        """
        self._log("Executing task")

        # check that we have the AudioFile object
        if self.task.audio_file is None:
            self._log("The task does not seem to have its audio file set", Logger.WARNING)
            return False
        if (
                (self.task.audio_file.audio_length is None) or
                (self.task.audio_file.audio_length <= 0)
            ):
            self._log("The task seems to have an invalid audio file", Logger.WARNING)
            return False

        # check that we have the TextFile object
        if self.task.text_file is None:
            self._log("The task does not seem to have its text file set", Logger.WARNING)
            return False
        if len(self.task.text_file) == 0:
            self._log("The task seems to have no text fragments", Logger.WARNING)
            return False

        self._log("Both audio and text input file are present")
        self.cleanup_info = []

        # STEP 1 : convert (real) audio to wave
        self._log("STEP 1 BEGIN")
        result, real_handler, real_path = self._convert()
        self.cleanup_info.append([real_handler, real_path])
        if not result:
            self._log("STEP 1 FAILURE")
            self._cleanup()
            return False
        self._log("STEP 1 END")

        # STEP 2 : synthesize text to wave
        self._log("STEP 2 BEGIN")
        result, synt_handler, synt_path, synt_anchors = self._synthesize()
        self.cleanup_info.append([synt_handler, synt_path])
        if not result:
            self._log("STEP 2 FAILURE")
            self._cleanup()
            return False
        self._log("STEP 2 END")

        # STEP 3 : align waves
        self._log("STEP 3 BEGIN")
        result, wave_map, real_wave_full_mfcc, real_wave_length = self._align_waves(real_path, synt_path)
        if not result:
            self._log("STEP 3 FAILURE")
            self._cleanup()
            return False
        self._log("STEP 3 END")

        # STEP 4 : align text
        self._log("STEP 4 BEGIN")
        result, text_map = self._align_text(wave_map, synt_anchors)
        if not result:
            self._log("STEP 4 FAILURE")
            self._cleanup()
            return False
        self._log("STEP 4 END")

        # STEP 5 : adjust boundaries
        self._log("STEP 5 BEGIN")
        result, adjusted_map = self._adjust_boundaries(
            text_map,
            real_wave_full_mfcc,
            real_wave_length
        )
        if not result:
            self._log("STEP 5 FAILURE")
            self._cleanup()
            return False
        self._log("STEP 5 END")

        # STEP 6 : create syncmap and add it to task
        self._log("STEP 6 BEGIN")
        result = self._create_syncmap(adjusted_map)
        if not result:
            self._log("STEP 6 FAILURE")
            self._cleanup()
            return False
        self._log("STEP 6 END")

        # STEP 7 : cleanup
        self._log("STEP 7 BEGIN")
        self._cleanup()
        self._log("STEP 7 END")
        self._log("Execution completed")
        return True

    def _cleanup(self):
        """
        Remove all temporary files.
        """
        for info in self.cleanup_info:
            handler, path = info
            if handler is not None:
                try:
                    self._log(["Closing handler '%s'...", handler])
                    os.close(handler)
                    self._log("Succeeded")
                except:
                    self._log("Failed")
            if path is not None:
                try:
                    self._log(["Removing path '%s'...", path])
                    os.remove(path)
                    self._log("Succeeded")
                except:
                    self._log("Failed")
        self.cleanup_info = []

    def _convert(self):
        """
        Convert the audio file into a ``wav`` file.

        Return a triple:

        1. a success bool flag
        2. handler of the generated wave file
        3. path of the generated wave file
        """
        self._log("Converting real audio to wav")
        handler = None
        path = None
        try:
            self._log("Creating an output tempfile")
            handler, path = tempfile.mkstemp(
                suffix=".wav",
                dir=gf.custom_tmp_dir()
            )
            self._log("Creating a FFMPEGWrapper")
            ffmpeg = FFMPEGWrapper(logger=self.logger)
            self._log("Converting...")
            ffmpeg.convert(
                input_file_path=self.task.audio_file_path_absolute,
                output_file_path=path,
                head_length=self.task.configuration.is_audio_file_head_length,
                process_length=self.task.configuration.is_audio_file_process_length)
            self._log("Converting... done")
            self._log("Converting real audio to wav: succeeded")
            return (True, handler, path)
        except:
            self._log("Converting real audio to wav: failed")
            return (False, handler, path)

    def _synthesize(self):
        """
        Synthesize text into a ``wav`` file.

        Return a quadruple:

        1. a success bool flag
        2. handler of the generated wave file
        3. path of the generated wave file
        4. the list of anchors, that is, a list of floats
           each representing the start time of the corresponding
           text fragment in the generated wave file
           ``[start_1, start_2, ..., start_n]``
        """
        self._log("Synthesizing text")
        handler = None
        path = None
        anchors = None
        try:
            self._log("Creating an output tempfile")
            handler, path = tempfile.mkstemp(
                suffix=".wav",
                dir=gf.custom_tmp_dir()
            )
            self._log("Creating Synthesizer object")
            synt = Synthesizer(logger=self.logger)
            self._log("Synthesizing...")
            anchors = synt.synthesize(self.task.text_file, path)
            self._log("Synthesizing... done")
            self._log("Synthesizing text: succeeded")
            return (True, handler, path, anchors)
        except:
            self._log("Synthesizing text: failed")
            return (False, handler, path, anchors)

    def _align_waves(self, real_path, synt_path):
        """
        Align two ``wav`` files.

        Return a pair:

        1. a success bool flag
        2. the computed alignment map, that is,
           a list of pairs of floats, each representing
           corresponding time instants
           in the real and synt wave, respectively
           ``[real_time, synt_time]``
        3. the MFCCs of the real wave
        4. the length of the real wave
        """
        self._log("Aligning waves")
        try:
            self._log("Creating DTWAligner object")
            aligner = DTWAligner(real_path, synt_path, logger=self.logger)
            self._log("Computing MFCC...")
            aligner.compute_mfcc()
            self._log("Computing MFCC... done")
            real_mfcc = aligner.real_wave_full_mfcc
            real_len = aligner.real_wave_length
            self._log("Computing path...")
            aligner.compute_path()
            self._log("Computing path... done")
            self._log("Computing map...")
            computed_map = aligner.computed_map
            self._log("Computing map... done")
            return (True, computed_map, real_mfcc, real_len)
        except:
            return (False, None, None, None)

    def _align_text(self, wave_map, synt_anchors):
        """
        Align the text with the real wave,
        using the ``wave_map`` (containing the mapping
        between real and synt waves) and ``synt_anchors``
        (containing the start times of text fragments
        in the synt wave).

        Return a pair:

        1. a success bool flag
        2. the computed interval map, that is,
           a list of triples ``[start_time, end_time, fragment_id]``
        """
        self._log("Align text")
        self._log(["Number of frames:    %d", len(wave_map)])
        self._log(["Number of fragments: %d", len(synt_anchors)])
        try:
            real_times = numpy.array([t[0] for t in wave_map])
            synt_times = numpy.array([t[1] for t in wave_map])
            real_anchors = []
            anchor_index = 0
            # TODO numpy-fy this loop
            for anchor in synt_anchors:
                time, fragment_id, fragment_text = anchor
                self._log("Looking for argmin index...")
                # TODO allow an user-specified function instead of min
                # partially solved by AdjustBoundaryAlgorithm
                index = (numpy.abs(synt_times - time)).argmin()
                self._log("Looking for argmin index... done")
                real_time = real_times[index]
                real_anchors.append([real_time, fragment_id, fragment_text])
                self._log(["Time for anchor %d: %f", anchor_index, real_time])
                anchor_index += 1

            # dummy last anchor, starting at the real file duration
            real_anchors.append([real_times[-1], None, None])

            # compute map
            self._log("Computing interval map...")
            # TODO numpy-fy this loop
            computed_map = []
            for i in range(len(real_anchors) - 1):
                fragment_id = real_anchors[i][1]
                fragment_text = real_anchors[i][2]
                start = real_anchors[i][0]
                end = real_anchors[i+1][0]
                computed_map.append([start, end, fragment_id, fragment_text])
            self._log("Computing interval map... done")

            # return computed map
            self._log("Returning interval map")
            return (True, computed_map)
        except:
            return (False, None)

    def _adjust_boundaries(
            self,
            text_map,
            real_wave_full_mfcc,
            real_wave_length
        ):
        """
        Adjust the boundaries between consecutive fragments.

        Return a pair:

        1. a success bool flag
        2. the computed interval map, that is,
           a list of triples ``[start_time, end_time, fragment_id]``

        """
        algo = self.task.configuration.adjust_boundary_algorithm
        value = None
        if algo is None:
            self._log("No adjust boundary algorithm specified: returning")
            return (True, text_map)
        elif algo == AdjustBoundaryAlgorithm.AUTO:
            self._log("Requested adjust boundary algorithm AUTO: returning")
            return (True, text_map)
        elif algo == AdjustBoundaryAlgorithm.AFTERCURRENT:
            value = self.task.configuration.adjust_boundary_aftercurrent_value
        elif algo == AdjustBoundaryAlgorithm.BEFORENEXT:
            value = self.task.configuration.adjust_boundary_beforenext_value
        elif algo == AdjustBoundaryAlgorithm.OFFSET:
            value = self.task.configuration.adjust_boundary_offset_value
        elif algo == AdjustBoundaryAlgorithm.PERCENT:
            value = self.task.configuration.adjust_boundary_percent_value
        elif algo == AdjustBoundaryAlgorithm.RATE:
            value = self.task.configuration.adjust_boundary_rate_value
        elif algo == AdjustBoundaryAlgorithm.RATEAGGRESSIVE:
            value = self.task.configuration.adjust_boundary_rate_value
        self._log(["Requested algo %s and value %s", algo, value])

        try:
            self._log("Running VAD...")
            vad = VAD(logger=self.logger)
            vad.wave_mfcc = real_wave_full_mfcc
            vad.wave_len = real_wave_length
            vad.compute_vad()
            self._log("Running VAD... done")
        except:
            return (False, None)

        self._log("Creating AdjustBoundaryAlgorithm object")
        adjust_boundary = AdjustBoundaryAlgorithm(
            algorithm=algo,
            text_map=text_map,
            speech=vad.speech,
            nonspeech=vad.nonspeech,
            value=value,
            logger=self.logger
        )
        self._log("Adjusting boundaries...")
        adjusted_map = adjust_boundary.adjust()
        self._log("Adjusting boundaries... done")
        return (True, adjusted_map)

    def _create_syncmap(self, text_map):
        """
        Create a sync map out of the provided interval map,
        and store it in the task object.

        Return a success bool flag.
        """
        self._log("Creating SyncMap")
        self._log(["Number of fragments: %d", len(text_map)])
        if len(text_map) != len(self.task.text_file.fragments):
            return False
        try:
            sync_map = SyncMap()
            i = 0
            head = 0
            if self.task.configuration.is_audio_file_head_length is not None:
                head = gf.safe_float(self.task.configuration.is_audio_file_head_length, 0)
            for fragment in self.task.text_file.fragments:
                start = head + text_map[i][0]
                end = head + text_map[i][1]
                sync_map_frag = SyncMapFragment(fragment, start, end)
                sync_map.append(sync_map_frag)
                i += 1
            self.task.sync_map = sync_map
            return True
        except:
            return False



