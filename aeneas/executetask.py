#!/usr/bin/env python
# coding=utf-8

"""
Execute a task, that is, compute the sync map for it.
"""

import numpy
import os
import tempfile

from aeneas.adjustboundaryalgorithm import AdjustBoundaryAlgorithm
from aeneas.audiofile import AudioFileMonoWAV
from aeneas.dtw import DTWAligner
from aeneas.ffmpegwrapper import FFMPEGWrapper
from aeneas.language import Language
from aeneas.logger import Logger
from aeneas.sd import SD
from aeneas.syncmap import SyncMap
from aeneas.syncmap import SyncMapFragment
from aeneas.syncmap import SyncMapHeadTailFormat
from aeneas.synthesizer import Synthesizer
from aeneas.textfile import TextFragment
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

        :raise ExecuteTaskInputError: if there is a problem with the input parameters
        :raise ExecuteTaskExecutionError: if there is a problem during the task execution
        """
        self._log("Executing task")

        # check that we have the AudioFile object
        if self.task.audio_file is None:
            self._failed("The task does not seem to have its audio file set", False)
        if (
                (self.task.audio_file.audio_length is None) or
                (self.task.audio_file.audio_length <= 0)
            ):
            self._failed("The task seems to have an invalid audio file", False)

        # check that we have the TextFile object
        if self.task.text_file is None:
            self._failed("The task does not seem to have its text file set", False)
        if len(self.task.text_file) == 0:
            self._failed("The task text file seems to have no text fragments", False)
        if self.task.text_file.chars == 0:
            self._failed("The task text file seems to have empty text", False)

        self._log("Both audio and text input file are present")
        self.cleanup_info = []

        # real full wave    = the real audio file, converted to WAVE format
        # real trimmed wave = real full wave, possibly with head and/or tail trimmed off
        # synt wave         = WAVE file synthesized from text; it will be aligned to real trimmed wave

        step_index = 0
        try:
            # STEP 0 : convert audio file to real full wave
            self._log("STEP %d BEGIN" % (step_index))
            real_full_handler, real_full_path = self._convert()
            self.cleanup_info.append([real_full_handler, real_full_path])
            self._log("STEP %d END" % (step_index))
            step_index += 1

            # STEP 1 : extract MFCCs from real full wave
            self._log("STEP %d BEGIN" % (step_index))
            real_full_wave_full_mfcc, real_full_wave_length = self._extract_mfcc(real_full_path)
            self._log("STEP %d END" % (step_index))
            step_index += 1

            # STEP 2 : cut head and/or tail off
            #          detecting head/tail if requested, and
            #          overwriting real_path
            #          at the end, read_path will not have the head/tail
            self._log("STEP %d BEGIN" % (step_index))
            self._cut_head_tail(real_full_path)
            real_trimmed_path = real_full_path
            self._log("STEP %d END" % (step_index))
            step_index += 1

            # STEP 3 : synthesize text to wave
            self._log("STEP %d BEGIN" % (step_index))
            synt_handler, synt_path, synt_anchors = self._synthesize()
            self.cleanup_info.append([synt_handler, synt_path])
            self._log("STEP %d END" % (step_index))
            step_index += 1

            # STEP 4 : align waves
            self._log("STEP %d BEGIN" % (step_index))
            wave_map = self._align_waves(real_trimmed_path, synt_path)
            self._log("STEP %d END" % (step_index))
            step_index += 1

            # STEP 5 : align text
            self._log("STEP %d BEGIN" % (step_index))
            text_map = self._align_text(wave_map, synt_anchors)
            self._log("STEP %d END" % (step_index))
            step_index += 1

            # STEP 6 : translate the text_map, possibly putting back the head/tail
            self._log("STEP %d BEGIN" % (step_index))
            translated_text_map = self._translate_text_map(
                text_map,
                real_full_wave_length
            )
            self._log("STEP %d END" % (step_index))
            step_index += 1

            # STEP 7 : adjust boundaries
            self._log("STEP %d BEGIN" % (step_index))
            adjusted_map = self._adjust_boundaries(
                translated_text_map,
                real_full_wave_full_mfcc,
                real_full_wave_length
            )
            self._log("STEP %d END" % (step_index))
            step_index += 1

            # STEP 8 : create syncmap and add it to task
            self._log("STEP %d BEGIN" % (step_index))
            self._create_syncmap(adjusted_map)
            self._log("STEP %d END" % (step_index))
            step_index += 1

            # STEP 9 : cleanup
            self._log("STEP %d BEGIN" % (step_index))
            self._cleanup()
            self._log("STEP %d END" % (step_index))
            step_index += 1

            self._log("Execution completed")
            return True
        except Exception as exc:
            self._log("STEP %d FAILURE" % step_index, Logger.CRITICAL)
            self._cleanup()
            self._failed(str(exc), True)
        self._log("Executing task... done")

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
        self._log("Cleaning up...")
        for info in self.cleanup_info:
            handler, path = info
            self._log(["Removing file '%s'", path])
            gf.delete_file(handler, path)
        self.cleanup_info = []
        self._log("Cleaning up... done")

    def _convert(self):
        """
        Convert the entire audio file into a ``wav`` file.

        (Head/tail will be cut off later.)

        Return a pair:

        1. handler of the generated wave file
        2. path of the generated wave file
        """
        self._log("Converting real audio to wav")
        handler = None
        path = None
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
            output_file_path=path)
        self._log("Converting... done")
        self._log("Converting real audio to wav: succeeded")
        return (handler, path)

    def _extract_mfcc(self, audio_file_path):
        """
        Extract the MFCCs of the real full wave.
        
        Return a pair:

        1. audio MFCCs
        2. audio length
        """
        self._log("Extracting MFCCs from real full wave")
        audio_file = AudioFileMonoWAV(audio_file_path, logger=self.logger)
        audio_file.extract_mfcc()
        self._log("Extracting MFCCs from real full wave: succeeded")
        return (audio_file.audio_mfcc, audio_file.audio_length)

    def _cut_head_tail(self, audio_file_path):
        """
        Set the audio file head or tail,
        suitably cutting the audio file on disk,
        and setting the corresponding parameters in the task configuration.
        """
        self._log("Setting head and/or tail")
        configuration = self.task.configuration
        head_length = configuration.is_audio_file_head_length
        process_length = configuration.is_audio_file_process_length
        tail_length = configuration.is_audio_file_tail_length
        detect_head_min = configuration.is_audio_file_detect_head_min
        detect_head_max = configuration.is_audio_file_detect_head_max
        detect_tail_min = configuration.is_audio_file_detect_tail_min
        detect_tail_max = configuration.is_audio_file_detect_tail_max

        # explicit head or process?
        explicit = (
            (head_length is not None) or
            (process_length is not None) or
            (tail_length is not None)
        )

        # at least one detect parameter?
        detect = (
            (detect_head_min is not None) or
            (detect_head_max is not None) or
            (detect_tail_min is not None) or
            (detect_tail_max is not None)
        )

        if not (explicit or detect):
            # nothing to do
            self._log("No explicit head/process or detect head/tail")
        else:
            # we need to load the audio data
            audio_file = AudioFileMonoWAV(audio_file_path, logger=self.logger)
            audio_file.load_data()

            if explicit:
                self._log("Explicit head, process, or tail")
            else:
                self._log("No explicit head, process, or tail => detecting head/tail")

                head = 0.0
                if (detect_head_min is not None) or (detect_head_max is not None):
                    self._log("Detecting head...")
                    detect_head_min = gf.safe_float(detect_head_min, gc.SD_MIN_HEAD_LENGTH)
                    detect_head_max = gf.safe_float(detect_head_max, gc.SD_MAX_HEAD_LENGTH)
                    self._log(["detect_head_min is %.3f", detect_head_min])
                    self._log(["detect_head_max is %.3f", detect_head_max])
                    start_detector = SD(audio_file, self.task.text_file, logger=self.logger)
                    head = start_detector.detect_head(detect_head_min, detect_head_max)
                    self._log(["Detected head: %.3f", head])

                tail = 0.0
                if (detect_tail_min is not None) or (detect_tail_max is not None):
                    self._log("Detecting tail...")
                    detect_tail_max = gf.safe_float(detect_tail_max, gc.SD_MAX_TAIL_LENGTH)
                    detect_tail_min = gf.safe_float(detect_tail_min, gc.SD_MIN_TAIL_LENGTH)
                    self._log(["detect_tail_min is %.3f", detect_tail_min])
                    self._log(["detect_tail_max is %.3f", detect_tail_max])
                    start_detector = SD(audio_file, self.task.text_file, logger=self.logger)
                    tail = start_detector.detect_tail(detect_tail_min, detect_tail_max)
                    self._log(["Detected tail: %.3f", tail])

                head_length = max(0, head)
                process_length = max(0, audio_file.audio_length - tail - head)
                tail_length = audio_file.audio_length - head_length - process_length

                # we need to set these values
                # in the config object for later use
                self.task.configuration.is_audio_file_head_length = head_length
                self.task.configuration.is_audio_file_process_length = process_length
                self._log(["Set head_length:    %.3f", head_length])
                self._log(["Set process_length: %.3f", process_length])

            # in case we are reading from config object
            if head_length is not None:
                self._log("head_length is not None, converting to float")
                head_length = float(head_length)
            else:
                self._log("head_length is None: setting it to 0.0")
                head_length = 0.0
            # note that process_length and tail_length are mutually exclusive
            # with process_length having precedence over tail_length
            if process_length is not None:
                self._log("process_length is not None, converting to float")
                process_length = float(process_length)
                if tail_length is not None:
                    self._log("tail_length is not None, but it will be ignored")
                    tail_length = float(tail_length)
            elif tail_length is not None:
                self._log("tail_length is not None, converting to float")
                tail_length = float(tail_length)
                self._log("computing process_length from tail_length")
                process_length = audio_file.audio_length - head_length - tail_length

            self._log(["is_audio_file_head_length is %s", str(head_length)])
            self._log(["is_audio_file_process_length is %s", str(process_length)])
            self._log(["is_audio_file_tail_length is %s", str(tail_length)])

            self._log("Trimming audio data...")
            audio_file.trim(head_length, process_length)
            self._log("Trimming audio data... done")

            self._log("Writing audio file...")
            audio_file.write(audio_file_path)
            self._log("Writing audio file... done")

            self._log("Clearing audio data...")
            audio_file.clear_data()
            self._log("Clearing audio data... done")

        self._log("Setting head and/or tail: succeeded")

    def _synthesize(self):
        """
        Synthesize text into a ``wav`` file.

        Return a triple:

        1. handler of the generated wave file
        2. path of the generated wave file
        3. the list of anchors, that is, a list of floats
           each representing the start time of the corresponding
           text fragment in the generated wave file
           ``[start_1, start_2, ..., start_n]``
        """
        self._log("Synthesizing text")
        handler = None
        path = None
        anchors = None
        self._log("Creating an output tempfile")
        handler, path = tempfile.mkstemp(
            suffix=".wav",
            dir=gf.custom_tmp_dir()
        )
        self._log("Creating Synthesizer object")
        synt = Synthesizer(logger=self.logger)
        self._log("Synthesizing...")
        result = synt.synthesize(self.task.text_file, path)
        anchors = result[0]
        self._log("Synthesizing... done")
        self._log("Synthesizing text: succeeded")
        return (handler, path, anchors)

    def _align_waves(self, real_path, synt_path):
        """
        Align two ``wav`` files.

        Return the computed alignment map, that is,
        a list of pairs of floats, each representing
        corresponding time instants
        in the real and synt wave, respectively
        ``[real_time, synt_time]``
        """
        self._log("Aligning waves")
        self._log("Creating DTWAligner object")
        aligner = DTWAligner(real_path, synt_path, logger=self.logger)
        self._log("Computing MFCC...")
        aligner.compute_mfcc()
        self._log("Computing MFCC... done")
        self._log("Computing path...")
        aligner.compute_path()
        self._log("Computing path... done")
        self._log("Computing map...")
        computed_map = aligner.computed_map
        self._log("Computing map... done")
        self._log("Aligning waves: succeeded")
        return computed_map

    def _align_text(self, wave_map, synt_anchors):
        """
        Align the text with the real wave,
        using the ``wave_map`` (containing the mapping
        between real and synt waves) and ``synt_anchors``
        (containing the start times of text fragments
        in the synt wave).

        Return the computed interval map, that is,
        a list of triples ``[start_time, end_time, fragment_id]``
        """
        self._log("Aligning text")
        self._log(["Number of frames:    %d", len(wave_map)])
        self._log(["Number of fragments: %d", len(synt_anchors)])

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
        self._log("Aligning text: succeeded")
        return computed_map

    def _translate_text_map(self, text_map, real_full_wave_length):
        """
        Translate the text_map by adding head and tail dummy fragments

        Return the translated text map
        """
        translated = []
        head = gf.safe_float(self.task.configuration.is_audio_file_head_length, 0)
        translated.append([0, head, None, None])
        end = 0
        for element in text_map:
            start, end, fragment_id, fragment_text = element
            start += head
            end += head
            translated.append([start, end, fragment_id, fragment_text])
        translated.append([end, real_full_wave_length, None, None])
        return translated

    def _adjust_boundaries(
            self,
            text_map,
            real_wave_full_mfcc,
            real_wave_length
        ):
        """
        Adjust the boundaries between consecutive fragments.

        Return the computed interval map, that is,
        a list of triples ``[start_time, end_time, fragment_id]``
        """
        self._log("Adjusting boundaries")
        algo = self.task.configuration.adjust_boundary_algorithm
        value = None
        if algo is None:
            self._log("No adjust boundary algorithm specified: returning")
            return text_map
        elif algo == AdjustBoundaryAlgorithm.AUTO:
            self._log("Requested adjust boundary algorithm AUTO: returning")
            return text_map
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

        self._log("Running VAD...")
        vad = VAD(real_wave_full_mfcc, real_wave_length, logger=self.logger)
        vad.compute_vad()
        self._log("Running VAD... done")

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
        self._log("Adjusting boundaries: succeeded")
        return adjusted_map

    def _create_syncmap(self, adjusted_map):
        """
        Create a sync map out of the provided interval map,
        and store it in the task object.
        """
        self._log("Creating sync map")
        self._log(["Number of fragments in adjusted map (including HEAD and TAIL): %d", len(adjusted_map)])

        # adjusted map has 2 elements (HEAD and TAIL) more than text_file
        #if len(adjusted_map) != len(self.task.text_file.fragments) + 2:
        #    self._log("The number of sync map fragments does not match the number of text fragments (+2)", Logger.CRITICAL)
        #    return False
            
        sync_map = SyncMap()
        head = adjusted_map[0]
        tail = adjusted_map[-1]

        # get language
        language = Language.EN
        self._log(["Language set to default: %s", language])
        if len(self.task.text_file.fragments) > 0:
            language = self.task.text_file.fragments[0].language
            self._log(["Language read from text_file: %s", language])

        # get head/tail format
        head_tail_format = self.task.configuration.os_file_head_tail_format
        self._log(["Head/tail format: %s", str(head_tail_format)])

        # add head sync map fragment if needed
        if head_tail_format == SyncMapHeadTailFormat.ADD:
            head_frag = TextFragment(u"HEAD", language, [u""])
            sync_map_frag = SyncMapFragment(head_frag, head[0], head[1])
            sync_map.append_fragment(sync_map_frag)
            self._log(["Adding head (ADD): %.3f %.3f", head[0], head[1]])

        # stretch first and last fragment timings if needed
        if head_tail_format == SyncMapHeadTailFormat.STRETCH:
            self._log(["Stretching (STRETCH): %.3f => %.3f (head) and %.3f => %.3f (tail)", adjusted_map[1][0], head[0], adjusted_map[-2][1], tail[1]])
            adjusted_map[1][0] = head[0]
            adjusted_map[-2][1] = tail[1]

        i = 1
        for fragment in self.task.text_file.fragments:
            start = adjusted_map[i][0]
            end = adjusted_map[i][1]
            sync_map_frag = SyncMapFragment(fragment, start, end)
            sync_map.append_fragment(sync_map_frag)
            i += 1

        # add tail sync map fragment if needed
        if head_tail_format == SyncMapHeadTailFormat.ADD:
            tail_frag = TextFragment(u"TAIL", language, [u""])
            sync_map_frag = SyncMapFragment(tail_frag, tail[0], tail[1])
            sync_map.append_fragment(sync_map_frag)
            self._log(["Adding tail (ADD): %.3f %.3f", tail[0], tail[1]])

        self.task.sync_map = sync_map
        self._log("Creating sync map: succeeded")



