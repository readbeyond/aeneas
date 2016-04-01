#!/usr/bin/env python
# coding=utf-8

"""
This module contains the following classes:

* :class:`~aeneas.executetask.ExecuteTask`, a class to process a task;
* :class:`~aeneas.executetask.ExecuteTaskExecutionError`, and
* :class:`~aeneas.executetask.ExecuteTaskInputError`,
  representing errors generated while processing tasks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy

from aeneas.adjustboundaryalgorithm import AdjustBoundaryAlgorithm
from aeneas.audiofile import AudioFile
from aeneas.audiofilemfcc import AudioFileMFCC
from aeneas.dtw import DTWAligner
from aeneas.ffmpegwrapper import FFMPEGWrapper
from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.sd import SD
from aeneas.syncmap import SyncMap
from aeneas.syncmap import SyncMapFragment
from aeneas.syncmap import SyncMapHeadTailFormat
from aeneas.synthesizer import Synthesizer
from aeneas.task import Task
from aeneas.textfile import TextFileFormat
from aeneas.textfile import TextFragment
from aeneas.timevalue import TimeValue
from aeneas.tree import Tree
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

class ExecuteTaskExecutionError(Exception):
    """
    Error raised when the execution of the task fails for internal reasons.
    """
    pass



class ExecuteTaskInputError(Exception):
    """
    Error raised when the input parameters of the task are invalid or missing.
    """
    pass



class ExecuteTask(Loggable):
    """
    Execute a task, that is, compute the sync map for it.

    :param task: the task to be executed
    :type  task: :class:`~aeneas.task.Task`
    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    TAG = u"ExecuteTask"

    def __init__(self, task=None, rconf=None, logger=None):
        super(ExecuteTask, self).__init__(rconf=rconf, logger=logger)
        self.task = task
        self.step_index = 1
        self.step_label = u""
        self.step_begin_time = None
        self.step_total = 0.000
        if task is not None:
            self.load_task(self.task)

    def load_task(self, task):
        """
        Load the task from the given ``Task`` object.

        :param task: the task to load
        :type  task: :class:`~aeneas.task.Task`
        :raises: :class:`~aeneas.executetask.ExecuteTaskInputError`: if ``task`` is not an instance of :class:`~aeneas.task.Task`
        """
        if not isinstance(task, Task):
            self.log_exc(u"task is not an instance of Task", None, True, ExecuteTaskInputError)
        self.task = task

    def _step_begin(self, label, log=True):
        """ Log begin of a step """
        if log:
            self.step_label = label
            self.step_begin_time = self.log(u"STEP %d BEGIN (%s)" % (self.step_index, label))

    def _step_end(self, log=True):
        """ Log end of a step """
        if log:
            step_end_time = self.log(u"STEP %d END (%s)" % (self.step_index, self.step_label))
            diff = (step_end_time - self.step_begin_time)
            diff = float(diff.seconds + diff.microseconds / 1000000.0)
            self.step_total += diff
            self.log(u"STEP %d DURATION %.3f (%s)" % (self.step_index, diff, self.step_label))
            self.step_index += 1

    def _step_failure(self, exc):
        """ Log failure of a step """
        self.log_crit(u"STEP %d (%s) FAILURE" % (self.step_index, self.step_label))
        self.step_index += 1
        self.log_exc(u"Unexpected error while executing task", exc, True, ExecuteTaskExecutionError)

    def _step_total(self):
        """ Log total """
        self.log(u"STEP T DURATION %.3f" % (self.step_total))

    def execute(self):
        """
        Execute the task.
        The sync map produced will be stored inside the task object.

        :raises: :class:`~aeneas.executetask.ExecuteTaskInputError`: if there is a problem with the input parameters
        :raises: :class:`~aeneas.executetask.ExecuteTaskExecutionError`: if there is a problem during the task execution
        """
        self.log(u"Executing task...")

        # check that we have the AudioFile object
        if self.task.audio_file is None:
            self.log_exc(u"The task does not seem to have its audio file set", None, True, ExecuteTaskInputError)
        if (
                (self.task.audio_file.audio_length is None) or
                (self.task.audio_file.audio_length <= 0)
            ):
            self.log_exc(u"The task seems to have an invalid audio file", None, True, ExecuteTaskInputError)
        task_max_audio_length = self.rconf[RuntimeConfiguration.TASK_MAX_AUDIO_LENGTH]
        if (
                (task_max_audio_length > 0) and
                (self.task.audio_file.audio_length > task_max_audio_length)
            ):
            self.log_exc(u"The audio file of the task has length %.3f, more than the maximum allowed (%.3f)." % (self.task.audio_file.audio_length, task_max_audio_length), None, True, ExecuteTaskInputError)

        # check that we have the TextFile object
        if self.task.text_file is None:
            self.log_exc(u"The task does not seem to have its text file set", None, True, ExecuteTaskInputError)
        if len(self.task.text_file) == 0:
            self.log_exc(u"The task text file seems to have no text fragments", None, True, ExecuteTaskInputError)
        task_max_text_length = self.rconf[RuntimeConfiguration.TASK_MAX_TEXT_LENGTH]
        if (
                (task_max_text_length > 0) and
                (len(self.task.text_file) > task_max_text_length)
            ):
            self.log_exc(u"The text file of the task has %d fragments, more than the maximum allowed (%d)." % (len(self.task.text_file), task_max_text_length), None, True, ExecuteTaskInputError)
        if self.task.text_file.chars == 0:
            self.log_exc(u"The task text file seems to have empty text", None, True, ExecuteTaskInputError)

        self.log(u"Both audio and text input file are present")

        # execute
        self.step_index = 1
        self.step_total = 0.000
        if self.task.text_file.file_format in [TextFileFormat.MPLAIN, TextFileFormat.MUNPARSED]:
            self._execute_multi_level_task()
        else:
            self._execute_single_level_task()
        self.log(u"Executing task... done")

    def _execute_single_level_task(self):
        """ Execute a single-level task """
        self.log(u"Executing single level task...")
        try:
            # load audio file, extract MFCCs from real wave, clear audio file
            self._step_begin(u"extract MFCC real wave")
            real_wave_mfcc = self._extract_mfcc(file_path=self.task.audio_file_path_absolute, file_path_is_mono_wave=False)
            self._step_end()

            # compute head and/or tail and set it
            self._step_begin(u"compute head tail")
            (head_length, process_length, tail_length) = self._compute_head_process_tail(real_wave_mfcc)
            real_wave_mfcc.set_head_middle_tail(head_length, process_length, tail_length)
            self._step_end()

            # compute a time map alignment
            time_map = self._execute_inner(real_wave_mfcc, self.task.text_file, adjust_boundaries=True, log=True)

            # convert time_map to tree and create syncmap and add it to task
            self._step_begin(u"create sync map")
            tree = self._level_time_map_to_tree(self.task.text_file, time_map)
            self.task.sync_map = self._create_syncmap(tree)
            self._step_end()

            # check for fragments with zero duration
            self._step_begin(u"check zero duration")
            self._check_no_zero(self.rconf.mws)
            self._step_end()

            # log total
            self._step_total()
            self.log(u"Executing single level task... done")
        except Exception as exc:
            self._step_failure(exc)

    def _execute_multi_level_task(self):
        """ Execute a multi-level task """
        self.log(u"Executing multi level task...")

        self.log(u"Saving rconf...")
        # save original rconf
        orig_rconf = self.rconf.clone()
        # clone rconfs and set granularity
        level_rconfs = [None, self.rconf.clone(), self.rconf.clone(), self.rconf.clone()]
        level_mfccs = [None, None, None, None]
        for i in range(1, len(level_rconfs)):
            level_rconfs[i].set_granularity(i)
            self.log([u"Level %d mws: %.3f", i, level_rconfs[i].mws])
        self.log(u"Saving rconf... done")

        try:
            self.log(u"Creating AudioFile object...")
            audio_file = self._load_audio_file()
            self.log(u"Creating AudioFile object... done")

            # extract MFCC for each level
            for i in range(1, len(level_rconfs)):
                self._step_begin(u"extract MFCC real wave level %d" % i)
                if (i == 1) or (level_rconfs[i].mws != level_rconfs[i-1].mws) or (level_rconfs[i].mwl != level_rconfs[i-1].mwl):
                    self.rconf = level_rconfs[i]
                    level_mfccs[i] = self._extract_mfcc(audio_file=audio_file)
                else:
                    self.log(u"Keeping MFCC real wave from previous level")
                    level_mfccs[i] = level_mfccs[i-1]
                self._step_end()

            self.log(u"Clearing AudioFile object...")
            self.rconf = level_rconfs[1]
            self._clear_audio_file(audio_file)
            self.log(u"Clearing AudioFile object... done")

            # compute head tail for the entire real wave (level 1)
            self._step_begin(u"compute head tail")
            (head_length, process_length, tail_length) = self._compute_head_process_tail(level_mfccs[1])
            level_mfccs[1].set_head_middle_tail(head_length, process_length, tail_length)
            self._step_end()

            # compute alignment at each level
            tree = Tree()
            sync_roots = [tree]
            text_files = [self.task.text_file]
            aht = [None, True, False, False]
            aba = [None, True, True, False]
            for i in range(1, len(level_rconfs)):
                self._step_begin(u"compute alignment level %d" % i)
                text_files, sync_roots = self._execute_level(i, level_rconfs[i], level_mfccs[i], text_files, sync_roots, aht[i], aba[i])
                self._step_end()

            self._step_begin(u"select levels")
            tree = self._select_levels(tree)
            self._step_end()

            self._step_begin(u"create sync map")
            self.rconf = orig_rconf
            self.task.sync_map = self._create_syncmap(tree)
            self._step_end()

            self._step_begin(u"check zero duration")
            self._check_no_zero(level_rconfs[-1].mws)
            self._step_end()

            self._step_total()
            self.log(u"Executing multi level task... done")
        except Exception as exc:
            self._step_failure(exc)

    def _execute_level(self, level, rconf, audio_file_mfcc, text_files, sync_roots, add_head_tail, adjust_boundaries):
        """
        Compute the alignment for all the nodes in the given level.

        Return a pair (next_level_text_files, next_level_sync_roots),
        containing two lists of text file subtrees and sync map subtrees
        on the next level.

        :param int level: the level
        :param rconf: the runtime configuration for this level
        :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
        :param audio_file_mfcc: the audio MFCC representation for this level
        :type  audio_file_mfcc: :class:`~aeneas.audiofilemfcc.AudioFileMFCC`
        :param list text_files: a list of :class:`~aeneas.textfile.TextFile` objects,
                                each representing a (sub)tree of the Task text file
        :param list sync_roots: a list of :class:`~aeneas.tree.Tree` objects,
                                each representing a SyncMapFragment tree,
                                one for each element in ``text_files``
        :param bool add_head_tail: if ``True``, add head and tail nodes to the sync map tree
        :param bool adjust_boundaries: if ``True``, execute the adjust boundary algorithm
        :rtype: (list, list)
        """
        self.rconf = rconf
        i = 0
        next_level_text_files = []
        next_level_sync_roots = []
        for text_file in text_files:
            self.log([u"Text level %d, fragment %d", level, i])
            self.log([u"  Len:   %d", len(text_file)])
            sync_root = sync_roots[i]
            if (level > 1) and (len(text_file) == 1):
                self.log(u"  Level > 1 and only one child => returning trivial timemap")
                time_map = [
                    (TimeValue("0.000"), sync_root.value.begin),
                    (sync_root.value.begin, sync_root.value.end),
                    (sync_root.value.end, audio_file_mfcc.audio_length)
                ]
            else:
                self.log(u"  Level 1 or more than one child => computing timemap")
                if not sync_root.is_empty:
                    begin = sync_root.value.begin
                    end = sync_root.value.end
                    self.log([u"  Begin: %.3f", begin])
                    self.log([u"  End:   %.3f", end])
                    audio_file_mfcc.set_head_middle_tail(head_length=begin, middle_length=(end - begin))
                else:
                    self.log(u"  No begin or end to set")
                time_map = self._execute_inner(audio_file_mfcc, text_file, adjust_boundaries=adjust_boundaries, log=False)
            self.log([u"  Map:   %s", str(time_map)])
            self._level_time_map_to_tree(text_file, time_map, sync_root, add_head_tail=add_head_tail)
            # store next level roots
            next_level_text_files.extend(text_file.children_not_empty)
            src = sync_root.children
            if add_head_tail:
                # if we added head and tail,
                # we must not pass them to the next level
                src = src[1:-1]
            next_level_sync_roots.extend(src)
            i += 1
        return (next_level_text_files, next_level_sync_roots)

    def _execute_inner(self, audio_file_mfcc, text_file, adjust_boundaries=True, log=True):
        """
        Align a subinterval of the given AudioFileMFCC
        with the given TextFile.

        Return the computed time map, as a list of intervals.

        The begin and end positions inside the AudioFileMFCC
        must have been set ahead by the caller.

        The text fragments being aligned are the vchildren of ``text_file``.

        :param audio_file_mfcc: the audio file MFCC representation
        :type  audio_file_mfcc: :class:`~aeneas.audiofilemfcc.AudioFileMFCC`
        :param text_file: the text file subtree to align
        :type  text_file: :class:`~aeneas.textfile.TextFile`
        :param bool adjust_boundaries: if ``True``, execute the adjust boundary algorithm
        :param bool log: if ``True``, log steps
        :rtype: list
        """
        self._step_begin(u"synthesize text", log=log)
        synt_handler, synt_path, synt_anchors, synt_mono = self._synthesize(text_file)
        self._step_end(log=log)

        self._step_begin(u"extract MFCC synt wave", log=log)
        synt_wave_mfcc = self._extract_mfcc(file_path=synt_path, file_path_is_mono_wave=synt_mono)
        gf.delete_file(synt_handler, synt_path)
        self._step_end(log=log)

        self._step_begin(u"align waves", log=log)
        indices = self._align_waves(audio_file_mfcc, synt_wave_mfcc, synt_anchors)
        self._step_end(log=log)

        self._step_begin(u"adjust boundaries", log=log)
        time_map = self._adjust_boundaries(audio_file_mfcc, text_file, indices, adjust_boundaries)
        self._step_end(log=log)

        return time_map

    def _load_audio_file(self):
        """
        Load audio in memory.

        :rtype: :class:`~aeneas.audiofile.AudioFile`
        """
        self._step_begin(u"load audio file")
        audio_file = AudioFile(
            file_path=self.task.audio_file_path_absolute,
            is_mono_wave=False,
            rconf=self.rconf,
            logger=self.logger
        )
        audio_file.read_samples_from_file()
        self._step_end()
        return audio_file

    def _clear_audio_file(self, audio_file):
        """
        Clear audio from memory.

        :param audio_file: the object to clear
        :type  audio_file: :class:`~aeneas.audiofile.AudioFile`
        """
        self._step_begin(u"clear audio file")
        audio_file.clear_data()
        audio_file = None
        self._step_end()

    def _extract_mfcc(self, file_path=None, file_path_is_mono_wave=False, audio_file=None):
        """
        Extract the MFCCs from the given audio file.

        :rtype: :class:`~aeneas.audiofilemfcc.AudioFileMFCC`
        """
        return AudioFileMFCC(
            file_path=file_path,
            file_path_is_mono_wave=file_path_is_mono_wave,
            audio_file=audio_file,
            rconf=self.rconf,
            logger=self.logger
        )

    def _compute_head_process_tail(self, audio_file_mfcc):
        """
        Set the audio file head or tail,
        by either reading the explicit values
        from the Task configuration,
        or using SD to determine them.

        This function returns the lengths, in seconds,
        of the (head, process, tail).

        :rtype: tuple (float, float, float)
        """
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
            self.log(u"Setting explicit head process tail")
        else:
            self.log(u"Detecting head tail...")
            sd = SD(audio_file_mfcc, self.task.text_file, rconf=self.rconf, logger=self.logger)
            head_length = TimeValue("0.000")
            process_length = None
            tail_length = TimeValue("0.000")
            if (head_min is not None) or (head_max is not None):
                self.log(u"Detecting HEAD...")
                head_length = sd.detect_head(head_min, head_max)
                self.log([u"Detected HEAD: %.3f", head_length])
                self.log(u"Detecting HEAD... done")
            if (tail_min is not None) or (tail_max is not None):
                self.log(u"Detecting TAIL...")
                tail_length = sd.detect_tail(tail_min, tail_max)
                self.log([u"Detected TAIL: %.3f", tail_length])
                self.log(u"Detecting TAIL... done")
            self.log(u"Detecting head tail... done")
        self.log([u"Head:    %s", gf.safe_float(head_length, None)])
        self.log([u"Process: %s", gf.safe_float(process_length, None)])
        self.log([u"Tail:    %s", gf.safe_float(tail_length, None)])
        return (head_length, process_length, tail_length)

    def _synthesize(self, text_file):
        """
        Synthesize text into a WAVE file.

        Return:

        1. handler of the generated wave file
        2. path of the generated wave file
        3. the list of anchors, that is, a list of floats
           each representing the start time of the corresponding
           text fragment in the generated wave file
           ``[start_1, start_2, ..., start_n]``
        4. if the synthesizer produced a PCM16 mono WAVE file

        :param synthesizer: the synthesizer to use
        :type  synthesizer: :class:`~aeneas.synthesizer.Synthesizer`
        :rtype: tuple (handler, string, list)
        """
        synthesizer = Synthesizer(rconf=self.rconf, logger=self.logger)
        handler, path = gf.tmp_file(suffix=u".wav", root=self.rconf[RuntimeConfiguration.TMP_PATH])
        result = synthesizer.synthesize(text_file, path)
        anchors = result[0]
        return (handler, path, anchors, synthesizer.output_is_mono_wave)

    def _align_waves(self, real_wave_mfcc, synt_wave_mfcc, synt_anchors):
        """
        Align two AudioFileMFCC objects,
        representing WAVE files.

        Return a list of boundary indices.
        """
        self.log(u"Creating DTWAligner...")
        aligner = DTWAligner(real_wave_mfcc, synt_wave_mfcc, rconf=self.rconf, logger=self.logger)
        self.log(u"Creating DTWAligner... done")
        self.log(u"Computing boundary indices...")
        boundary_indices = aligner.compute_boundaries(synt_anchors)
        self.log(u"Computing boundary indices... done")
        return boundary_indices

    def _adjust_boundaries(self, real_wave_mfcc, text_file, boundary_indices, adjust_boundaries=True):
        """
        Adjust boundaries as requested by the user.

        Return the computed time map, that is,
        a list of pairs ``[start_time, end_time]``,
        of length equal to number of fragments + 2,
        where the two extra elements are for
        the HEAD (first) and TAIL (last).
        """
        # boundary_indices contains the boundary indices in the all_mfcc of real_wave_mfcc
        # starting with the (head-1st fragment) and ending with (-1th fragment-tail)
        if adjust_boundaries:
            aba_algorithm, aba_parameters = self.task.configuration.aba_parameters()
            self.log([u"Running algorithm: '%s'", aba_algorithm])
        else:
            self.log(u"Forced running algorithm: 'auto'")
            aba_algorithm = AdjustBoundaryAlgorithm.AUTO
            aba_parameters = None
        return AdjustBoundaryAlgorithm(
            algorithm=aba_algorithm,
            parameters=aba_parameters,
            real_wave_mfcc=real_wave_mfcc,
            boundary_indices=boundary_indices,
            text_file=text_file,
            rconf=self.rconf,
            logger=self.logger
        ).to_time_map()

    def _level_time_map_to_tree(self, text_file, time_map, tree=None, add_head_tail=True):
        """
        Convert a level time map into a Tree of SyncMapFragments.

        The time map is
        a list of pairs ``[start_time, end_time]``,
        of length equal to number of fragments + 2,
        where the two extra elements are for
        the HEAD (first) and TAIL (last).

        :param text_file: the text file object
        :type  text_file: :class:`~aeneas.textfile.TextFile`
        :param list time_map: the time map
        :param tree: the tree; if ``None``, a new Tree will be built
        :type  tree: :class:`~aeneas.tree.Tree`
        :rtype: :class:`~aeneas.tree.Tree`
        """
        if tree is None:
            tree = Tree()
        if add_head_tail:
            fragments = (
                [TextFragment(u"HEAD", self.task.configuration["language"], [u""])] +
                text_file.fragments +
                [TextFragment(u"TAIL", self.task.configuration["language"], [u""])]
            )
            i = 0
        else:
            fragments = text_file.fragments
            i = 1
        for fragment in fragments:
            interval = time_map[i]
            sm_frag = SyncMapFragment(fragment, interval[0], interval[1])
            tree.add_child(Tree(value=sm_frag))
            i += 1
        return tree

    def _select_levels(self, tree):
        """
        Select the correct levels in the tree,
        reading the ``os_task_file_levels``
        parameter in the Task configuration.

        If ``None`` or invalid, return the current sync map tree
        unchanged.
        Otherwise, return only the levels appearing in it.

        :param tree: a Tree of SyncMapFragments
        :type  tree: :class:`~aeneas.tree.Tree`
        :rtype: :class:`~aeneas.tree.Tree`
        """
        levels = self.task.configuration["o_levels"]
        self.log([u"Levels: '%s'", levels])
        if (levels is None) or (len(levels) < 1):
            return tree
        try:
            levels = [int(l) for l in levels if int(l) > 0]
            self.log([u"Converted levels: %s", levels])
        except ValueError:
            self.log_warn(u"Cannot convert levels to list of int, returning unchanged")
            return tree
        # remove head and tail nodes
        head = tree.vchildren[0]
        tail = tree.vchildren[-1]
        tree.remove_child(0)
        tree.remove_child(-1)
        # keep only the selected levels
        tree.keep_levels(levels)
        # add head and tail back
        tree.add_child(Tree(value=head), as_last=False)
        tree.add_child(Tree(value=tail), as_last=True)
        # return the new tree
        return tree

    def _create_syncmap(self, tree):
        """
        Return a sync map corresponding to the provided text file and time map.

        :param tree: a Tree of SyncMapFragments
        :type  tree: :class:`~aeneas.tree.Tree`
        :rtype: :class:`~aeneas.syncmap.SyncMap`
        """
        self.log([u"Fragments in time map (including HEAD/TAIL): %d", len(tree)])
        head_tail_format = self.task.configuration["o_h_t_format"]
        self.log([u"Head/tail format: %s", str(head_tail_format)])

        children = tree.vchildren
        head = children[0]
        first = children[1]
        last = children[-2]
        tail = children[-1]

        # remove HEAD fragment if needed
        if head_tail_format != SyncMapHeadTailFormat.ADD:
            tree.remove_child(0)
            self.log(u"Removed HEAD")

        # stretch first and last fragment timings if needed
        if head_tail_format == SyncMapHeadTailFormat.STRETCH:
            self.log([u"Stretched first.begin: %.3f => %.3f (head)", first.begin, head.begin])
            self.log([u"Stretched last.end:    %.3f => %.3f (tail)", last.end, tail.end])
            first.begin = head.begin
            last.end = tail.end

        # remove TAIL fragment if needed
        if head_tail_format != SyncMapHeadTailFormat.ADD:
            tree.remove_child(-1)
            self.log(u"Removed TAIL")

        # return sync map
        sync_map = SyncMap()
        sync_map.fragments_tree = tree
        return sync_map

    # TODO can this be done during the alignment?
    def _check_no_zero(self, min_mws):
        """ Check for fragments with zero duration """
        if self.task.configuration["o_no_zero"]:
            self.log(u"Checking for fragments with zero duration...")
            # TODO use min_mws when doable, e.g. only one fragment?
            delta = TimeValue("0.001")
            leaves = self.task.sync_map.fragments_tree.vleaves_not_empty
            # first and last leaves are HEAD and TAIL, skipping them
            max_index = len(leaves) - 1
            self.log([u"Fragment min index: %d", 1])
            self.log([u"Fragment max index: %d", max_index - 1])
            for i in range(1, max_index):
                self.log([u"Checking index:     %d", i])
                j = i
                while (j < max_index) and (leaves[j].end == leaves[i].begin):
                    j += 1
                if j != i:
                    self.log(u"Fragment(s) with zero duration:")
                    for k in range(i, j):
                        self.log([u"  %d : %s", k, leaves[k]])

                    if leaves[j].end - leaves[j].begin > (j - i) * delta:
                        # there is room after
                        # to move each zero fragment forward by 0.001
                        for k in range(j - i):
                            shift = (k + 1) * delta
                            leaves[i + k].end += shift
                            leaves[i + k + 1].begin += shift
                            self.log([u"  Moved fragment %d forward by %.3f", i + k, shift])
                    else:
                        self.log_warn(u"  Unable to fix")
                    i = j - 1
            self.log(u"Checking for fragments with zero duration... done")
        else:
            self.log(u"Not checking for fragments with zero duration")



