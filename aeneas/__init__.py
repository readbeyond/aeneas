#!/usr/bin/env python
# coding=utf-8

"""
**aeneas** is a Python library and a set of tools
to automagically synchronize audio and text.
"""

from aeneas.adjustboundaryalgorithm import AdjustBoundaryAlgorithm
from aeneas.analyzecontainer import AnalyzeContainer
from aeneas.audiofile import AudioFile
from aeneas.container import Container, ContainerFormat
from aeneas.dtw import DTWAlgorithm, DTWAligner
from aeneas.espeakwrapper import ESPEAKWrapper
from aeneas.executejob import ExecuteJob
from aeneas.executetask import ExecuteTask
from aeneas.ffmpegwrapper import FFMPEGWrapper
from aeneas.ffprobewrapper import FFPROBEWrapper
import aeneas.globalconstants as gc
import aeneas.globalfunctions as gf
from aeneas.hierarchytype import HierarchyType
from aeneas.idsortingalgorithm import IDSortingAlgorithm
from aeneas.job import Job, JobConfiguration
from aeneas.language import Language
from aeneas.logger import Logger
from aeneas.sd import SD, SDMetric
from aeneas.syncmap import SyncMap, SyncMapFragment, SyncMapFormat, SyncMapHeadTailFormat
from aeneas.synthesizer import Synthesizer
from aeneas.task import Task, TaskConfiguration
from aeneas.textfile import TextFile, TextFileFormat, TextFragment
from aeneas.vad import VAD
from aeneas.validator import Validator

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.2.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"



