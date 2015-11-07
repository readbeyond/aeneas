#!/usr/bin/env python
# coding=utf-8

"""
**aeneas** is a Python library and a set of tools
to automagically synchronize audio and text.
"""

from aeneas.adjustboundaryalgorithm import AdjustBoundaryAlgorithm
from aeneas.analyzecontainer import AnalyzeContainer
from aeneas.audiofile import AudioFile
from aeneas.audiofile import AudioFileMonoWAV
from aeneas.audiofile import AudioFileUnsupportedFormatError
from aeneas.container import Container
from aeneas.container import ContainerFormat
from aeneas.downloader import Downloader
from aeneas.dtw import DTWAlgorithm
from aeneas.dtw import DTWAligner
from aeneas.espeakwrapper import ESPEAKWrapper
from aeneas.executejob import ExecuteJob
from aeneas.executetask import ExecuteTask
from aeneas.executetask import ExecuteTaskExecutionError
from aeneas.executetask import ExecuteTaskInputError
from aeneas.ffmpegwrapper import FFMPEGWrapper
from aeneas.ffprobewrapper import FFPROBEParsingError
from aeneas.ffprobewrapper import FFPROBEUnsupportedFormatError
from aeneas.ffprobewrapper import FFPROBEWrapper
from aeneas.hierarchytype import HierarchyType
from aeneas.idsortingalgorithm import IDSortingAlgorithm
from aeneas.job import Job
from aeneas.job import JobConfiguration
from aeneas.language import Language
from aeneas.logger import Logger
from aeneas.sd import SD
from aeneas.sd import SDMetric
from aeneas.syncmap import SyncMap
from aeneas.syncmap import SyncMapFormat
from aeneas.syncmap import SyncMapFragment
from aeneas.syncmap import SyncMapHeadTailFormat
from aeneas.syncmap import SyncMapMissingParameterError
from aeneas.synthesizer import Synthesizer
from aeneas.task import Task
from aeneas.task import TaskConfiguration
from aeneas.textfile import TextFile
from aeneas.textfile import TextFileFormat
from aeneas.textfile import TextFragment
from aeneas.vad import VAD
from aeneas.validator import Validator
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



