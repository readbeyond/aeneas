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
This "hydra" tool invokes another aeneas.tool,
according to the specified tool switch.
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from aeneas.tools.abstract_cli_program import AbstractCLIProgram
from aeneas.tools.convert_syncmap import ConvertSyncMapCLI
from aeneas.tools.download import DownloadCLI
from aeneas.tools.execute_job import ExecuteJobCLI
from aeneas.tools.execute_task import ExecuteTaskCLI
from aeneas.tools.extract_mfcc import ExtractMFCCCLI
from aeneas.tools.ffmpeg_wrapper import FFMPEGWrapperCLI
from aeneas.tools.ffprobe_wrapper import FFPROBEWrapperCLI
from aeneas.tools.plot_waveform import PlotWaveformCLI
from aeneas.tools.read_audio import ReadAudioCLI
from aeneas.tools.read_text import ReadTextCLI
from aeneas.tools.run_sd import RunSDCLI
from aeneas.tools.run_vad import RunVADCLI
from aeneas.tools.synthesize_text import SynthesizeTextCLI
from aeneas.tools.validate import ValidateCLI
import aeneas.globalfunctions as gf


class HydraCLI(AbstractCLIProgram):
    """
    This "hydra" tool invokes another aeneas.tool,
    according to the specified tool switch.
    """
    NAME = gf.file_name_without_extension(__file__)

    HELP = {
        "description": u"Invoke the specified aeneas tool",
        "synopsis": [
            (u"TOOL_PARAMETER TOOL_ARGUMENTS", True)
        ],
        "options": [
        ],
        "parameters": [
            u"--convert-syncmap: call aeneas.tools.convert_syncmap",
            u"--download: call aeneas.tools.download",
            u"--execute-job: call aeneas.tools.execute_job",
            u"--execute-task: call aeneas.tools.execute_task (default)",
            u"--extract-mfcc: call aeneas.tools.extract_mfcc",
            u"--ffmpeg-wrapper: call aeneas.tools.ffmpeg_wrapper",
            u"--ffprobe-wrapper: call aeneas.tools.ffprobe_wrapper",
            u"--plot-waveform: call aeneas.tools.plot_waveform",
            u"--read-audio: call aeneas.tools.read_audio",
            u"--read-text: call aeneas.tools.read_text",
            u"--run-sd: call aeneas.tools.run_sd",
            u"--run-vad: call aeneas.tools.run_vad",
            u"--synthesize-text: call aeneas.tools.synthesize_text",
            u"--validate: call aeneas.tools.validate",
        ],
        "examples": [
            u"--execute-task --help",
            u"--execute-task --examples",
            u"--execute-task --example-json",
            u"--execute-job --help",
        ]
    }

    TOOLS = [
        (ConvertSyncMapCLI, [u"--convert-syncmap"]),
        (DownloadCLI, [u"--download"]),
        (ExecuteJobCLI, [u"--execute-job"]),
        (ExecuteTaskCLI, [u"--execute-task"]),
        (ExtractMFCCCLI, [u"--extract-mfcc"]),
        (FFMPEGWrapperCLI, [u"--ffmpeg-wrapper"]),
        (FFPROBEWrapperCLI, [u"--ffprobe-wrapper"]),
        (PlotWaveformCLI, [u"--plot-waveform"]),
        (ReadAudioCLI, [u"--read-audio"]),
        (ReadTextCLI, [u"--read-text"]),
        (RunSDCLI, [u"--run-sd"]),
        (RunVADCLI, [u"--run-vad"]),
        (SynthesizeTextCLI, [u"--synthesize-text"]),
        (ValidateCLI, [u"--validate"]),
    ]

    def perform_command(self):
        """
        Perform command and return the appropriate exit code.

        :rtype: int
        """
        # if no actual arguments, print help
        if len(self.actual_arguments) < 1:
            return self.print_help(short=True)

        # check if we have a recognized tool switch
        for cls, switches in self.TOOLS:
            if self.has_option(switches):
                arguments = [a for a in sys.argv if a not in switches]
                return cls(invoke=(self.invoke + u" %s" % switches[0])).run(arguments=arguments)

        # check if we have -h, --help, or --version
        if u"-h" in self.actual_arguments:
            return self.print_help(short=True)
        if u"--help" in self.actual_arguments:
            return self.print_help(short=False)
        if u"--version" in self.actual_arguments:
            return self.print_name_version()

        # default to run ExecuteTaskCLI
        return ExecuteTaskCLI(invoke=self.invoke).run(arguments=sys.argv)


def main():
    """
    Execute program.
    """
    HydraCLI().run(arguments=sys.argv, show_help=False)

if __name__ == '__main__':
    main()
