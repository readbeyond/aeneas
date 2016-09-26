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
Plot a waveform and labels to file.
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

from aeneas.audiofile import AudioFile
from aeneas.syncmap import SyncMap
from aeneas.syncmap import SyncMapFormat
from aeneas.tools.abstract_cli_program import AbstractCLIProgram
import aeneas.globalfunctions as gf


class PlotWaveformCLI(AbstractCLIProgram):
    """
    Plot a waveform and labels to file.
    """
    AUDIO_FILE = gf.relative_path("res/audio.mp3", __file__)
    VAD_FILE = gf.relative_path("res/sonnet.vad", __file__)
    OUTPUT_FILE = "output/sonnet.png"

    NAME = gf.file_name_without_extension(__file__)

    HELP = {
        "description": u"Plot a waveform and labels to file.",
        "synopsis": [
            (u"AUDIO_FILE OUTPUT_FILE [-i LABEL_FILE [-i LABEL_FILE [...]]]", True)
        ],
        "examples": [
            u"%s %s -i %s" % (AUDIO_FILE, OUTPUT_FILE, VAD_FILE)
        ],
        "options": [
            u"--fast : enable fast waveform rendering (default: False)",
            u"--hzoom=ZOOM : horizontal zoom (int, default: 5)",
            u"--label=LABEL : label for the plot (str)",
            u"--text : show fragment text instead of identifier",
            u"--time-step=STEP : print time ticks every STEP seconds (int)",
            u"--vzoom=ZOOM : vertical zoom (int, default: 30)"
        ]
    }

    def perform_command(self):
        """
        Perform command and return the appropriate exit code.

        :rtype: int
        """
        if len(self.actual_arguments) < 2:
            return self.print_help()
        input_file_path = self.actual_arguments[0]
        output_file_path = self.actual_arguments[1]

        if not self.check_input_file(input_file_path):
            return self.ERROR_EXIT_CODE
        if not self.check_output_file(output_file_path):
            return self.ERROR_EXIT_CODE

        fast = self.has_option("--fast")
        fragment_text = self.has_option("--text")
        h_zoom = gf.safe_int(self.has_option_with_value("--hzoom"), 5)
        label = self.has_option_with_value("--label")
        time_step = gf.safe_int(self.has_option_with_value("--time-step"), 0)
        v_zoom = gf.safe_int(self.has_option_with_value("--vzoom"), 30)

        labels = not self.has_option("--no-labels")
        begin_times = not self.has_option("--no-begin-times")
        end_times = not self.has_option("--no-end-times")
        begin_guides = not self.has_option("--no-begin-guides")
        end_guides = not self.has_option("--no-end-guides")

        try:
            # import or ImportError
            from aeneas.plotter import PlotLabelset
            from aeneas.plotter import PlotTimeScale
            from aeneas.plotter import PlotWaveform
            from aeneas.plotter import Plotter

            # create plotter object
            self.print_info(u"Plotting to file...")
            plotter = Plotter(rconf=self.rconf, logger=self.logger)

            # add waveform
            afm = AudioFile(input_file_path, rconf=self.rconf, logger=self.logger)
            afm.read_samples_from_file()
            plotter.add_waveform(PlotWaveform(afm, label=label, fast=fast, rconf=self.rconf, logger=self.logger))

            # add time scale, if requested
            if time_step > 0:
                plotter.add_timescale(PlotTimeScale(afm.audio_length, time_step=time_step, rconf=self.rconf, logger=self.logger))

            # add labelsets, if any
            for i in range(len(self.actual_arguments)):
                if (self.actual_arguments[i] == "-i") and (i + 1 < len(self.actual_arguments)):
                    label_file_path = self.actual_arguments[i + 1]
                    extension = gf.file_extension(label_file_path)
                    if extension == "vad":
                        labelset = self._read_syncmap_file(label_file_path, SyncMapFormat.TSV, False)
                        ls = PlotLabelset(labelset, parameters=None, rconf=self.rconf, logger=self.logger)
                        ls.parameters["labels"] = False
                        ls.parameters["begin_time"] = begin_times
                        ls.parameters["end_time"] = end_times
                        ls.parameters["begin_guide"] = begin_guides
                        ls.parameters["end_guide"] = end_guides
                        plotter.add_labelset(ls)
                    if extension in SyncMapFormat.ALLOWED_VALUES:
                        labelset = self._read_syncmap_file(label_file_path, extension, fragment_text)
                        ls = PlotLabelset(labelset, parameters=None, rconf=self.rconf, logger=self.logger)
                        ls.parameters["labels"] = labels
                        ls.parameters["begin_time"] = begin_times
                        ls.parameters["end_time"] = end_times
                        ls.parameters["begin_guide"] = begin_guides
                        ls.parameters["end_guide"] = end_guides
                        plotter.add_labelset(ls)

            # output to file
            plotter.draw_png(output_file_path, h_zoom=h_zoom, v_zoom=v_zoom)
            self.print_info(u"Plotting to file... done")
            self.print_success(u"Created file '%s'" % output_file_path)
            return self.NO_ERROR_EXIT_CODE
        except ImportError:
            self.print_error(u"You need to install Python module Pillow to output image to file. Run:")
            self.print_error(u"$ pip install Pillow")
            self.print_error(u"or, to install for all users:")
            self.print_error(u"$ sudo pip install Pillow")
        except Exception as exc:
            self.print_error(u"An unexpected error occurred while generating the image file:")
            self.print_error(u"%s" % exc)

        return self.ERROR_EXIT_CODE

    def _read_syncmap_file(self, path, extension, text=False):
        """ Read labels from a SyncMap file """
        syncmap = SyncMap(logger=self.logger)
        syncmap.read(extension, path, parameters=None)
        if text:
            return [(f.begin, f.end, u" ".join(f.text_fragment.lines)) for f in syncmap.fragments]
        return [(f.begin, f.end, f.text_fragment.identifier) for f in syncmap.fragments]


def main():
    """
    Execute program.
    """
    PlotWaveformCLI().run(arguments=sys.argv)

if __name__ == '__main__':
    main()
