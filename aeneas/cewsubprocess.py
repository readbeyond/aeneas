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
This module contains the following classes:

* :class:`aeneas.cewsubprocess.CEWSubprocess` which is an
  helper class executes the :mod:`aeneas.cew` C extension
  in a separate process via ``subprocess``.

This module works around a problem with the ``eSpeak`` library,
which seems to generate different audio data for the same
input parameters/text, when run multiple times in the same process.
See the following discussions for details:

#. https://groups.google.com/d/msg/aeneas-forced-alignment/NLbtSRf2_vg/mMHuTQiFEgAJ
#. https://sourceforge.net/p/espeak/mailman/message/34861696/

.. warning:: This module might be removed in a future version.

.. versionadded:: 1.5.0
"""

from __future__ import absolute_import
from __future__ import print_function
import io
import subprocess
import sys

from aeneas.exacttiming import TimeValue
from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration
import aeneas.globalfunctions as gf


class CEWSubprocess(Loggable):
    """
    This helper class executes the ``aeneas.cew`` C extension
    in a separate process by running
    the :func:`aeneas.cewsubprocess.CEWSubprocess.main` function
    via ``subprocess``.

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    TAG = u"CEWSubprocess"

    def synthesize_multiple(self, audio_file_path, c_quit_after, c_backwards, u_text):
        """
        Synthesize the text contained in the given fragment list
        into a ``wav`` file.

        :param string audio_file_path: the path to the output audio file
        :param float c_quit_after: stop synthesizing as soon as
                                   reaching this many seconds
        :param bool c_backwards: synthesizing from the end of the text file
        :param object u_text: a list of ``(voice_code, text)`` tuples
        :rtype: tuple ``(sample_rate, synthesized, intervals)``
        """
        self.log([u"Audio file path: '%s'", audio_file_path])
        self.log([u"c_quit_after: '%.3f'", c_quit_after])
        self.log([u"c_backwards: '%d'", c_backwards])

        text_file_handler, text_file_path = gf.tmp_file()
        data_file_handler, data_file_path = gf.tmp_file()
        self.log([u"Temporary text file path: '%s'", text_file_path])
        self.log([u"Temporary data file path: '%s'", data_file_path])

        self.log(u"Populating the text file...")
        with io.open(text_file_path, "w", encoding="utf-8") as tmp_text_file:
            for f_voice_code, f_text in u_text:
                tmp_text_file.write(u"%s %s\n" % (f_voice_code, f_text))
        self.log(u"Populating the text file... done")

        arguments = [
            self.rconf[RuntimeConfiguration.CEW_SUBPROCESS_PATH],
            "-m",
            "aeneas.cewsubprocess",
            "%.3f" % c_quit_after,
            "%d" % c_backwards,
            text_file_path,
            audio_file_path,
            data_file_path
        ]
        self.log([u"Calling with arguments '%s'", u" ".join(arguments)])
        proc = subprocess.Popen(
            arguments,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True)
        proc.communicate()

        self.log(u"Reading output data...")
        with io.open(data_file_path, "r", encoding="utf-8") as data_file:
            lines = data_file.read().splitlines()
            sr = int(lines[0])
            sf = int(lines[1])
            intervals = []
            for line in lines[2:]:
                values = line.split(u" ")
                if len(values) == 2:
                    intervals.append((TimeValue(values[0]), TimeValue(values[1])))
        self.log(u"Reading output data... done")

        self.log(u"Deleting text and data files...")
        gf.delete_file(text_file_handler, text_file_path)
        gf.delete_file(data_file_handler, data_file_path)
        self.log(u"Deleting text and data files... done")

        return (sr, sf, intervals)


def main():
    """
    Run ``aeneas.cew``, reading input text from file and writing audio and interval data to file.
    """

    # make sure we have enough parameters
    if len(sys.argv) < 6:
        print("You must pass five arguments: QUIT_AFTER BACKWARDS TEXT_FILE_PATH AUDIO_FILE_PATH DATA_FILE_PATH")
        return 1

    # read parameters
    c_quit_after = float(sys.argv[1])   # NOTE: cew needs float, not TimeValue
    c_backwards = int(sys.argv[2])
    text_file_path = sys.argv[3]
    audio_file_path = sys.argv[4]
    data_file_path = sys.argv[5]

    # read (voice_code, text) from file
    s_text = []
    with io.open(text_file_path, "r", encoding="utf-8") as text:
        for line in text.readlines():
            # NOTE: not using strip() to avoid removing trailing blank characters
            line = line.replace(u"\n", u"").replace(u"\r", u"")
            idx = line.find(" ")
            if idx > 0:
                f_voice_code = line[:idx]
                f_text = line[(idx + 1):]
                s_text.append((f_voice_code, f_text))

    # convert to bytes/unicode as required by subprocess
    c_text = []
    if gf.PY2:
        for f_voice_code, f_text in s_text:
            c_text.append((gf.safe_bytes(f_voice_code), gf.safe_bytes(f_text)))
    else:
        for f_voice_code, f_text in s_text:
            c_text.append((gf.safe_unicode(f_voice_code), gf.safe_unicode(f_text)))

    try:
        import aeneas.cew.cew
        sr, sf, intervals = aeneas.cew.cew.synthesize_multiple(
            audio_file_path,
            c_quit_after,
            c_backwards,
            c_text
        )
        with io.open(data_file_path, "w", encoding="utf-8") as data:
            data.write(u"%d\n" % (sr))
            data.write(u"%d\n" % (sf))
            data.write(u"\n".join([u"%.3f %.3f" % (i[0], i[1]) for i in intervals]))
    except Exception as exc:
        print(u"Unexpected error: %s" % str(exc))


if __name__ == "__main__":
    main()
