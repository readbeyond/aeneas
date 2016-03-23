#!/usr/bin/env python
# coding=utf-8

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

.. warning:: This module might be removed in a future version

.. versionadded:: 1.5.0
"""

from __future__ import absolute_import
from __future__ import print_function
import io
import subprocess
import sys

from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.timevalue import TimeValue
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

    def synthesize_single(self, audio_file_path, voice_code, text):
        """
        Create a ``wav`` audio file containing the synthesized text.

        The ``text`` must be a unicode string encodable with UTF-8,
        otherwise ``espeak`` might fail.

        Return the duration of the synthesized audio file, in seconds.

        :param string audio_file_path: the path of the output audio file
        :param string voice_code: the code of the voice to use
        :param string text: the text to synthesize
        :rtype: :class:`~aeneas.timevalue.TimeValue`
        """
        u_text = [(voice_code, text)]
        sr, sf, intervals = self.synthesize_multiple(audio_file_path, 0, 0, u_text)
        if len(intervals) > 0:
            return intervals[0][1]
        return None

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
    c_quit_after = float(sys.argv[1]) # NOTE: cew needs float, not TimeValue
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
                f_text = line[idx+1:]
                #print("%s => '%s' and '%s'" % (line, f_voice_code, f_text))
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



