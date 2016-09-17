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

* :class:`~aeneas.diagnostics.Diagnostics`,
  checking whether the setup of ``aeneas`` was successful.

This module can be executed from command line with::

    python -m aeneas.diagnostics

.. versionadded:: 1.4.1
"""

from __future__ import absolute_import
from __future__ import print_function
import sys

import aeneas.globalfunctions as gf


class Diagnostics(object):
    """
    Check whether the setup of ``aeneas`` was successful.
    """
    @classmethod
    def check_shell_encoding(cls):
        """
        Check whether ``sys.stdin`` and ``sys.stdout`` are UTF-8 encoded.

        Return ``True`` on failure and ``False`` on success.

        :rtype: bool
        """
        is_in_utf8 = True
        is_out_utf8 = True
        if sys.stdin.encoding not in ["UTF-8", "UTF8"]:
            is_in_utf8 = False
        if sys.stdout.encoding not in ["UTF-8", "UTF8"]:
            is_out_utf8 = False
        if (is_in_utf8) and (is_out_utf8):
            gf.print_success(u"shell encoding OK")
        else:
            gf.print_warning(u"shell encoding WARNING")
            if not is_in_utf8:
                gf.print_warning(u"  The default input encoding of your shell is not UTF-8")
            if not is_out_utf8:
                gf.print_warning(u"  The default output encoding of your shell is not UTF-8")
            gf.print_info(u"  If you plan to use aeneas on the command line,")
            if gf.is_posix():
                gf.print_info(u"  you might want to 'export PYTHONIOENCODING=UTF-8' in your shell")
            else:
                gf.print_info(u"  you might want to 'set PYTHONIOENCODING=UTF-8' in your shell")
            return True
        return False

    @classmethod
    def check_ffprobe(cls):
        """
        Check whether ``ffprobe`` can be called.

        Return ``True`` on failure and ``False`` on success.

        :rtype: bool
        """
        try:
            from aeneas.ffprobewrapper import FFPROBEWrapper
            file_path = gf.absolute_path(u"tools/res/audio.mp3", __file__)
            prober = FFPROBEWrapper()
            properties = prober.read_properties(file_path)
            gf.print_success(u"ffprobe        OK")
            return False
        except:
            pass
        gf.print_error(u"ffprobe        ERROR")
        gf.print_info(u"  Please make sure you have ffprobe installed correctly")
        gf.print_info(u"  (usually it is provided by the ffmpeg installer)")
        gf.print_info(u"  and that its path is in your PATH environment variable")
        return True

    @classmethod
    def check_ffmpeg(cls):
        """
        Check whether ``ffmpeg`` can be called.

        Return ``True`` on failure and ``False`` on success.

        :rtype: bool
        """
        try:
            from aeneas.ffmpegwrapper import FFMPEGWrapper
            input_file_path = gf.absolute_path(u"tools/res/audio.mp3", __file__)
            handler, output_file_path = gf.tmp_file(suffix=u".wav")
            converter = FFMPEGWrapper()
            result = converter.convert(input_file_path, output_file_path)
            gf.delete_file(handler, output_file_path)
            if result:
                gf.print_success(u"ffmpeg         OK")
                return False
        except:
            pass
        gf.print_error(u"ffmpeg         ERROR")
        gf.print_info(u"  Please make sure you have ffmpeg installed correctly")
        gf.print_info(u"  and that its path is in your PATH environment variable")
        return True

    @classmethod
    def check_espeak(cls):
        """
        Check whether ``espeak`` can be called.

        Return ``True`` on failure and ``False`` on success.

        :rtype: bool
        """
        try:
            from aeneas.textfile import TextFile
            from aeneas.textfile import TextFragment
            from aeneas.ttswrappers.espeakttswrapper import ESPEAKTTSWrapper
            text = u"From fairest creatures we desire increase,"
            text_file = TextFile()
            text_file.add_fragment(TextFragment(language=u"eng", lines=[text], filtered_lines=[text]))
            handler, output_file_path = gf.tmp_file(suffix=u".wav")
            ESPEAKTTSWrapper().synthesize_multiple(text_file, output_file_path)
            gf.delete_file(handler, output_file_path)
            gf.print_success(u"espeak         OK")
            return False
        except:
            pass
        gf.print_error(u"espeak         ERROR")
        gf.print_info(u"  Please make sure you have espeak installed correctly")
        gf.print_info(u"  and that its path is in your PATH environment variable")
        gf.print_info(u"  You might also want to check that the espeak-data directory")
        gf.print_info(u"  is set up correctly, for example, it has the correct permissions")
        return True

    @classmethod
    def check_tools(cls):
        """
        Check whether ``aeneas.tools.*`` can be imported.

        Return ``True`` on failure and ``False`` on success.

        :rtype: bool
        """
        try:
            from aeneas.tools.convert_syncmap import ConvertSyncMapCLI
            # disabling this check, as it requires the optional dependency pafy
            # COMMENTED from aeneas.tools.download import DownloadCLI
            from aeneas.tools.execute_job import ExecuteJobCLI
            from aeneas.tools.execute_task import ExecuteTaskCLI
            from aeneas.tools.extract_mfcc import ExtractMFCCCLI
            from aeneas.tools.ffmpeg_wrapper import FFMPEGWrapperCLI
            from aeneas.tools.ffprobe_wrapper import FFPROBEWrapperCLI
            # disabling this check, as it requires the optional dependency Pillow
            # COMMENTED from aeneas.tools.plot_waveform import PlotWaveformCLI
            from aeneas.tools.read_audio import ReadAudioCLI
            from aeneas.tools.read_text import ReadTextCLI
            from aeneas.tools.run_sd import RunSDCLI
            from aeneas.tools.run_vad import RunVADCLI
            from aeneas.tools.synthesize_text import SynthesizeTextCLI
            from aeneas.tools.validate import ValidateCLI
            gf.print_success(u"aeneas.tools   OK")
            return False
        except:
            pass
        gf.print_error(u"aeneas.tools   ERROR")
        gf.print_info(u"  Unable to import one or more aeneas.tools")
        gf.print_info(u"  Please check that you installed aeneas properly")
        return True

    @classmethod
    def check_cdtw(cls):
        """
        Check whether Python C extension ``cdtw`` can be imported.

        Return ``True`` on failure and ``False`` on success.

        :rtype: bool
        """
        if gf.can_run_c_extension("cdtw"):
            gf.print_success(u"aeneas.cdtw    AVAILABLE")
            return False
        gf.print_warning(u"aeneas.cdtw    NOT AVAILABLE")
        gf.print_info(u"  You can still run aeneas but it will be significantly slower")
        gf.print_info(u"  Please refer to the installation documentation for details")
        return True

    @classmethod
    def check_cmfcc(cls):
        """
        Check whether Python C extension ``cmfcc`` can be imported.

        Return ``True`` on failure and ``False`` on success.

        :rtype: bool
        """
        if gf.can_run_c_extension("cmfcc"):
            gf.print_success(u"aeneas.cmfcc   AVAILABLE")
            return False
        gf.print_warning(u"aeneas.cmfcc   NOT AVAILABLE")
        gf.print_info(u"  You can still run aeneas but it will be significantly slower")
        gf.print_info(u"  Please refer to the installation documentation for details")
        return True

    @classmethod
    def check_cew(cls):
        """
        Check whether Python C extension ``cew`` can be imported.

        Return ``True`` on failure and ``False`` on success.

        :rtype: bool
        """
        if gf.can_run_c_extension("cew"):
            gf.print_success(u"aeneas.cew     AVAILABLE")
            return False
        gf.print_warning(u"aeneas.cew     NOT AVAILABLE")
        gf.print_info(u"  You can still run aeneas but it will be a bit slower")
        gf.print_info(u"  Please refer to the installation documentation for details")
        return True

    @classmethod
    def check_all(cls, tools=True, encoding=True, c_ext=True):
        """
        Perform all checks.

        Return a tuple of booleans ``(errors, warnings, c_ext_warnings)``.

        :param bool tools: if ``True``, check aeneas tools
        :param bool encoding: if ``True``, check shell encoding
        :param bool c_ext: if ``True``, check Python C extensions
        :rtype: (bool, bool, bool)
        """
        # errors are fatal
        if cls.check_ffprobe():
            return (True, False, False)
        if cls.check_ffmpeg():
            return (True, False, False)
        if cls.check_espeak():
            return (True, False, False)
        if (tools) and (cls.check_tools()):
            return (True, False, False)
        # warnings are non-fatal
        warnings = False
        c_ext_warnings = False
        if encoding:
            warnings = cls.check_shell_encoding()
        if c_ext:
            # we do not want lazy evaluation
            c_ext_warnings = cls.check_cdtw() or c_ext_warnings
            c_ext_warnings = cls.check_cmfcc() or c_ext_warnings
            c_ext_warnings = cls.check_cew() or c_ext_warnings
        # return results
        return (False, warnings, c_ext_warnings)


def main():
    errors, warnings, c_ext_warnings = Diagnostics.check_all()
    if errors:
        sys.exit(1)
    if c_ext_warnings:
        gf.print_warning(u"All required dependencies are met but at least one Python C extension is not available")
        sys.exit(2)
    else:
        gf.print_success(u"All required dependencies are met and all available Python C extensions are working")
        sys.exit(0)


if __name__ == '__main__':
    main()
