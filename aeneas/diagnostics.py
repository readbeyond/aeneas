#!/usr/bin/env python
# coding=utf-8

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

SETUP_COMMAND = u"'python setup.py build_ext --inplace'"

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
            from aeneas.espeakwrapper import ESPEAKWrapper
            text = u"From fairest creatures we desire increase,"
            language = u"eng"
            handler, output_file_path = gf.tmp_file(suffix=u".wav")
            espeak = ESPEAKWrapper()
            result = espeak.synthesize_single(
                text,
                language,
                output_file_path
            )
            gf.delete_file(handler, output_file_path)
            if result:
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
            # disabling this check, as it contains optional dependency pafy
            #from aeneas.tools.download import DownloadCLI
            from aeneas.tools.execute_job import ExecuteJobCLI
            from aeneas.tools.execute_task import ExecuteTaskCLI
            from aeneas.tools.extract_mfcc import ExtractMFCCCLI
            from aeneas.tools.ffmpeg_wrapper import FFMPEGWrapperCLI
            from aeneas.tools.ffprobe_wrapper import FFPROBEWrapperCLI
            # disabling this check, as it contains optional dependency Pillow
            #from aeneas.tools.plot_waveform import PlotWaveformCLI
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
            gf.print_success(u"aeneas.cdtw    COMPILED")
            return False
        gf.print_warning(u"aeneas.cdtw    NOT COMPILED")
        gf.print_info(u"  You can still run aeneas but it will be significantly slower")
        gf.print_info(u"  To compile the cdtw module, run %s" % SETUP_COMMAND)
        return True

    @classmethod
    def check_cmfcc(cls):
        """
        Check whether Python C extension ``cmfcc`` can be imported.

        Return ``True`` on failure and ``False`` on success.

        :rtype: bool
        """
        if gf.can_run_c_extension("cmfcc"):
            gf.print_success(u"aeneas.cmfcc   COMPILED")
            return False
        gf.print_warning(u"aeneas.cmfcc   NOT COMPILED")
        gf.print_info(u"  You can still run aeneas but it will be significantly slower")
        gf.print_info(u"  To compile the cmfcc module, run %s" % SETUP_COMMAND)
        return True

    @classmethod
    def check_cew(cls):
        """
        Check whether Python C extension ``cew`` can be imported.

        Return ``True`` on failure and ``False`` on success.

        For those OSes where ``cew`` is not available,
        print a warning and return ``False`` (success).

        :rtype: bool
        """
        if not gf.is_linux():
            gf.print_warning(u"aeneas.cew     NOT AVAILABLE")
            gf.print_info(u"  The Python C Extension cew is not available for your OS")
            gf.print_info(u"  You can still run aeneas but it will be a bit slower (than Linux)")
            return False
        if gf.can_run_c_extension("cew"):
            gf.print_success(u"aeneas.cew     COMPILED")
            return False
        gf.print_warning(u"aeneas.cew     NOT COMPILED")
        gf.print_info(u"  You can still run aeneas but it will be a bit slower")
        gf.print_info(u"  To compile the cew module, run %s" % SETUP_COMMAND)
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
        gf.print_warning(u"All required dependencies are met but at least one available Python C extension is not compiled")
        sys.exit(2)
    else:
        gf.print_success(u"All required dependencies are met and all available Python C extensions are compiled")
        sys.exit(0)



if __name__ == '__main__':
    main()



