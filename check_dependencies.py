#!/usr/bin/env python
# coding=utf-8

"""
Check whether the setup of aeneas was successful.

Requires the audio file aeneas/tools/res/audio.mp3

Running this script makes sense only
if you git-cloned the original GitHub repository
and/or if you are interested in contributing to the
development of aeneas.
"""

import os
import tempfile

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.3.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

SETUP_COMMAND = "'python setup.py build_ext --inplace'"

def on_error(msg):
    print "[ERRO] %s" % msg

def on_info(msg):
    print "[INFO] %s" % msg

def on_warning(msg):
    print "[WARN] %s" % msg

def get_abs_path(rel_path):
    file_dir = os.path.dirname(__file__)
    return os.path.join(file_dir, rel_path)

def step1():
    on_info("Test 1/7 (import)...")
    try:
        on_info("  Trying to import package aeneas...")
        import aeneas
        on_info("  Trying to import package aeneas... succeeded.")
        return True
    except ImportError:
        on_error("  Unable to import package aeneas.")
        on_error("  Check that you have installed the following Python (2.7.x) packages:")
        on_error("  1. BeautifulSoup")
        on_error("  2. lxml")
        on_error("  3. numpy")
    except:
        pass
    return False

def step2():
    on_info("Test 2/7 (ffprobe)...")
    try:
        on_info("  Trying to call ffprobe...")
        from aeneas.ffprobewrapper import FFPROBEWrapper
        file_path = get_abs_path("aeneas/tools/res/audio.mp3")
        prober = FFPROBEWrapper()
        properties = prober.read_properties(file_path)
        on_info("  Trying to call ffprobe... succeeded.")
        return True
    except:
        on_error("  Unable to call ffprobe.")
        on_error("  Please make sure you have ffprobe installed correctly and that it is in your $PATH.")
    return False

def step3():
    on_info("Test 3/7 (ffmpeg)...")
    try:
        on_info("  Trying to call ffmpeg...")
        from aeneas.ffmpegwrapper import FFMPEGWrapper
        import aeneas.globalfunctions as gf
        input_file_path = get_abs_path("aeneas/tools/res/audio.mp3")
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        converter = FFMPEGWrapper()
        result = converter.convert(input_file_path, output_file_path)
        gf.delete_file(handler, output_file_path)
        if result:
            on_info("  Trying to call ffmpeg... succeeded.")
            return True
        else:
            on_error("  Unable to call ffmpeg.")
            on_error("  Please make sure you have ffmpeg installed correctly and that it is in your $PATH.")
    except:
        on_error("  Unable to call ffmpeg.")
        on_error("  Please make sure you have ffmpeg installed correctly and that it is in your $PATH.")
    return False

def step4():
    on_info("Test 4/7 (espeak)...")
    try:
        on_info("  Trying to call espeak...")
        from aeneas.espeakwrapper import ESPEAKWrapper
        from aeneas.language import Language
        import aeneas.globalfunctions as gf
        text = u"From fairest creatures we desire increase,"
        language = Language.EN
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        espeak = ESPEAKWrapper()
        result = espeak.synthesize_single(
            text,
            language,
            output_file_path,
            force_pure_python=True
        )
        gf.delete_file(handler, output_file_path)
        if result:
            on_info("  Trying to call espeak... succeeded.")
            return True
        else:
            on_error("  Unable to call espeak.")
            on_error("  Please make sure you have espeak installed correctly and that it is in your $PATH.")
    except:
        on_error("  Unable to call espeak.")
        on_error("  Please make sure you have espeak installed correctly and that it is in your $PATH.")
    return False

def stepC1():
    on_info("Test 5/7 (cdtw)...")
    try:
        import aeneas.cdtw
        on_info("  Python C Extension cdtw correctly loaded")
        return True
    except:
        on_warning("  Unable to load Python C Extension cdtw")
        on_warning("  You can still run aeneas, but it will much slower")
        on_warning("  Try running %s to compile the cdtw module" % SETUP_COMMAND)
    return False

def stepC2():
    on_info("Test 6/7 (cmfcc)...")
    try:
        import aeneas.cmfcc
        on_info("  Python C Extension cmfcc correctly loaded")
        return True
    except:
        on_warning("  Unable to load Python C Extension cmfcc")
        on_warning("  You can still run aeneas, but it will be a bit slower")
        on_warning("  Try running %s to compile the cmfcc module" % SETUP_COMMAND)
    return False

def stepC3():
    on_info("Test 7/7 (cew)...")
    if not ((os.name == "posix") and (os.uname()[0] == "Linux")):
        on_info("  Python C Extension cew is not available for your OS")
        on_info("  You can still run aeneas, but it will be a bit slower than Linux")
        return True 
    try:
        import aeneas.cew
        on_info("  Python C Extension cew correctly loaded")
        return True
    except:
        on_warning("  Unable to load Python C Extension cew")
        on_warning("  You can still run aeneas, but it will be a bit slower")
        on_warning("  Try running %s to compile the cew module" % SETUP_COMMAND)
    return False

def main():
    if not step1():
        return

    if not step2():
        return

    if not step3():
        return

    if not step4():
        return

    has_cdtw = stepC1()
    has_cmfcc = stepC2()
    has_cew = stepC3()

    if has_cdtw and has_cmfcc and has_cew:
        on_info("Congratulations, all dependencies are met and core C extensions are available.")
    else:
        on_warning("All dependencies are met, but at least one core C extension has not been compiled.")
        on_warning("Try running %s to compile the C extensions." % SETUP_COMMAND)

    on_info("Enjoy running aeneas!")



if __name__ == '__main__':
    main()



