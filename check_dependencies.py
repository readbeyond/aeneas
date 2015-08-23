#!/usr/bin/env python
# coding=utf-8

"""
Check dependencies
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
__version__ = "1.1.1"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

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
    on_info("Test 1/6 (import)...")
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
        on_error("  4. scikits.audiolab")
    except:
        pass
    return False

def step2():
    on_info("Test 2/6 (ffprobe)...")
    try:
        on_info("  Trying to call ffprobe...")
        from aeneas.ffprobewrapper import FFPROBEWrapper
        file_path = get_abs_path("aeneas/tests/res/container/job/assets/p001.mp3")
        prober = FFPROBEWrapper()
        properties = prober.read_properties(file_path)
        on_info("  Trying to call ffprobe... succeeded.")
        return True
    except:
        on_error("  Unable to call ffprobe.")
        on_error("  Please make sure you have ffprobe installed correctly and that it is in your $PATH.")
    return False

def step3():
    on_info("Test 3/6 (ffmpeg)...")
    try:
        on_info("  Trying to call ffmpeg...")
        from aeneas.ffmpegwrapper import FFMPEGWrapper
        input_file_path = get_abs_path("aeneas/tests/res/container/job/assets/p001.mp3")
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        converter = FFMPEGWrapper()
        result = converter.convert(input_file_path, output_file_path)
        os.close(handler)
        os.remove(output_file_path)
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
    on_info("Test 4/6 (espeak)...")
    try:
        on_info("  Trying to call espeak...")
        from aeneas.espeakwrapper import ESPEAKWrapper
        from aeneas.language import Language
        text = u"From fairest creatures we desire increase,"
        language = Language.EN
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        espeak = ESPEAKWrapper()
        result = espeak.synthesize(text, language, output_file_path)
        os.close(handler)
        os.remove(output_file_path)
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
    on_info("Test 5/6 (cdtw)...")
    try:
        import aeneas.cdtw
        on_info("  Python C Extension cdtw correctly loaded")
        return True
    except:
        on_warning("  Unable to load Python C Extension cdtw")
        on_warning("  You can still run aeneas, but it will be slower")
        on_warning("  Try running \"bash compile_c_extensions.sh\" to compile the cdtw module")
    return False

def stepC2():
    on_info("Test 6/6 (cmfcc)...")
    try:
        import aeneas.cmfcc
        on_info("  Python C Extension cmfcc correctly loaded")
        return True
    except:
        on_warning("  Unable to load Python C Extension cmfcc")
        on_warning("  You can still run aeneas, but it will be slower")
        on_warning("  Try running \"bash compile_c_extensions.sh\" to compile the cmfcc module")
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
   
    if has_cdtw and has_cmfcc:
        on_info("Congratulations, all dependencies are met and C extensions are available.")
    else:
        on_warning("All dependencies are met, but C extensions are not available.")
        on_warning("Try running \"bash compile_c_extensions.sh\" to compile the C extensions.")
    
    on_info("Enjoy running aeneas!")



if __name__ == '__main__':
    main()



