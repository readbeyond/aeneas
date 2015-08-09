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
    Copyright 2013-2015, ReadBeyond Srl (www.readbeyond.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.0.4"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

def on_error(msg):
    print "[ERRO] %s" % msg

def on_info(msg):
    print "[INFO] %s" % msg

def get_abs_path(rel_path):
    file_dir = os.path.dirname(__file__)
    return os.path.join(file_dir, rel_path)

def main():

    on_info("Test 1/4...")
    try:
        on_info("Trying to import package aeneas...")
        import aeneas
        on_info("Trying to import package aeneas... succeeded.")
    except ImportError:
        on_error("Unable to import package aeneas.")
        on_error("Check that you have installed the following Python (2.7.x) packages:")
        on_error("1. BeautifulSoup")
        on_error("2. lxml")
        on_error("3. numpy")
        on_error("4. scikits.audiolab")
        return

    on_info("Test 2/4...")
    try:
        on_info("Trying to call ffprobe...")
        from aeneas.ffprobewrapper import FFPROBEWrapper
        file_path = get_abs_path("aeneas/tests/res/container/job/assets/p001.mp3")
        prober = FFPROBEWrapper()
        properties = prober.read_properties(file_path)
        on_info("Trying to call ffprobe... succeeded.")
    except:
        on_error("Unable to call ffprobe.")
        on_error("Please make sure you have ffprobe installed correctly and that it is in your $PATH.")
        return

    on_info("Test 3/4...")
    try:
        on_info("Trying to call ffmpeg...")
        from aeneas.ffmpegwrapper import FFMPEGWrapper
        input_file_path = get_abs_path("aeneas/tests/res/container/job/assets/p001.mp3")
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        converter = FFMPEGWrapper()
        result = converter.convert(input_file_path, output_file_path)
        os.close(handler)
        os.remove(output_file_path)
        if not result:
            on_error("Unable to call ffmpeg.")
            on_error("Please make sure you have ffmpeg installed correctly and that it is in your $PATH.")
            return
        on_info("Trying to call ffmpeg... succeeded.")
    except:
        on_error("Unable to call ffmpeg.")
        on_error("Please make sure you have ffmpeg installed correctly and that it is in your $PATH.")
        return

    on_info("Test 4/4...")
    try:
        on_info("Trying to call espeak...")
        from aeneas.espeakwrapper import ESPEAKWrapper
        from aeneas.language import Language
        text = u"From fairest creatures we desire increase,"
        language = Language.EN
        handler, output_file_path = tempfile.mkstemp(suffix=".wav")
        espeak = ESPEAKWrapper()
        result = espeak.synthesize(text, language, output_file_path)
        os.close(handler)
        os.remove(output_file_path)
        if not result:
            on_error("Unable to call espeak.")
            on_error("Please make sure you have espeak installed correctly and that it is in your $PATH.")
            return
        on_info("Trying to call espeak... succeeded.")
    except:
        on_error("Unable to call espeak.")
        on_error("Please make sure you have espeak installed correctly and that it is in your $PATH.")
        return

    on_info("Congratulations, all dependencies are met.")
    on_info("Enjoy running aeneas!")



if __name__ == '__main__':
    main()



