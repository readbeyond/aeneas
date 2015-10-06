#!/usr/bin/env python
# coding=utf-8

"""
Download files from various Web sources.
"""

import tempfile

from aeneas.logger import Logger
import aeneas.globalfunctions as gf

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.2.1"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

class Downloader(object):
    """
    Download files from various Web sources.

    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = "Downloader"

    def __init__(self, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def audio_from_youtube(
            self,
            source_url,
            output_file_path=None,
            best_audio=True
    ):
        """
        Download an audio stream from a YouTube video,
        and save it to file.

        You need to have installed the Python module ``pafy``
        (``$ pip install pafy``).

        Return the path of the created file.

        :param source_url: the URL of the YouTube video
        :type  soruce_url: string (url)
        :param best_audio: if ``True``, download the best audio stream available;
                           if ``False``, download the worst (smallest) one.
        :type  best_audio: bool
        :param output_file_path: the path where the downloaded audio should be saved;
                                 if ``None``, create a temporary file
        :type  output_file_path: string (path)
        :rtype: string (path)

        :raise ImportError: if ``pafy`` is not installed
        :raise IOError: if ``output_file_path`` cannot be written
        """
        try:
            import pafy
        except ImportError as exc:
            self._log("pafy is not installed", Logger.CRITICAL)
            raise exc

        output_path = output_file_path
        if output_file_path is None:
            self._log("output_path is None: creating temp file")
            handler, output_path = tempfile.mkstemp(
                #suffix=".wav",
                dir=gf.custom_tmp_dir()
            )
        else:
            if not gf.file_can_be_written(output_path):
                self._log(["Path '%s' cannot be written (wrong permissions?)", output_path], Logger.CRITICAL)
                raise IOError("Path '%s' cannot be written (wrong permissions?)" % output_path)

        video = pafy.new(source_url)
        (audiostream, extension) = self._select_audiostream(
            video.audiostreams,
            best_audio
        )
        if output_file_path is None:
            gf.delete_file(handler, output_path)
            output_path += "." + extension

        self._log(["output_path is '%s'", output_path])
        self._log("Downloading...")
        audiostream.download(filepath=output_path, quiet=True)
        self._log("Downloading... done")

        return output_path

    def _select_audiostream(self, audiostreams, best_audio=True):
        """ Select the largest or the smallest audiostream """
        tmp = []
        i = 0
        for audio in audiostreams:
            tmp.append([audio.get_filesize(), audio.bitrate, audio.extension, i])
            i += 1
        tmp = sorted(tmp)
        self._log("Audiostreams:")
        for audio in tmp:
            self._log("  " + str(audio))
        if best_audio:
            self._log("Selecting largest audiostream")
            selected = tmp[-1]
        else:
            self._log("Selecting smallest audiostream")
            selected = tmp[0]
        return (audiostreams[selected[3]], selected[2])



