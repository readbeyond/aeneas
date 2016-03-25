#!/usr/bin/env python
# coding=utf-8

"""
Download files from various Web sources.
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.logger import Logger
from aeneas.runtimeconfiguration import RuntimeConfiguration
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

class Downloader(object):
    """
    Download files from various Web sources.

    :param rconf: a runtime configuration. Default: ``None``, meaning that
                  default settings will be used.
    :type  rconf: :class:`aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`aeneas.logger.Logger`
    """

    TAG = u"Downloader"

    def __init__(self, rconf=None, logger=None):
        self.logger = logger if logger is not None else Logger()
        self.rconf = rconf if rconf is not None else RuntimeConfiguration()

    def _log(self, message, severity=Logger.DEBUG):
        """ Log """
        self.logger.log(message, severity, self.TAG)

    def audio_from_youtube(
            self,
            source_url,
            download=True,
            output_file_path=None,
            preferred_index=None,
            largest_audio=True,
            preferred_format=None
    ):
        """
        Download an audio stream from a YouTube video,
        and save it to file.

        If ``download`` is ``False``, return the list
        of available audiostreams but do not download.

        Otherwise, download the audio stream best matching
        the provided parameters, as follows.
        If ``preferred_index`` is not ``None``,
        download the audio stream at that index.
        If ``largest_audio`` is ``True``,
        download the largest audiostream;
        otherwise, download the smallest audiostream.
        If ``preferred_format`` is not ``None``,
        download the audiostream having that format.
        The latter option works in combination with ``largest_audio``.

        Return the path of the downloaded file.

        :param string source_url: the URL of the YouTube video
        :param bool download: if ``True``, download the audio stream
                              best matching ``preferred_index`` or ``preferred_format``
                              and ``largest_audio``;
                              if ``False``, return the list of available audio streams
        :param string output_file_path: the path where the downloaded audio should be saved;
                                        if ``None``, create a temporary file
        :param int preferred_index: preferably download this audio stream
        :param bool largest_audio: if ``True``, download the largest audio stream available;
                                   if ``False``, download the smallest one.
        :param string preferred_format: preferably download this audio format

        :rtype: string or list of pafy audio streams

        :raise ImportError: if ``pafy`` is not installed
        :raise OSError: if ``output_file_path`` cannot be written
        :raise ValueError: if ``source_url`` is not a valid YouTube URL
        """
        def select_audiostream(audiostreams):
            """ Select the audiostream best matching the given parameters. """
            if preferred_index is not None:
                if preferred_index in range(len(audiostreams)):
                    self._log([u"Selecting audiostream with index %d", preferred_index])
                    return audiostreams[preferred_index]
                else:
                    self._log([u"Audio stream index %d not allowed", preferred_index], Logger.WARNING)
                    self._log(u"Ignoring the requested audio stream index", Logger.WARNING)
            # selecting by preferred format
            streams = audiostreams
            if preferred_format is not None:
                self._log([u"Selecting audiostreams by preferred format %s", preferred_format])
                streams = [audiostream for audiostream in streams if audiostream.extension == preferred_format]
                if len(streams) < 1:
                    self._log([u"No audiostream with preferred format %s", preferred_format])
                    streams = audiostreams
            # sort by size
            streams = sorted([(audio.get_filesize(), audio) for audio in streams])
            if largest_audio:
                self._log(u"Selecting largest audiostream")
                selected = streams[-1][1]
            else:
                self._log(u"Selecting smallest audiostream")
                selected = streams[0][1]
            return selected

        try:
            import pafy
        except ImportError as exc:
            self._log(u"Python module pafy is not installed", Logger.CRITICAL)
            raise exc

        try:
            video = pafy.new(source_url)
        except (IOError, OSError, ValueError) as exc:
            self._log([u"The specified source URL '%s' is not a valid YouTube URL", source_url], Logger.CRITICAL)
            raise ValueError("The specified source URL is not a valid YouTube URL")

        if not download:
            self._log(u"Returning the list of audio streams")
            return video.audiostreams

        output_path = output_file_path
        if output_file_path is None:
            self._log(u"output_path is None: creating temp file")
            handler, output_path = gf.tmp_file(root=self.rconf[RuntimeConfiguration.TMP_PATH])
        else:
            if not gf.file_can_be_written(output_path):
                self._log([u"Path '%s' cannot be written (wrong permissions?)", output_path], Logger.CRITICAL)
                raise OSError("Path '%s' cannot be written (wrong permissions?)" % output_path)

        audiostream = select_audiostream(video.audiostreams)
        if output_file_path is None:
            gf.delete_file(handler, output_path)
            output_path += "." + audiostream.extension

        self._log([u"output_path is '%s'", output_path])
        self._log(u"Downloading...")
        audiostream.download(filepath=output_path, quiet=True)
        self._log(u"Downloading... done")
        return output_path





