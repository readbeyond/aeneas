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

* :class:`~aeneas.downloader.Downloader`, which download files from various Web sources.

.. note:: This module requires Python modules ``youtube-dl`` and ``pafy`` (``pip install youtube-dl pafy``).
"""

from __future__ import absolute_import
from __future__ import print_function

from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration
import aeneas.globalfunctions as gf


class Downloader(Loggable):
    """
    Download files from various Web sources.

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    TAG = u"Downloader"

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
        :raises: ImportError: if ``pafy`` is not installed
        :raises: OSError: if ``output_file_path`` cannot be written
        :raises: ValueError: if ``source_url`` is not a valid YouTube URL
        """
        def select_audiostream(audiostreams):
            """ Select the audiostream best matching the given parameters. """
            if preferred_index is not None:
                if preferred_index in range(len(audiostreams)):
                    self.log([u"Selecting audiostream with index %d", preferred_index])
                    return audiostreams[preferred_index]
                else:
                    self.log_warn([u"Audio stream index '%d' not allowed", preferred_index])
                    self.log_warn(u"Ignoring the requested audio stream index")
            # selecting by preferred format
            streams = audiostreams
            if preferred_format is not None:
                self.log([u"Selecting audiostreams by preferred format %s", preferred_format])
                streams = [audiostream for audiostream in streams if audiostream.extension == preferred_format]
                if len(streams) < 1:
                    self.log([u"No audiostream with preferred format %s", preferred_format])
                    streams = audiostreams
            # sort by size
            streams = sorted([(audio.get_filesize(), audio) for audio in streams])
            if largest_audio:
                self.log(u"Selecting largest audiostream")
                selected = streams[-1][1]
            else:
                self.log(u"Selecting smallest audiostream")
                selected = streams[0][1]
            return selected

        try:
            import pafy
        except ImportError as exc:
            self.log_exc(u"Python module pafy is not installed", exc, True, ImportError)

        try:
            video = pafy.new(source_url)
        except (IOError, OSError, ValueError) as exc:
            self.log_exc(u"The specified source URL '%s' is not a valid YouTube URL or you are offline" % (source_url), exc, True, ValueError)

        if not download:
            self.log(u"Returning the list of audio streams")
            return video.audiostreams

        output_path = output_file_path
        if output_file_path is None:
            self.log(u"output_path is None: creating temp file")
            handler, output_path = gf.tmp_file(root=self.rconf[RuntimeConfiguration.TMP_PATH])
        else:
            if not gf.file_can_be_written(output_path):
                self.log_exc(u"Path '%s' cannot be written. Wrong permissions?" % (output_path), None, True, OSError)

        audiostream = select_audiostream(video.audiostreams)
        if output_file_path is None:
            gf.delete_file(handler, output_path)
            output_path += "." + audiostream.extension

        self.log([u"output_path is '%s'", output_path])
        self.log(u"Downloading...")
        audiostream.download(filepath=output_path, quiet=True)
        self.log(u"Downloading... done")
        return output_path
