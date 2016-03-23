#!/usr/bin/env python
# coding=utf-8

"""
aeneas.cwave is a Python C extension to read WAVE mono files.

.. function:: cwave.get_audio_info(audio_file_path)

    Read the sample rate and length of the given WAVE mono file.

    The returned tuple ``(sr, length)`` contains
    the sample rate and the number of samples
    of the WAVE file.

    :param string audio_file_path: the path of the WAVE file to be read, UTF-8 encoded
    :rtype: tuple

.. function:: cwave.read_audio_data(audio_file_path, from_sample, num_samples)

    Read audio samples from the given WAVE mono file.

    The returned tuple ``(sr, data)`` contains
    the sample rate of the WAVE file,
    and the samples read as a NumPy 1D array
    of ``float64`` values in ``[-1.0, 1.0]``.

    :param string audio_file_path: the path of the WAVE file to be read, UTF-8 encoded
    :param int from_sample: index of the first sample to be read
    :param int num_samples: number of samples to be read
    :rtype: tuple
"""

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL 3"
__version__ = "1.5.0"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"



