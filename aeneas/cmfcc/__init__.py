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
aeneas.cmfcc is a Python C Extension for computing the MFCCs from a WAVE mono file.

.. function:: cmfcc.compute_from_data(data, sample_rate, filter_bank_size, mfcc_size, fft_order, lower_frequency, upper_frequency, emphasis_factor, window_length, window_shift)

    Compute MFCCs for a given WAVE mono file,
    passed as a NumPy 1D array of ``float64`` values in ``[-1.0, 1.0]``.

    The returned tuple ``(mfcc, length, sr)`` contains
    the MFCCs as a NumPy 2D matrix of shape ``(n, mfcc_size)``,
    and the number of samples and sample rate of the WAVE file.

    The last two elements ``length`` and ``sr``
    are returned to make the signature of this function
    consistent with that of function :func:`cmfcc.compute_from_file`.

    :param data: the audio data
    :type  data: :class:`numpy.ndarray` (1D)
    :param int sample_rate: the audio sample rate
    :param int filter_bank_size: the number of Mel filters
    :param int mfcc_size: the number of MFCC coefficients
    :param int fft_order: the order of the FFT
    :param float lower_frequency: the lower frequency to cut, in Hz
    :param float upper_frequency: the upper frequency to cut, in Hz
    :param float emphasis_factor: the pre-emphasis factor
    :param float window_length: the length of the MFCC window, in seconds
    :param float window_shift: the shift of the MFCC window, in seconds
    :rtype: tuple

.. function:: cmfcc.compute_from_file(audio_file_path, filter_bank_size, mfcc_size, fft_order, lower_frequency, upper_frequency, emphasis_factor, window_length, window_shift)

    Compute MFCCs for a given WAVE mono file,
    passed as a file path on disk.

    The returned tuple ``(mfcc, length, sr)`` contains
    the MFCCs as a NumPy 2D matrix of shape ``(n, mfcc_size)``,
    and the number of samples and sample rate of the WAVE file.

    :param string audio_file_path: the path of the WAVE file to be created, UTF-8 encoded
    :param int filter_bank_size: the number of Mel filters
    :param int mfcc_size: the number of MFCC coefficients
    :param int fft_order: the order of the FFT
    :param float lower_frequency: the lower frequency to cut, in Hz
    :param float upper_frequency: the upper frequency to cut, in Hz
    :param float emphasis_factor: the pre-emphasis factor
    :param float window_length: the length of the MFCC window, in seconds
    :param float window_shift: the shift of the MFCC window, in seconds
    :rtype: tuple
"""
