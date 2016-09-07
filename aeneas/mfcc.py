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

* :class:`~aeneas.mfcc.MFCC`, computing Mel-frequency cepstral coefficients (MFCCs).

This file is a modified version of the ``mfcc.py`` file
by David Huggins-Daines from the CMU Sphinx-III project.
You can find the original file in the ``thirdparty/`` directory.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy

from aeneas.logger import Loggable
from aeneas.runtimeconfiguration import RuntimeConfiguration


class MFCC(Loggable):
    """
    A class for computing Mel-frequency cepstral coefficients (MFCCs).

    :param rconf: a runtime configuration
    :type  rconf: :class:`~aeneas.runtimeconfiguration.RuntimeConfiguration`
    :param logger: the logger object
    :type  logger: :class:`~aeneas.logger.Logger`
    """

    CUTOFF = 0.00001
    """ Cut-off threshold """

    MEL_10 = 2595.0
    """ Base Mel frequency """

    TAG = u"MFCC"

    def __init__(self, rconf=None, logger=None):
        super(MFCC, self).__init__(rconf=rconf, logger=logger)

        # store parameters in local attributes
        self.filter_bank_size = self.rconf[RuntimeConfiguration.MFCC_FILTERS]
        self.mfcc_size = self.rconf[RuntimeConfiguration.MFCC_SIZE]
        self.fft_order = self.rconf[RuntimeConfiguration.MFCC_FFT_ORDER]
        self.lower_frequency = self.rconf[RuntimeConfiguration.MFCC_LOWER_FREQUENCY]
        self.upper_frequency = self.rconf[RuntimeConfiguration.MFCC_UPPER_FREQUENCY]
        self.emphasis_factor = self.rconf[RuntimeConfiguration.MFCC_EMPHASIS_FACTOR]
        self.window_length = self.rconf[RuntimeConfiguration.MFCC_WINDOW_LENGTH]
        self.window_shift = self.rconf[RuntimeConfiguration.MFCC_WINDOW_SHIFT]

        # initialize DCT matrix
        self._create_dct_matrix()

        # initialized later by compute_from_data()
        self.data = None
        self.sample_rate = None
        self.filters = None
        self.hamming_window = None

    @classmethod
    def _hz2mel(cls, frequency):
        """
        Convert the given frequency in Hz to the Mel scale.

        :param float frequency: the Hz frequency to convert
        :rtype: float
        """
        return cls.MEL_10 * math.log10(1.0 + (frequency / 700.0))

    @classmethod
    def _mel2hz(cls, mel):
        """
        Convert the given Mel value to Hz frequency.

        :param float mel: the Mel value to convert
        :rtype: float
        """
        return 700.0 * (10 ** (mel / cls.MEL_10) - 1)

    def _create_dct_matrix(self):
        """
        Create the not-quite-DCT matrix as used by Sphinx,
        and store it in ```self.s2dct```.
        """
        self.s2dct = numpy.zeros((self.mfcc_size, self.filter_bank_size))
        for i in range(0, self.mfcc_size):
            freq = numpy.pi * float(i) / self.filter_bank_size
            self.s2dct[i] = numpy.cos(freq * numpy.arange(0.5, 0.5 + self.filter_bank_size, 1.0, 'float64'))
        self.s2dct[:, 0] *= 0.5
        self.s2dct = self.s2dct.transpose()

    def _create_mel_filter_bank(self):
        """
        Create the Mel filter bank,
        and store it in ``self.filters``.

        Note that it is a function of the audio sample rate,
        so it cannot be created in the class initializer,
        but only later in :func:`aeneas.mfcc.MFCC.compute_from_data`.
        """
        self.filters = numpy.zeros((1 + (self.fft_order // 2), self.filter_bank_size), 'd')
        dfreq = float(self.sample_rate) / self.fft_order
        nyquist_frequency = self.sample_rate / 2
        if self.upper_frequency > nyquist_frequency:
            self.log_exc(u"Upper frequency %f exceeds Nyquist frequency %f" % (self.upper_frequency, nyquist_frequency), None, True, ValueError)
        melmax = MFCC._hz2mel(self.upper_frequency)
        melmin = MFCC._hz2mel(self.lower_frequency)
        dmelbw = (melmax - melmin) / (self.filter_bank_size + 1)
        filt_edge = MFCC._mel2hz(melmin + dmelbw * numpy.arange(self.filter_bank_size + 2, dtype='d'))

        # TODO can this code be written more numpy-style?
        #      (the performance loss is negligible, it is just ugly to see)
        for whichfilt in range(0, self.filter_bank_size):
            # int() casts to native int instead of working with numpy.float64
            leftfr = int(round(filt_edge[whichfilt] / dfreq))
            centerfr = int(round(filt_edge[whichfilt + 1] / dfreq))
            rightfr = int(round(filt_edge[whichfilt + 2] / dfreq))
            fwidth = (rightfr - leftfr) * dfreq
            height = 2.0 / fwidth
            if centerfr != leftfr:
                leftslope = height / (centerfr - leftfr)
            else:
                leftslope = 0
            freq = leftfr + 1
            while freq < centerfr:
                self.filters[freq, whichfilt] = (freq - leftfr) * leftslope
                freq = freq + 1
            # the next if should always be true!
            if freq == centerfr:
                self.filters[freq, whichfilt] = height
                freq = freq + 1
            if centerfr != rightfr:
                rightslope = height / (centerfr - rightfr)
            while freq < rightfr:
                self.filters[freq, whichfilt] = (freq - rightfr) * rightslope
                freq = freq + 1

    def _pre_emphasis(self):
        """
        Pre-emphasize the entire signal at once by self.emphasis_factor,
        overwriting ``self.data``.
        """
        self.data = numpy.append(self.data[0], self.data[1:] - self.emphasis_factor * self.data[:-1])

    def compute_from_data(self, data, sample_rate):
        """
        Compute MFCCs for the given audio data.

        The audio data must be a 1D :class:`numpy.ndarray`,
        that is, it must represent a monoaural (single channel)
        array of ``float64`` values in ``[-1.0, 1.0]``.

        :param data: the audio data
        :type  data: :class:`numpy.ndarray` (1D)
        :param int sample_rate: the sample rate of the audio data, in samples/s (Hz)
        :raises: ValueError: if the data is not a 1D :class:`numpy.ndarray` (i.e., not mono),
                             or if the data is empty
        :raises: ValueError: if the upper frequency defined in the ``rconf`` is
                             larger than the Nyquist frequenct (i.e., half of ``sample_rate``)
        """
        def _process_frame(self, frame):
            """
            Process each frame, returning the log(power()) of it.
            """
            # apply Hamming window
            frame *= self.hamming_window
            # compute RFFT
            fft = numpy.fft.rfft(frame, self.fft_order)
            # equivalent to power = fft.real * fft.real + fft.imag * fft.imag
            power = numpy.square(numpy.absolute(fft))
            #
            # return the log(power()) of the transformed vector
            # v1
            # COMMENTED logspec = numpy.log(numpy.dot(power, self.filters).clip(self.CUTOFF, numpy.inf))
            # COMMENTED return numpy.dot(logspec, self.s2dct) / self.filter_bank_size
            # v2
            return numpy.log(numpy.dot(power, self.filters).clip(self.CUTOFF, numpy.inf))

        if len(data.shape) != 1:
            self.log_exc(u"The audio data must be a 1D numpy array (mono).", None, True, ValueError)
        if len(data) < 1:
            self.log_exc(u"The audio data must not be empty.", None, True, ValueError)

        self.data = data
        self.sample_rate = sample_rate

        # number of samples in the audio
        data_length = len(self.data)

        # frame length in number of samples
        frame_length = int(self.window_length * self.sample_rate)

        # frame length must be at least equal to the FFT order
        frame_length_padded = max(frame_length, self.fft_order)

        # frame shift in number of samples
        frame_shift = int(self.window_shift * self.sample_rate)

        # number of MFCC vectors (one for each frame)
        # this number includes the last shift,
        # where the data will be padded with zeros
        # if the remaining samples are less than frame_length_padded
        number_of_frames = int((1.0 * data_length) / frame_shift)

        # create Hamming window
        self.hamming_window = numpy.hamming(frame_length_padded)

        # build Mel filter bank
        self._create_mel_filter_bank()

        # pre-emphasize the entire audio data
        self._pre_emphasis()

        # allocate the MFCCs matrix
        # v1
        # COMMENTED mfcc = numpy.zeros((number_of_frames, self.mfcc_size), 'float64')
        # v2
        mfcc = numpy.zeros((number_of_frames, self.filter_bank_size), 'float64')

        # compute MFCCs one frame at a time
        for frame_index in range(number_of_frames):
            # COMMENTED print("Computing frame %d / %d" % (frame_index, number_of_frames))

            # get the start and end indices for this frame,
            # do not overrun the data length
            frame_start = frame_index * frame_shift
            frame_end = min(frame_start + frame_length_padded, data_length)

            # frame is zero-padded if the remaining samples
            # are less than its length
            frame = numpy.zeros(frame_length_padded)
            frame[0:(frame_end - frame_start)] = self.data[frame_start:frame_end]

            # process the frame
            mfcc[frame_index] = _process_frame(self, frame)

        # v1
        # COMMENTED return mfcc
        # v2
        # return the dot product with the DCT matrix
        return numpy.dot(mfcc, self.s2dct) / self.filter_bank_size
