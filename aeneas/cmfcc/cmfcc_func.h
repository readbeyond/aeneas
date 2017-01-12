/*

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2017, Alberto Pettarin (www.albertopettarin.it)
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

Python C Extension for computing the MFCCs from a WAVE mono file.

*/

#include "../cint/cint.h"

#define CMFCC_SUCCESS 0
#define CMFCC_FAILURE 1

// compute MFCC from data loaded in RAM
int compute_mfcc_from_data(
    double *data_ptr,
    const uint32_t data_length,
    const uint32_t sample_rate,
    const uint32_t filter_bank_size,
    const uint32_t mfcc_size,
    const uint32_t fft_order,
    const double lower_frequency,
    const double upper_frequency,
    const double emphasis_factor,
    const double window_length,
    const double window_shift,
    double **mfcc_ptr,
    uint32_t *mfcc_length
);

// compute MFCC from file on disk
int compute_mfcc_from_file(
    char *audio_file_path,
    const uint32_t filter_bank_size,
    const uint32_t mfcc_size,
    const uint32_t fft_order,
    const double lower_frequency,
    const double upper_frequency,
    const double emphasis_factor,
    const double window_length,
    const double window_shift,
    uint32_t *data_length,
    uint32_t *sample_rate,
    double **mfcc_ptr,
    uint32_t *mfcc_length
);

