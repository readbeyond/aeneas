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

Python C Extension for computing the DTW

*/

#include "../cint/cint.h"

#define CDTW_SUCCESS 0
#define CDTW_FAILURE 1

struct PATH_CELL {
    uint32_t i;                                 // row index in the virtual full matrix (n x m)
    uint32_t j;                                 // column index in the virtual full matrix (n x m)
};

// compute cost matrix from mfcc?
int _compute_cost_matrix(
    double *mfcc1_ptr,                          // pointer to the MFCCs of the first wave (2D, l x n)
    double *mfcc2_ptr,                          // pointer to the MFCCs of the second wave (2D, l x m)
    const uint32_t delta,                       // margin parameter
    double *cost_matrix_ptr,                    // pointer to the cost matrix (2D, n x delta)
    uint32_t *centers_ptr,                      // pointer to the centers (1D, n); centers[i] = center for the i-th row
    const uint32_t n,                           // number of frames (MFCC vectors) of the first wave
    const uint32_t m,                           // number of frames (MFCC vectors) of the second wave
    const uint32_t l                            // MFCC size
);

// compute accumulated cost matrix, not in-place
int _compute_accumulated_cost_matrix(
    const double *cost_matrix_ptr,              // pointer to the cost matrix (2D, n x delta)
    const uint32_t *centers_ptr,                // pointer to the centers (1D, n)
    const uint32_t n,                           // number of frames of the first wave
    const uint32_t delta,                       // margin parameter
    double *accumulated_cost_matrix_ptr         // pointer to the accumulated cost matrix (2D, n x delta)
);

// compute accumulated cost matrix, in-place
// (i.e., this function overwrites cost_matrix with the accumulated cost values)
int _compute_accumulated_cost_matrix_in_place(
    double *cost_matrix_ptr,                    // pointer to the cost matrix (2D, n x delta)
    const uint32_t *centers_ptr,                // pointer to the centers (1D, n)
    const uint32_t n,                           // number of frames of the first wave
    const uint32_t delta                        // margin parameter
);
   
// compute best path and return it as a list of (i, j) tuples, from (0,0) to (n-1, m-1)
int _compute_best_path(
    const double *accumulated_cost_matrix_ptr,  // pointer to the accumulated cost matrix (2D, n x delta)
    const uint32_t *centers_ptr,                // pointer to the centers (1D, n)
    const uint32_t n,                           // number of frames of the first wave
    const uint32_t delta,                       // margin parameter
    struct PATH_CELL **best_path_ptr,           // pointer to the list of cells making the best path
    uint32_t *best_path_len                     // length of the best path
);



