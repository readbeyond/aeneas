/*

Python C Extension for computing the DTW

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

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>

#include "cdtw_func.h"

#ifndef NPY_INFINITY
#define NPY_INFINITY DBL_MAX
#endif

#define MOVE0 0 // up
#define MOVE1 1 // left
#define MOVE2 2 // up and left

// return the max(0, center_j - half_delta)
uint32_t _nonnegative_difference(uint32_t center_j, uint32_t half_delta) {
    if (half_delta > center_j) {
        return 0;
    }
    return center_j - half_delta;
}

// return the argmin of the three arguments
unsigned int _three_way_argmin(const double cost0, const double cost1, const double cost2) {
    if ((cost0 <= cost1) && (cost0 <= cost2)) {
        return MOVE0;
    }
    if (cost1 <= cost2) {
        return MOVE1;
    }
    return MOVE2;
}

// return the min of three arguments
double _three_way_min(const double cost0, const double cost1, const double cost2) {
    if ((cost0 <= cost1) && (cost0 <= cost2)) {
        return cost0;
    }
    if (cost1 <= cost2) {
        return cost1;
    }
    return cost2;
}

// copy the row-th row of cost_matrix into buffer
void _copy_cost_matrix_row(const double *cost_matrix_ptr, const uint32_t row, const uint32_t width, double *buffer_ptr) {
    memcpy(buffer_ptr, cost_matrix_ptr + row * width, width * sizeof(double));
}

// appen the given (i, j) cell to the k-th position of the best path
void _append(struct PATH_CELL *best_path_ptr, const uint32_t k, const uint32_t i, const uint32_t j) {
    best_path_ptr[k].i = i;
    best_path_ptr[k].j = j;
}

// reverse the best path
void _reverse(struct PATH_CELL *best_path_ptr, const uint32_t best_path_len) {
    uint32_t a, b, tmp_i, tmp_j;

    // reverse the min path
    for (a = 0; a < best_path_len / 2; ++a) {
        b = best_path_len - 1 - a;
        tmp_i = best_path_ptr[a].i;
        tmp_j = best_path_ptr[a].j;
        best_path_ptr[a].i = best_path_ptr[b].i;
        best_path_ptr[a].j = best_path_ptr[b].j;
        best_path_ptr[b].i = tmp_i;
        best_path_ptr[b].j = tmp_j;
    }
}

// compute the norm2 of the given MFCCs vector
void _compute_norm2(double *mfcc_ptr, const uint32_t mfcc_len, const uint32_t mfcc_size, double *norm2_ptr) {
    uint32_t i, k;
    double v, sum;

    for (i = 0; i < mfcc_len; ++i) {
        sum = 0.0;
        for (k = 0; k < mfcc_size; ++k) {
            v = mfcc_ptr[k * mfcc_len + i];
            sum += v * v;
        }
        norm2_ptr[i] = sqrt(sum);
    }
}

// compute cost matrix from mfcc?
int _compute_cost_matrix(
        double *mfcc1_ptr,          // pointer to the MFCCs of the first wave (2D, l x n)
        double *mfcc2_ptr,          // pointer to the MFCCs of the second wave (2D, l x m)
        const uint32_t delta,       // margin parameter
        double *cost_matrix_ptr,    // pointer to the cost matrix (2D, n x delta)
        uint32_t *centers_ptr,      // pointer to the centers (1D, n); centers[i] = center for the i-th row
        const uint32_t n,           // number of frames (MFCC vectors) of the first wave
        const uint32_t m,           // number of frames (MFCC vectors) of the second wave
        const uint32_t l            // MFCC size
    ) {

    double *norm2_1_ptr, *norm2_2_ptr;
    double sum;
    uint32_t center_j, range_start, range_end;
    uint32_t i, j, k;

    // compute norm2 vectors
    norm2_1_ptr = (double *)calloc(n, sizeof(double));
    norm2_2_ptr = (double *)calloc(m, sizeof(double));
    if ((norm2_1_ptr == NULL) || (norm2_2_ptr == NULL)) {
        return CDTW_FAILURE;
    }
    _compute_norm2(mfcc1_ptr, n, l, norm2_1_ptr);
    _compute_norm2(mfcc2_ptr, m, l, norm2_2_ptr);

    for (i = 0; i < n; ++i) {
        center_j = (int)floor(m * (1.0 * i / n));
        range_start = _nonnegative_difference(center_j, delta / 2);
        range_end = range_start + delta;
        if (range_end > m) {
            range_end = m;
            range_start = range_end - delta;
        }
        centers_ptr[i] = range_start;
        for (j = range_start; j < range_end; ++j) {
            sum = 0.0;
            for (k = 0; k < l; ++k) {
                sum += (mfcc1_ptr[k * n + i] * mfcc2_ptr[k * m + j]);
            }
            cost_matrix_ptr[(i * delta) + (j - range_start)] = 1 - (sum / (norm2_1_ptr[i] * norm2_2_ptr[j]));
        } 
    }

    // deallocate norm2 vectors as they are no longer needed
    free((void *)norm2_1_ptr);
    free((void *)norm2_2_ptr);
    return CDTW_SUCCESS;
}

// compute accumulated cost matrix, not in-place
int _compute_accumulated_cost_matrix(
        const double *cost_matrix_ptr,  // pointer to the cost matrix (2D, n x delta)
        const uint32_t *centers_ptr,    // pointer to the centers (1D, n)
        const uint32_t n,               // number of frames of the first wave
        const uint32_t delta,           // margin parameter
        double *accumulated_cost_matrix_ptr // pointer to the accumulated cost matrix (2D, n x delta)
    ) {
    
    double cost0, cost1, cost2;
    uint32_t current_idx, offset;
    uint32_t i, j;
    
    accumulated_cost_matrix_ptr[0] = cost_matrix_ptr[0];
    for (j = 1; j < delta; ++j) {
        accumulated_cost_matrix_ptr[j] = accumulated_cost_matrix_ptr[j-1] + cost_matrix_ptr[j];
    }
    for (i = 1; i < n; ++i) {
        offset = centers_ptr[i] - centers_ptr[i-1];
        for (j = 0; j < delta; ++j) {
            cost0 = NPY_INFINITY;
            if ((j+offset) < delta) {
                cost0 = accumulated_cost_matrix_ptr[(i-1) * delta + (j+offset)];
            }
            cost1 = NPY_INFINITY;
            if (j > 0) {
                cost1 = accumulated_cost_matrix_ptr[  (i) * delta + (j-1)];
            }
            cost2 = NPY_INFINITY;
            if (((j+offset-1) < delta) && ((j+offset-1) >= 0)) {
                cost2 = accumulated_cost_matrix_ptr[(i-1) * delta + (j+offset-1)];
            }
            current_idx = i * delta + j;
            accumulated_cost_matrix_ptr[current_idx] = cost_matrix_ptr[current_idx] + _three_way_min(cost0, cost1, cost2);
        }
    }
    return CDTW_SUCCESS;
}

// compute accumulated cost matrix, in-place
// (i.e., this function overwrites cost_matrix with the accumulated cost values)
int _compute_accumulated_cost_matrix_in_place(
        double *cost_matrix_ptr,        // pointer to the cost matrix (2D, n x delta)
        const uint32_t *centers_ptr,    // pointer to the centers (1D, n)
        const uint32_t n,               // number of frames of the first wave
        const uint32_t delta            // margin parameter
    ) {
   
    double *current_row_ptr;
    double cost0, cost1, cost2;
    uint32_t current_idx, offset;
    uint32_t i, j;
    
    // to compute the i-th row of the accumulated cost matrix
    // we only need the i-th row of the cost matrix
    current_row_ptr = (double *)calloc(delta, sizeof(double));
    if (current_row_ptr == NULL) {
        return CDTW_FAILURE;
    }
    
    // copy the first row of cost_matrix_ptr to current row buffer
    _copy_cost_matrix_row(cost_matrix_ptr, 0, delta, current_row_ptr);
    //cost_matrix_ptr[0] = current_row_ptr[0]; // not needed!
    for (j = 1; j < delta; ++j) {
        cost_matrix_ptr[j] = current_row_ptr[j] + cost_matrix_ptr[j-1];
    }
    for (i = 1; i < n; ++i) {
        // copy current row of cost_matrix_ptr (= i-th row of cost_matrix, not accumulated) to current row buffer
        _copy_cost_matrix_row(cost_matrix_ptr, i, delta, current_row_ptr);
        offset = centers_ptr[i] - centers_ptr[i-1];
        for (j = 0; j < delta; ++j) {
            cost0 = NPY_INFINITY;
            if ((j+offset) < delta) {
                cost0 = cost_matrix_ptr[(i-1) * delta + (j+offset)];
            }
            cost1 = NPY_INFINITY;
            if (j > 0) {
                cost1 = cost_matrix_ptr[  (i) * delta + (j-1)];
            }
            cost2 = NPY_INFINITY;
            if (((j+offset-1) < delta) && ((j+offset-1) >= 0)) {
                cost2 = cost_matrix_ptr[(i-1) * delta + (j+offset-1)];
            }
            current_idx = i * delta + j;
            cost_matrix_ptr[current_idx] = current_row_ptr[j] + _three_way_min(cost0, cost1, cost2);
        }
    }
    free((void *)current_row_ptr);
    return CDTW_SUCCESS;
}

// compute best path and return it as a list of (i, j) tuples, from (0,0) to (n-1, delta-1)
int _compute_best_path(
        const double *accumulated_cost_matrix_ptr,  // pointer to the accumulated cost matrix (2D, n x delta)
        const uint32_t *centers_ptr,                // pointer to the centers (1D, n)
        const uint32_t n,                           // number of frames of the first wave
        const uint32_t delta,                       // margin parameter
        struct PATH_CELL **best_path_ptr,           // pointer to the list of cells making the best path
        uint32_t *best_path_len                     // length of the best path
    ) {

    double cost0, cost1, cost2;
    uint32_t argmin, r_j, offset;
    uint32_t i, j, k, max_path_len;

    // allocate space for keeping the best path
    //
    // NOTE: the length of any path is at most
    //       n + (centers[n-1] + delta)
    //       because: n = num rows and centers[n-1] + delta = num cols
    //       but of course it might be much less
    //       allocating statically to avoid reallocations while appending cells
    //      
    max_path_len = n + centers_ptr[n-1] + delta;
    *best_path_ptr = (struct PATH_CELL *)calloc(max_path_len, sizeof(struct PATH_CELL));
    if ((*best_path_ptr) == NULL) {
        return CDTW_FAILURE;
    }

    i = n - 1;
    j = centers_ptr[i] + delta - 1;
    k = 0;
    _append(*best_path_ptr, k++, i, j);

    while ((i > 0) || (j > 0)) {
        if (i == 0) {
            _append(*best_path_ptr, k++, 0, --j);
        } else if (j == 0) {
            _append(*best_path_ptr, k++, --i, j);
        } else {
            offset = centers_ptr[i] - centers_ptr[i-1];
            r_j = j - centers_ptr[i];
            cost0 = NPY_INFINITY;
            if ((r_j+offset) < delta) {
                cost0 = accumulated_cost_matrix_ptr[(i-1) * delta + (r_j+offset)];
            }
            cost1 = NPY_INFINITY;
            if (r_j > 0) {
                cost1 = accumulated_cost_matrix_ptr[  (i) * delta + (r_j-1)];
            }
            cost2 = NPY_INFINITY;
            if ((r_j > 0) && ((r_j+offset-1 < delta) && ((r_j+offset-1) >= 0))) {
                cost2 = accumulated_cost_matrix_ptr[(i-1) * delta + (r_j+offset-1)];
            }
            argmin = _three_way_argmin(cost0, cost1, cost2);
            if (argmin == MOVE0) {
                _append(*best_path_ptr, k++, --i, j);
            } else if (argmin == MOVE1) {
                _append(*best_path_ptr, k++, i, --j);
            } else {
                _append(*best_path_ptr, k++, --i, --j);
            }
        }
    }

    // k holds the number of cells in the best path
    *best_path_len = k;

    // reverse the path
    _reverse(*best_path_ptr, k);
    return CDTW_SUCCESS;
}



