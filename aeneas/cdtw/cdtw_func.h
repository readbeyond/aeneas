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

#include "cint.h"

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



