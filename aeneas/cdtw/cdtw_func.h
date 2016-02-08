/*

Python C Extension for computing the DTW

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.4.1"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

*/

struct PATH_CELL {
    unsigned int i;
    unsigned int j;
};

// compute cost matrix from mfcc?
void _compute_cost_matrix(
    double *mfcc1_ptr,                      // pointer to the MFCCs of the first wave (2D, l x n)
    double *mfcc2_ptr,                      // pointer to the MFCCs of the second wave (2D, l x m)
    const unsigned int delta,               // margin parameter
    double *cost_matrix_ptr,                // pointer to the cost matrix (2D, n x delta)
    unsigned int *centers_ptr,              // pointer to the centers (1D, n); centers[i] = center for the i-th row; delta/2 <= centers[i] < m - delta/2
    const unsigned int n,                   // number of frames of the first wave
    const unsigned int m,                   // number of frames of the second wave
    const unsigned int l                    // number of MFCCs
);

// compute accumulated cost matrix, not in-place
void _compute_accumulated_cost_matrix(
    double *cost_matrix_ptr,                // pointer to the cost matrix (2D, n x delta)
    unsigned int *centers_ptr,              // pointer to the centers (1D, n)
    const unsigned int n,                   // number of frames of the first wave
    const unsigned int delta,               // margin parameter
    double *accumulated_cost_matrix_ptr     // pointer to the accumulated cost matrix (2D, n x delta)
);

// compute accumulated cost matrix, in-place
// (i.e., this function overwrites cost_matrix with the accumulated cost values)
void _compute_accumulated_cost_matrix_in_place(
    double *cost_matrix_ptr,                // pointer to the cost matrix (2D, n x delta)
    unsigned int *centers_ptr,              // pointer to the centers (1D, n)
    const unsigned int n,                   // number of frames of the first wave
    const unsigned int delta                // margin parameter
);
   
// compute best path and return it as a list of (i, j) tuples, from (0,0) to (n-1, delta-1)
void _compute_best_path(
    double *accumulated_cost_matrix_ptr,    // pointer to the accumulated cost matrix (2D, n x delta)
    unsigned int *centers_ptr,              // pointer to the centers (1D, n)
    const unsigned int n,                   // number of frames of the first wave
    const unsigned int delta,               // margin parameter
    struct PATH_CELL **best_path_ptr,       // pointer to the list of cells making the best path
    unsigned int *best_path_len             // length of the best path
);



