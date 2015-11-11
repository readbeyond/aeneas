/*

Python C Extension for computing the DTW

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.3.2"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <numpy/npy_math.h>

// return the max of the given arguments
static int _max(int a, int b) {
    if (a > b) {
        return a;
    }
    return b;
}

// return the argmin of the three arguments
static int _three_way_argmin(double cost0, double cost1, double cost2) {
    if ((cost0 <= cost1) && (cost0 <= cost2)) {
        return 0;
    }
    if (cost1 <= cost2) {
        return 1;
    }
    return 2;
}

// return the min of three arguments
static double _three_way_min(double cost0, double cost1, double cost2) {
    if ((cost0 <= cost1) && (cost0 <= cost2)) {
        return cost0;
    }
    if (cost1 <= cost2) {
        return cost1;
    }
    return cost2;
}

// append a new tuple (i, j) to the given list
static void _append(PyObject *list, int i, int j) {
    PyObject *tuple;
    
    tuple = PyTuple_New(2);
    // PyTuple_SetItem steals a reference, so no PyDECREF is needed
    PyTuple_SetItem(tuple, 0, Py_BuildValue("i", i));
    PyTuple_SetItem(tuple, 1, Py_BuildValue("i", j));
    PyList_Append(list, tuple);
}

// compute cost matrix from mfcc? and norm2_?
static void _compute_cost_matrix(
        double *mfcc1_ptr,          // pointer to the MFCCs of the first wave (2D, l x n)
        double *mfcc2_ptr,          // pointer to the MFCCs of the second wave (2D, l x m)
        double *norm2_1_ptr,        // pointer to the norm2 of the first wave (1D, n)
        double *norm2_2_ptr,        // pointer to the norm2 of the second wave (1D, m)
        int delta,                  // margin parameter
        double *cost_matrix_ptr,    // pointer to the cost matrix (2D, n x delta)
        int *centers_ptr,           // pointer to the centers (1D, n); centers[i] = center for the i-th row; delta/2 <= centers[i] < m - delta/2
        int n,                      // number of frames of the first wave
        int m,                      // number of frames of the second wave
        int l                       // number of MFCCs
    ) {

    double sum;
    int center_j, range_start, range_end;
    int i, j, k;
    for (i = 0; i < n; ++i) {
        center_j = (int)floor(m * (1.0 * i / n));
        range_start = _max(0, center_j - (delta / 2));
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
}

// compute accumulated cost matrix, not in-place
static void _compute_accumulated_cost_matrix(
        double *cost_matrix_ptr,                // pointer to the cost matrix (2D, n x delta)
        int *centers_ptr,                       // pointer to the centers (1D, n)
        int n,                                  // number of frames of the first wave
        int delta,                              // margin parameter
        double *accumulated_cost_matrix_ptr     // pointer to the accumulated cost matrix (2D, n x delta)
    ) {
    
    double cost0, cost1, cost2;
    int current_idx, offset;
    int i, j;
    
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
}

/*
// print the first max elements of the row-th row of matrix
// use this function for debugging purposes only!
static void _print_row(double *ptr, int row, int width, int max) {
    int j, offset;

    offset = row * width;
    printf("     ");
    for (j = 0; j < max; ++j) {
        printf(" %.6f", ptr[offset + j]);
    }
    printf("\n");
}
*/

// copy the row-th row of cost_matrix into buffer
static void _copy_cost_matrix_row(double *cost_matrix_ptr, int row, int width, double *buffer_ptr) {
    memcpy(buffer_ptr, cost_matrix_ptr + row * width, width * sizeof(double));
}

// compute accumulated cost matrix, in-place
// (i.e., this function overwrites cost_matrix with the accumulated cost values)
static void _compute_accumulated_cost_matrix_in_place(
        double *cost_matrix_ptr,                // pointer to the cost matrix (2D, n x delta)
        int *centers_ptr,                       // pointer to the centers (1D, n)
        int n,                                  // number of frames of the first wave
        int delta                               // margin parameter
    ) {
   
    double *current_row_ptr;
    double cost0, cost1, cost2;
    int current_idx, offset;
    int i, j;
    
    // to compute the i-th row of the accumulated cost matrix
    // we only need the i-th row of the cost matrix
    current_row_ptr = (double *)malloc(delta * sizeof(double));
    
    // copy the first row of cost_matrix_ptr to current row buffer
    _copy_cost_matrix_row(cost_matrix_ptr, 0, delta, current_row_ptr);
    //cost_matrix_ptr[0] = current_row_ptr[0];
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
}

// compute best path and return it as a list of (i, j) tuples, from (0,0) to (n-1, delta-1)
static void _compute_best_path(
        double *accumulated_cost_matrix_ptr,    // pointer to the accumulated cost matrix (2D, n x delta)
        int *centers_ptr,                       // pointer to the centers (1D, n)
        int n,                                  // number of frames of the first wave
        int delta,                              // margin parameter
        PyObject *best_path_ptr                 // pointer to the list (of tuples) to be returned
    ){

    double cost0, cost1, cost2;
    int argmin, offset;
    int i, j, r_j;

    i = n - 1;
    j = delta - 1 + centers_ptr[i];
    _append(best_path_ptr, i, j);
    while ((i > 0) || (j > 0)) {
        if (i == 0) {
            _append(best_path_ptr, 0, --j);
        } else if (j == 0) {
            _append(best_path_ptr, --i, j);
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
            if (argmin == 0) {
                _append(best_path_ptr, --i, j);
            } else if (argmin == 1) {
                _append(best_path_ptr, i, --j);
            } else {
                _append(best_path_ptr, --i, --j);
            }
        }
    }
    PyList_Reverse(best_path_ptr);
}



// compute the best path "all in one"
// take the PyObject containing the following arguments:
//   - mfcc1:       2D array (l x n) of double, MFCCs of the first wave
//   - mfcc2:       2D array (l x m) of double, MFCCs of the second wave
//   - norm2_1:     1D array (n x 1) of double, norm2 of the first wave
//   - norm2_2:     1D array (m x 1) of double, norm2 of the second wave
//   - delta:       int, the number of frames of margin
// and return the best path as a list of (i, j) tuples, from (0,0) to (n-1, delta-1)
static PyObject *cdtw_compute_best_path(PyObject *self, PyObject *args) {
    PyObject *mfcc1_raw;
    PyObject *mfcc2_raw;
    PyObject *norm2_1_raw;
    PyObject *norm2_2_raw;
    int delta;
 
    PyArrayObject *mfcc1, *mfcc2, *norm2_1, *norm2_2, *cost_matrix, *centers;
    PyObject *best_path_ptr;
    npy_intp cost_matrix_dimensions[2];
    npy_intp centers_dimensions[1];
    double *mfcc1_ptr, *mfcc2_ptr,  *norm2_1_ptr, *norm2_2_ptr, *cost_matrix_ptr;
    int *centers_ptr;
    int l1, l2, n, m;

    // O = object (do not convert or check for errors)
    // i = int
    if (!PyArg_ParseTuple(args, "OOOOi", &mfcc1_raw, &mfcc2_raw, &norm2_1_raw, &norm2_2_raw, &delta)) {
        PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments with the OOOOi mask");
        return NULL;
    }

    // convert to C contiguous array
    mfcc1   = (PyArrayObject *) PyArray_ContiguousFromObject(mfcc1_raw,   PyArray_DOUBLE, 2, 2);
    mfcc2   = (PyArrayObject *) PyArray_ContiguousFromObject(mfcc2_raw,   PyArray_DOUBLE, 2, 2);
    norm2_1 = (PyArrayObject *) PyArray_ContiguousFromObject(norm2_1_raw, PyArray_DOUBLE, 1, 1);
    norm2_2 = (PyArrayObject *) PyArray_ContiguousFromObject(norm2_2_raw, PyArray_DOUBLE, 1, 1);

    // check for conversion errors
    if ((mfcc1 == NULL) || (mfcc2 == NULL) || (norm2_1 == NULL) || (norm2_2 == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Error while converting arguments using PyArray_ContiguousFromObject");
        return NULL;
    }

    // NOTE: if arrived here, the mfcc? and norm2_? have the correct number of dimensions (2 and 1, respectively)
   
    // get the dimensions of the input arguments
    l1 = mfcc1->dimensions[0]; // number of MFCCs in the first wave
    l2 = mfcc2->dimensions[0]; // number of MFCCs in the second wave
    n  = mfcc1->dimensions[1]; // number of frames in the first wave
    m  = mfcc2->dimensions[1]; // number of frames in the second wave

    // check that the number of MFCCs is the same for both waves
    if (l1 != l2) {
        PyErr_SetString(PyExc_ValueError, "The number of MFCCs must be the same for both waves");
        return NULL;
    }

    // check that the norm (1D) arrays have the correct number of elements
    if (norm2_1->dimensions[0] != n) {
        PyErr_SetString(PyExc_ValueError, "The number of columns of mfcc1 must be equal to the number of elements of norm2_1");
        return NULL;
    }
    if (norm2_2->dimensions[0] != m) {
        PyErr_SetString(PyExc_ValueError, "The number of columns of mfcc2 must be equal to the number of elements of norm2_2");
        return NULL;
    }

    // delta cannot be greater than m 
    if (delta > m) {
        delta = m;
    }
    
    // pointer to cost matrix data
    mfcc1_ptr   = (double *)mfcc1->data;
    mfcc2_ptr   = (double *)mfcc2->data;
    norm2_1_ptr = (double *)norm2_1->data;
    norm2_2_ptr = (double *)norm2_2->data;
    
    // create cost matrix object
    cost_matrix_dimensions[0] = n;
    cost_matrix_dimensions[1] = delta;
    cost_matrix = (PyArrayObject *)PyArray_SimpleNew(2, cost_matrix_dimensions, PyArray_DOUBLE);
    cost_matrix_ptr = (double *)cost_matrix->data;

    // create centers object
    centers_dimensions[0] = n;
    centers = (PyArrayObject *)PyArray_SimpleNew(1, centers_dimensions, PyArray_INT32);
    centers_ptr = (int *)centers->data;

    // create best path array of integers
    best_path_ptr = PyList_New(0);

    // actual computation
    _compute_cost_matrix(mfcc1_ptr, mfcc2_ptr, norm2_1_ptr, norm2_2_ptr, delta, cost_matrix_ptr, centers_ptr, n, m, l1);
    _compute_accumulated_cost_matrix_in_place(cost_matrix_ptr, centers_ptr, n, delta);
    _compute_best_path(cost_matrix_ptr, centers_ptr, n, delta, best_path_ptr);

    // decrement reference to local object no longer needed
    Py_DECREF(mfcc1);
    Py_DECREF(mfcc2);
    Py_DECREF(norm2_1);
    Py_DECREF(norm2_2);
    Py_DECREF(cost_matrix);
    Py_DECREF(centers);

    return best_path_ptr;
}



// compute the cost matrix and the corresponding stripe centers 
// take the PyObject containing the following arguments:
//   - mfcc1:       2D array (l x n) of double, MFCCs of the first wave
//   - mfcc2:       2D array (l x m) of double, MFCCs of the second wave
//   - norm2_1:     1D array (n x 1) of double, norm2 of the first wave
//   - norm2_2:     1D array (m x 1) of double, norm2 of the second wave
//   - delta:       int, the number of frames of margin
// and return a tuple (cost_matrix, centers), where
//   - cost_matrix: 2D array (n x delta) of double
//   - centers:     1D array (n x 1) of int, centers[i] is the 0 <= center < m of the stripe at row i
static PyObject *cdtw_compute_cost_matrix_step(PyObject *self, PyObject *args) {
    PyObject *mfcc1_raw;
    PyObject *mfcc2_raw;
    PyObject *norm2_1_raw;
    PyObject *norm2_2_raw;
    int delta;

    PyArrayObject *mfcc1, *mfcc2, *norm2_1, *norm2_2, *cost_matrix, *centers;
    PyObject *tuple;
    npy_intp cost_matrix_dimensions[2];
    npy_intp centers_dimensions[1];
    double *mfcc1_ptr, *mfcc2_ptr,  *norm2_1_ptr, *norm2_2_ptr, *cost_matrix_ptr;
    int *centers_ptr;
    int l1, l2, n, m;
   
    // O = object (do not convert or check for errors)
    // i = int
    if (!PyArg_ParseTuple(args, "OOOOi", &mfcc1_raw, &mfcc2_raw, &norm2_1_raw, &norm2_2_raw, &delta)) {
        PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments with the OOOOi mask");
        return NULL;
    }

    // convert to C contiguous array
    mfcc1   = (PyArrayObject *) PyArray_ContiguousFromObject(mfcc1_raw,   PyArray_DOUBLE, 2, 2);
    mfcc2   = (PyArrayObject *) PyArray_ContiguousFromObject(mfcc2_raw,   PyArray_DOUBLE, 2, 2);
    norm2_1 = (PyArrayObject *) PyArray_ContiguousFromObject(norm2_1_raw, PyArray_DOUBLE, 1, 1);
    norm2_2 = (PyArrayObject *) PyArray_ContiguousFromObject(norm2_2_raw, PyArray_DOUBLE, 1, 1);

    // check for conversion errors
    if ((mfcc1 == NULL) || (mfcc2 == NULL) || (norm2_1 == NULL) || (norm2_2 == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Error while converting arguments using PyArray_ContiguousFromObject");
        return NULL;
    }

    // NOTE: if arrived here, the mfcc? and norm2_? have the correct number of dimensions (2 and 1, respectively)
   
    // get the dimensions of the input arguments
    l1 = mfcc1->dimensions[0]; // number of MFCCs in the first wave
    l2 = mfcc2->dimensions[0]; // number of MFCCs in the second wave
    n  = mfcc1->dimensions[1]; // number of frames in the first wave
    m  = mfcc2->dimensions[1]; // number of frames in the second wave

    // check that the number of MFCCs is the same for both waves
    if (l1 != l2) {
        PyErr_SetString(PyExc_ValueError, "The number of MFCCs must be the same for both waves");
        return NULL;
    }

    // check that the norm (1D) arrays have the correct number of elements
    if (norm2_1->dimensions[0] != n) {
        PyErr_SetString(PyExc_ValueError, "The number of columns of mfcc1 must be equal to the number of elements of norm2_1");
        return NULL;
    }
    if (norm2_2->dimensions[0] != m) {
        PyErr_SetString(PyExc_ValueError, "The number of columns of mfcc2 must be equal to the number of elements of norm2_2");
        return NULL;
    }

    // delta cannot be greater than m 
    if (delta > m) {
        delta = m;
    }

    // pointer to cost matrix data
    mfcc1_ptr   = (double *)mfcc1->data;
    mfcc2_ptr   = (double *)mfcc2->data;
    norm2_1_ptr = (double *)norm2_1->data;
    norm2_2_ptr = (double *)norm2_2->data;
    
    // create cost matrix object
    cost_matrix_dimensions[0] = n;
    cost_matrix_dimensions[1] = delta;
    cost_matrix = (PyArrayObject *)PyArray_SimpleNew(2, cost_matrix_dimensions, PyArray_DOUBLE);
    cost_matrix_ptr = (double *)cost_matrix->data;

    // create centers object
    centers_dimensions[0] = n;
    centers = (PyArrayObject *)PyArray_SimpleNew(1, centers_dimensions, PyArray_INT32);
    centers_ptr = (int *)centers->data;
    
    // compute cost matrix
    _compute_cost_matrix(mfcc1_ptr, mfcc2_ptr, norm2_1_ptr, norm2_2_ptr, delta, cost_matrix_ptr, centers_ptr, n, m, l1);

    // decrement reference to local object no longer needed
    Py_DECREF(mfcc1);
    Py_DECREF(mfcc2);
    Py_DECREF(norm2_1);
    Py_DECREF(norm2_2);

    // return tuple with computed cost matrix and centers
    // PyTuple_SetItem steals a reference, so no PyDECREF is needed 
    tuple = PyTuple_New(2);
    PyTuple_SetItem(tuple, 0, PyArray_Return(cost_matrix));
    PyTuple_SetItem(tuple, 1, PyArray_Return(centers));
    return tuple;
}



// compute the accumulated cost matrix
// take the PyObject containing the following arguments:
//   - cost_matrix: 2D array (n x delta) of double
//   - centers:     1D array (n x 1) of int, centers[i] is the 0 <= center < m of the stripe at row i
// and return the accumulated_cost_matrix (2D array (n x delta) of double)
static PyObject *cdtw_compute_accumulated_cost_matrix_step(PyObject *self, PyObject *args) {
    PyObject *cost_matrix_raw;
    PyObject *centers_raw;

    PyArrayObject *cost_matrix, *centers, *accumulated_cost_matrix;
    npy_intp accumulated_cost_matrix_dimensions[2];
    double *cost_matrix_ptr, *accumulated_cost_matrix_ptr;
    int *centers_ptr;
    int n, delta;

    // O = object (do not convert or check for errors)
    if (!PyArg_ParseTuple(args, "OO", &cost_matrix_raw, &centers_raw)) {
        PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments");
        return NULL;
    }

    // convert to C contiguous array
    cost_matrix = (PyArrayObject *) PyArray_ContiguousFromObject(cost_matrix_raw, PyArray_DOUBLE, 2, 2);
    centers     = (PyArrayObject *) PyArray_ContiguousFromObject(centers_raw,     PyArray_INT32,  1, 1);

    // pointer to cost matrix data
    cost_matrix_ptr = (double *)cost_matrix->data;
    
    // get the dimensions of the input arguments
    n = cost_matrix->dimensions[0];
    delta = cost_matrix->dimensions[1];

    // check that the number of centers is the same as the number of rows of the cost_matrix
    if (centers->dimensions[0] != n) {
        PyErr_SetString(PyExc_ValueError, "The number of rows of cost_matrix must be equal to the number of elements of centers");
        return NULL;
    }
   
    // pointer to centers data
    centers_ptr = (int *)centers->data;
    
    // create accumulated cost matrix object 
    accumulated_cost_matrix_dimensions[0] = n;
    accumulated_cost_matrix_dimensions[1] = delta;
    accumulated_cost_matrix = (PyArrayObject *)PyArray_SimpleNew(2, accumulated_cost_matrix_dimensions, NPY_DOUBLE);

    // pointer to accumulated cost matrix data
    accumulated_cost_matrix_ptr = (double *)accumulated_cost_matrix->data;

    // compute accumulated cost matrix
    _compute_accumulated_cost_matrix(cost_matrix_ptr, centers_ptr, n, delta, accumulated_cost_matrix_ptr);

    // decrement reference to local object no longer needed
    Py_DECREF(cost_matrix);
    Py_DECREF(centers);

    // return computed best path
    return PyArray_Return(accumulated_cost_matrix);
}



// compute the best path 
// take the PyObject containing the following arguments:
//   - accumulated_cost_matrix: 2D array (n x delta) of double
//   - centers:                 1D array (n x 1) of int, centers[i] is the 0 <= center < m of the stripe at row i
// and return the best path as a list of (i, j) tuples, from (0,0) to (n-1, delta-1)
static PyObject *cdtw_compute_best_path_step(PyObject *self, PyObject *args) {
    PyObject *accumulated_cost_matrix_raw;
    PyObject *centers_raw;

    PyArrayObject *accumulated_cost_matrix, *centers;
    PyObject *best_path_ptr;
    double *accumulated_cost_matrix_ptr;
    int *centers_ptr;
    int n, delta;

    // O = object (do not convert or check for errors)
    if (!PyArg_ParseTuple(args, "OO", &accumulated_cost_matrix_raw, &centers_raw)) {
        PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments");
        return NULL;
    }

    // convert to C contiguous array
    accumulated_cost_matrix = (PyArrayObject *) PyArray_ContiguousFromObject(accumulated_cost_matrix_raw, PyArray_DOUBLE, 2, 2);
    centers                 = (PyArrayObject *) PyArray_ContiguousFromObject(centers_raw,                 PyArray_INT32,  1, 1);

    // pointer to cost matrix data
    accumulated_cost_matrix_ptr = (double *)accumulated_cost_matrix->data;
    
    // get the dimensions of the input arguments
    n = accumulated_cost_matrix->dimensions[0];
    delta = accumulated_cost_matrix->dimensions[1];

    // check that the number of centers is the same as the number of rows of the accumulated_cost_matrix
    if (centers->dimensions[0] != n) {
        PyErr_SetString(PyExc_ValueError, "The number of rows of accumulated_cost_matrix must be equal to the number of elements of centers");
        return NULL;
    }
   
    // pointer to centers data
    centers_ptr = (int *)centers->data;
    
    // create best path array of integers
    best_path_ptr = PyList_New(0);
    
    // compute best path
    _compute_best_path(accumulated_cost_matrix_ptr, centers_ptr, n, delta, best_path_ptr);

    // decrement reference to local object no longer needed
    Py_DECREF(accumulated_cost_matrix);
    Py_DECREF(centers);

    // return computed best path
    return best_path_ptr;
}



static PyMethodDef cdtw_methods[] = {
    // compute best path at once 
    {
        "cdtw_compute_best_path",
        cdtw_compute_best_path,
        METH_VARARGS,
        "Given the MFCCs of the two waves, compute and return the DTW best path at once"
    },
    // compute in separate steps
    {
        "cdtw_compute_cost_matrix_step",
        cdtw_compute_cost_matrix_step,
        METH_VARARGS,
        "Given the MFCCs of the two waves, compute and return the DTW cost matrix"
    },
    {
        "cdtw_compute_accumulated_cost_matrix_step",
        cdtw_compute_accumulated_cost_matrix_step,
        METH_VARARGS,
        "Given the DTW cost matrix, compute and return the DTW accumulated cost matrix"
    },
    {
        "cdtw_compute_best_path_step",
        cdtw_compute_best_path_step,
        METH_VARARGS,
        "Given the DTW accumulated cost matrix, compute and return the DTW best path"
    },
    {
        NULL,
        NULL,
        0,
        NULL
    }
};

PyMODINIT_FUNC initcdtw(void)  {
    (void) Py_InitModule("cdtw", cdtw_methods);
    import_array();
}



