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

#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <numpy/npy_math.h>

#include "cdtw_func.h"

// append a new tuple (i, j) to the given list
static void _append(PyObject *list, uint32_t i, uint32_t j) {
    PyObject *tuple;
    
    tuple = PyTuple_New(2);
    // PyTuple_SetItem steals a reference, so no PyDECREF is needed
    PyTuple_SetItem(tuple, 0, Py_BuildValue("I", i));
    PyTuple_SetItem(tuple, 1, Py_BuildValue("I", j));
    PyList_Append(list, tuple);
}

// convert array of struct to list of tuples 
static void _array_to_list(struct PATH_CELL *best_path, uint32_t best_path_length, PyObject *list) {
    uint32_t k;

    for (k = 0; k < best_path_length; ++k) {
        _append(list, best_path[k].i, best_path[k].j);
    }
}

// compute the best path "all in one"
// take the PyObject containing the following arguments:
//   - mfcc1:       2D array (l x n) of double, MFCCs of the first wave
//   - mfcc2:       2D array (l x m) of double, MFCCs of the second wave
//   - delta:       uint, the number of frames of margin
// and return the best path as a list of (i, j) tuples, from (0,0) to (n-1, m-1)
static PyObject *compute_best_path(PyObject *self, PyObject *args) {
    PyObject *mfcc1_raw;
    PyObject *mfcc2_raw;
    uint32_t delta;
 
    PyArrayObject *mfcc1, *mfcc2, *cost_matrix, *centers;
    PyObject *best_path_ptr;
    npy_intp cost_matrix_dimensions[2];
    npy_intp centers_dimensions[1];
    double *mfcc1_ptr, *mfcc2_ptr, *cost_matrix_ptr;
    uint32_t *centers_ptr;
    uint32_t l1, l2, n, m;
    struct PATH_CELL *best_path;
    uint32_t best_path_length;

    // O = object (do not convert or check for errors)
    // I = unsigned int
    if (!PyArg_ParseTuple(args, "OOI", &mfcc1_raw, &mfcc2_raw, &delta)) {
        PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments with the OOOOi mask");
        return NULL;
    }

    // convert to C contiguous array
    mfcc1 = (PyArrayObject *)PyArray_ContiguousFromAny(mfcc1_raw, NPY_DOUBLE, 2, 2);
    mfcc2 = (PyArrayObject *)PyArray_ContiguousFromAny(mfcc2_raw, NPY_DOUBLE, 2, 2);

    // check for conversion errors
    if ((mfcc1 == NULL) || (mfcc2 == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Error while converting arguments using PyArray_ContiguousFromAny");
        return NULL;
    }

    // get the dimensions of the input arguments
    l1 = PyArray_DIMS(mfcc1)[0]; // number of MFCCs in the first wave
    l2 = PyArray_DIMS(mfcc2)[0]; // number of MFCCs in the second wave
    n = PyArray_DIMS(mfcc1)[1]; // number of frames in the first wave
    m = PyArray_DIMS(mfcc2)[1]; // number of frames in the second wave

    // check that the number of MFCCs is the same for both waves
    if (l1 != l2) {
        PyErr_SetString(PyExc_ValueError, "The number of MFCCs must be the same for both waves");
        return NULL;
    }

    // delta cannot be greater than m 
    if (delta > m) {
        delta = m;
    }
    
    // pointer to cost matrix data
    mfcc1_ptr = (double *)PyArray_DATA(mfcc1);
    mfcc2_ptr = (double *)PyArray_DATA(mfcc2);
    
    // create cost matrix object
    cost_matrix_dimensions[0] = n;
    cost_matrix_dimensions[1] = delta;
    cost_matrix = (PyArrayObject *)PyArray_SimpleNew(2, cost_matrix_dimensions, NPY_DOUBLE);
    cost_matrix_ptr = (double *)PyArray_DATA(cost_matrix);

    // create centers object
    centers_dimensions[0] = n;
    centers = (PyArrayObject *)PyArray_SimpleNew(1, centers_dimensions, NPY_UINT32);
    centers_ptr = (uint32_t *)PyArray_DATA(centers);

    // actual computation
    if (_compute_cost_matrix(mfcc1_ptr, mfcc2_ptr, delta, cost_matrix_ptr, centers_ptr, n, m, l1) != CDTW_SUCCESS) {
       Py_XDECREF(mfcc1);
       Py_XDECREF(mfcc2);
       Py_XDECREF(cost_matrix);
       Py_XDECREF(centers);
       PyErr_SetString(PyExc_ValueError, "Error while computing cost matrix");
       return NULL;
    }
    
    if (_compute_accumulated_cost_matrix_in_place(cost_matrix_ptr, centers_ptr, n, delta) != CDTW_SUCCESS) {
       Py_XDECREF(mfcc1);
       Py_XDECREF(mfcc2);
       Py_XDECREF(cost_matrix);
       Py_XDECREF(centers);
       PyErr_SetString(PyExc_ValueError, "Error while computing accumulated cost matrix");
       return NULL;
    }
    
    if (_compute_best_path(cost_matrix_ptr, centers_ptr, n, delta, &best_path, &best_path_length) != CDTW_SUCCESS) {
       Py_XDECREF(mfcc1);
       Py_XDECREF(mfcc2);
       Py_XDECREF(cost_matrix);
       Py_XDECREF(centers);
       PyErr_SetString(PyExc_ValueError, "Error while computing best path");
       return NULL;
    }

    // convert array of struct to list of tuples 
    best_path_ptr = PyList_New(0);
    _array_to_list(best_path, best_path_length, best_path_ptr);
    free((void *)best_path);
    best_path = NULL;

    // decrement reference to local object no longer needed
    Py_DECREF(mfcc1);
    Py_DECREF(mfcc2);
    Py_DECREF(cost_matrix);
    Py_DECREF(centers);

    return best_path_ptr;
}

// compute the cost matrix and the corresponding stripe centers 
// take the PyObject containing the following arguments:
//   - mfcc1:       2D array (l x n) of double, MFCCs of the first wave
//   - mfcc2:       2D array (l x m) of double, MFCCs of the second wave
//   - delta:       uint, the number of frames of margin
// and return a tuple (cost_matrix, centers), where
//   - cost_matrix: 2D array (n x delta) of double
//   - centers:     1D array (n x 1) of uint, centers[i] is the 0 <= center < m of the stripe at row i
static PyObject *compute_cost_matrix_step(PyObject *self, PyObject *args) {
    PyObject *mfcc1_raw;
    PyObject *mfcc2_raw;
    uint32_t delta;

    PyArrayObject *mfcc1, *mfcc2, *cost_matrix, *centers;
    PyObject *tuple;
    npy_intp cost_matrix_dimensions[2];
    npy_intp centers_dimensions[1];
    double *mfcc1_ptr, *mfcc2_ptr, *cost_matrix_ptr;
    uint32_t *centers_ptr;
    uint32_t l1, l2, n, m;
   
    // O = object (do not convert or check for errors)
    // I = unsigned int
    if (!PyArg_ParseTuple(args, "OOI", &mfcc1_raw, &mfcc2_raw, &delta)) {
        PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments with the OOOOi mask");
        return NULL;
    }

    // convert to C contiguous array
    mfcc1 = (PyArrayObject *)PyArray_ContiguousFromAny(mfcc1_raw, NPY_DOUBLE, 2, 2);
    mfcc2 = (PyArrayObject *)PyArray_ContiguousFromAny(mfcc2_raw, NPY_DOUBLE, 2, 2);

    // check for conversion errors
    if ((mfcc1 == NULL) || (mfcc2 == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Error while converting arguments using PyArray_ContiguousFromAny");
        return NULL;
    }

    // get the dimensions of the input arguments
    l1 = PyArray_DIMS(mfcc1)[0]; // number of MFCCs in the first wave
    l2 = PyArray_DIMS(mfcc2)[0]; // number of MFCCs in the second wave
    n = PyArray_DIMS(mfcc1)[1]; // number of frames in the first wave
    m = PyArray_DIMS(mfcc2)[1]; // number of frames in the second wave

    // check that the number of MFCCs is the same for both waves
    if (l1 != l2) {
        PyErr_SetString(PyExc_ValueError, "The number of MFCCs must be the same for both waves");
        return NULL;
    }

    // delta cannot be greater than m 
    if (delta > m) {
        delta = m;
    }

    // pointer to cost matrix data
    mfcc1_ptr = (double *)PyArray_DATA(mfcc1);
    mfcc2_ptr = (double *)PyArray_DATA(mfcc2);

    // create cost matrix object
    cost_matrix_dimensions[0] = n;
    cost_matrix_dimensions[1] = delta;
    cost_matrix = (PyArrayObject *)PyArray_SimpleNew(2, cost_matrix_dimensions, NPY_DOUBLE);
    cost_matrix_ptr = (double *)PyArray_DATA(cost_matrix);

    // create centers object
    centers_dimensions[0] = n;
    centers = (PyArrayObject *)PyArray_SimpleNew(1, centers_dimensions, NPY_UINT32);
    centers_ptr = (uint32_t *)PyArray_DATA(centers);
    
    // compute cost matrix
    if (_compute_cost_matrix(mfcc1_ptr, mfcc2_ptr, delta, cost_matrix_ptr, centers_ptr, n, m, l1) != CDTW_SUCCESS) {
        Py_XDECREF(mfcc1);
        Py_XDECREF(mfcc2);
        Py_XDECREF(cost_matrix);
        Py_XDECREF(centers);
        PyErr_SetString(PyExc_ValueError, "Error while computing cost matrix");
        return NULL;
    }

    // decrement reference to local object no longer needed
    Py_DECREF(mfcc1);
    Py_DECREF(mfcc2);

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
static PyObject *compute_accumulated_cost_matrix_step(PyObject *self, PyObject *args) {
    PyObject *cost_matrix_raw;
    PyObject *centers_raw;

    PyArrayObject *cost_matrix, *centers, *accumulated_cost_matrix;
    npy_intp accumulated_cost_matrix_dimensions[2];
    double *cost_matrix_ptr, *accumulated_cost_matrix_ptr;
    uint32_t *centers_ptr;
    uint32_t n, delta;

    // O = object (do not convert or check for errors)
    if (!PyArg_ParseTuple(args, "OO", &cost_matrix_raw, &centers_raw)) {
        PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments");
        return NULL;
    }

    // convert to C contiguous array
    cost_matrix = (PyArrayObject *) PyArray_ContiguousFromAny(cost_matrix_raw, NPY_DOUBLE, 2, 2);
    centers = (PyArrayObject *) PyArray_ContiguousFromAny(centers_raw, NPY_UINT32, 1, 1);

    // pointer to cost matrix data
    cost_matrix_ptr = (double *)PyArray_DATA(cost_matrix);
    
    // get the dimensions of the input arguments
    n = PyArray_DIMS(cost_matrix)[0];
    delta = PyArray_DIMS(cost_matrix)[1];

    // check that the number of centers is the same as the number of rows of the cost_matrix
    if (PyArray_DIMS(centers)[0] != n) {
        PyErr_SetString(PyExc_ValueError, "The number of rows of cost_matrix must be equal to the number of elements of centers");
        return NULL;
    }
   
    // pointer to centers data
    centers_ptr = (uint32_t *)PyArray_DATA(centers);
    
    // create accumulated cost matrix object 
    accumulated_cost_matrix_dimensions[0] = n;
    accumulated_cost_matrix_dimensions[1] = delta;
    accumulated_cost_matrix = (PyArrayObject *)PyArray_SimpleNew(2, accumulated_cost_matrix_dimensions, NPY_DOUBLE);

    // pointer to accumulated cost matrix data
    accumulated_cost_matrix_ptr = (double *)PyArray_DATA(accumulated_cost_matrix);

    // compute accumulated cost matrix
    if (_compute_accumulated_cost_matrix(cost_matrix_ptr, centers_ptr, n, delta, accumulated_cost_matrix_ptr) != CDTW_SUCCESS) {
        Py_XDECREF(cost_matrix);
        Py_XDECREF(centers);
        PyErr_SetString(PyExc_ValueError, "Error while computing accumulated cost matrix");
        return NULL;
    }

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
// and return the best path as a list of (i, j) tuples, from (0,0) to (n-1, m-1)
static PyObject *compute_best_path_step(PyObject *self, PyObject *args) {
    PyObject *accumulated_cost_matrix_raw;
    PyObject *centers_raw;

    PyArrayObject *accumulated_cost_matrix, *centers;
    PyObject *best_path_ptr;
    double *accumulated_cost_matrix_ptr;
    uint32_t *centers_ptr;
    uint32_t n, delta;
    struct PATH_CELL *best_path;
    uint32_t best_path_length;

    // O = object (do not convert or check for errors)
    if (!PyArg_ParseTuple(args, "OO", &accumulated_cost_matrix_raw, &centers_raw)) {
        PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments");
        return NULL;
    }

    // convert to C contiguous array
    accumulated_cost_matrix = (PyArrayObject *) PyArray_ContiguousFromAny(accumulated_cost_matrix_raw, NPY_DOUBLE, 2, 2);
    centers = (PyArrayObject *) PyArray_ContiguousFromAny(centers_raw, NPY_UINT32, 1, 1);

    // pointer to cost matrix data
    accumulated_cost_matrix_ptr = (double *)PyArray_DATA(accumulated_cost_matrix);
    
    // get the dimensions of the input arguments
    n = PyArray_DIMS(accumulated_cost_matrix)[0];
    delta = PyArray_DIMS(accumulated_cost_matrix)[1];

    // check that the number of centers is the same as the number of rows of the accumulated_cost_matrix
    if (PyArray_DIMS(centers)[0] != n) {
        PyErr_SetString(PyExc_ValueError, "The number of rows of accumulated_cost_matrix must be equal to the number of elements of centers");
        return NULL;
    }
   
    // pointer to centers data
    centers_ptr = (uint32_t *)PyArray_DATA(centers);
    
    // create best path array of integers
    best_path_ptr = PyList_New(0);
    
    // compute best path
    if (_compute_best_path(accumulated_cost_matrix_ptr, centers_ptr, n, delta, &best_path, &best_path_length) != CDTW_SUCCESS) {
        Py_XDECREF(accumulated_cost_matrix);
        Py_XDECREF(centers);
        PyErr_SetString(PyExc_ValueError, "Error while computing accumulated cost matrix");
        return NULL;
    }

    // convert array of struct to list of tuples 
    _array_to_list(best_path, best_path_length, best_path_ptr);
    free((void *)best_path);
    best_path = NULL;

    // decrement reference to local object no longer needed
    Py_DECREF(accumulated_cost_matrix);
    Py_DECREF(centers);

    // return computed best path
    return best_path_ptr;
}



static PyMethodDef cdtw_methods[] = {
    {
        "compute_best_path",
        compute_best_path,
        METH_VARARGS,
        "Given the MFCCs of the two waves, compute and return the DTW best path at once\n"
        ":param object mfcc1: numpy 2D matrix (mfcc_size, n) of MFCCs of the first wave\n"
        ":param object mfcc2: numpy 2D matrix (mfcc_size, m) of MFCCs of the second wave\n"
        ":param uint delta: the margin, in number of frames\n"
        ":rtype: a list of tuples (i, j), from (0, 0) to (n-1, m-1) representing the best path"
    },
    {
        "compute_cost_matrix_step",
        compute_cost_matrix_step,
        METH_VARARGS,
        "Given the MFCCs of the two waves, compute and return the DTW cost matrix\n"
        ":param object mfcc1: numpy 2D matrix (mfcc_size, n) of MFCCs of the first wave\n"
        ":param object mfcc2: numpy 2D matrix (mfcc_size, m) of MFCCs of the second wave\n"
        ":param uint delta: the margin, in number of frames\n"
        ":rtype: tuple (cost_matrix, centers)"
    },
    {
        "compute_accumulated_cost_matrix_step",
        compute_accumulated_cost_matrix_step,
        METH_VARARGS,
        "Given the DTW cost matrix, compute and return the DTW accumulated cost matrix\n"
        ":param object cost_matrix: the cost matrix (n, delta)\n"
        ":param object centers: the centers (n)\n"
        ":rtype: the accumulated cost matrix"
    },
    {
        "compute_best_path_step",
        compute_best_path_step,
        METH_VARARGS,
        "Given the DTW accumulated cost matrix, compute and return the DTW best path\n"
        ":param object accumulated_cost_matrix: the accumulated cost matrix (n, delta)\n"
        ":param object centers: the centers (n)\n"
        ":rtype: a list of tuples (i, j), from (0, 0) to (n-1, m-1) representing the best path"
    },
    {
        NULL,
        NULL,
        0,
        NULL
    }
};

// Python 2 and 3
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cdtw",         /* m_name */
    "cdtw",         /* m_doc */
    -1,             /* m_size */
    cdtw_methods,   /* m_methods */
    NULL,           /* m_reload */
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};
#endif

static PyObject *moduleinit(void) {
    PyObject *m;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("cdtw", cdtw_methods);
#endif

    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_cdtw(void) {
    PyObject *ret = moduleinit();
    import_array();
    return ret; 
}
#else
PyMODINIT_FUNC initcdtw(void) {
    moduleinit();
    import_array();
}
#endif



