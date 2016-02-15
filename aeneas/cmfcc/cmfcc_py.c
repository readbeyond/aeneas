/*

Python C Extension for computing the MFCCs from a WAVE mono file.

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

#include "cmfcc_func.h"

// compute the MFCCs of the given audio data (mono)
static PyObject *compute_from_data(PyObject *self, PyObject *args) {
    PyObject *data_raw;         // 1D array of double, holding the data
    uint32_t sample_rate;       // sample rate (default: 16000)
    uint32_t filter_bank_size;  // number of filters in the filter bank (default: 40)
    uint32_t mfcc_size;         // number of ceptral coefficients (default: 13)
    uint32_t fft_order;         // FFT order; must be a power of 2 (default: 512)
    double lower_frequency;     // lower frequency (default: 133.3333)
    double upper_frequency;     // upper frequency; must be <= sample_rate/2 = Nyquist frequency (default: 6855.4976)
    double emphasis_factor;     // pre-emphasis factor (default: 0.97)
    double window_length;       // window length (default: 0.0250)
    double window_shift;        // window shift (default: 0.010)

    PyObject *tuple;
    PyArrayObject *data, *mfcc;
    npy_intp mfcc_dimensions[2];
    double *data_ptr, *mfcc_ptr;
    uint32_t data_length, mfcc_length;

    // O = object (do not convert or check for errors)
    // I = uint32_teger
    // d = double
    if (!PyArg_ParseTuple(
            args,
            "OIIIIddddd",
            &data_raw,
            &sample_rate,
            &filter_bank_size,
            &mfcc_size,
            &fft_order,
            &lower_frequency,
            &upper_frequency,
            &emphasis_factor,
            &window_length,
            &window_shift)
    ) {
        PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments");
        return NULL;
    }

    // convert to C contiguous array
    data = (PyArrayObject *)PyArray_ContiguousFromAny(data_raw, NPY_DOUBLE, 1, 1);

    // pointer to data data
    data_ptr = (double *)PyArray_DATA(data);

    // number of audio samples in data (= duration in seconds * sample_rate)
    data_length = (uint32_t)PyArray_DIMS(data)[0];

    // compute MFCC matrix
    if (compute_mfcc_from_data(
        data_ptr,
        data_length,
        sample_rate,
        filter_bank_size,
        mfcc_size,
        fft_order,
        lower_frequency,
        upper_frequency,
        emphasis_factor,
        window_length,
        window_shift,
        &mfcc_ptr,
        &mfcc_length) != CMFCC_SUCCESS
    ) {
        // failed
        PyErr_SetString(PyExc_ValueError, "Error while calling compute_mfcc_from_data()");
        Py_XDECREF(data);
        return NULL;
    }

    // decrement reference to local object no longer needed
    Py_DECREF(data);

    // create mfcc object
    mfcc_dimensions[0] = mfcc_length;
    mfcc_dimensions[1] = mfcc_size;
    mfcc = (PyArrayObject *)PyArray_SimpleNewFromData(2, mfcc_dimensions, NPY_DOUBLE, mfcc_ptr);

    // build the tuple to be returned
    tuple = PyTuple_New(3);
    PyTuple_SetItem(tuple, 0, PyArray_Return(mfcc));
    PyTuple_SetItem(tuple, 1, Py_BuildValue("I", data_length));
    PyTuple_SetItem(tuple, 2, Py_BuildValue("I", sample_rate));
    return tuple;
}

// compute the MFCCs of the given audio file 
static PyObject *compute_from_file(PyObject *self, PyObject *args) {
    char *audio_file_path;      // path of the WAVE file
    uint32_t filter_bank_size;  // number of filters in the filter bank (default: 40)
    uint32_t mfcc_size;         // number of ceptral coefficients (default: 13)
    uint32_t fft_order;         // FFT order; must be a power of 2 (default: 512)
    double lower_frequency;     // lower frequency (default: 133.3333)
    double upper_frequency;     // upper frequency; must be <= sample_rate/2 = Nyquist frequency (default: 6855.4976)
    double emphasis_factor;     // pre-emphasis factor (default: 0.97)
    double window_length;       // window length (default: 0.0250)
    double window_shift;        // window shift (default: 0.010)

    PyObject *tuple;
    PyArrayObject *mfcc;
    npy_intp mfcc_dimensions[2];
    double *mfcc_ptr;
    uint32_t sample_rate;
    uint32_t data_length, mfcc_length;

    // s = string
    // I = uint32_teger
    // d = double
    if (!PyArg_ParseTuple(
            args,
            "sIIIddddd",
            &audio_file_path,
            &filter_bank_size,
            &mfcc_size,
            &fft_order,
            &lower_frequency,
            &upper_frequency,
            &emphasis_factor,
            &window_length,
            &window_shift)
    ) {
        PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments");
        return NULL;
    }

    // compute MFCC matrix
    if (compute_mfcc_from_file(
        audio_file_path,
        filter_bank_size,
        mfcc_size,
        fft_order,
        lower_frequency,
        upper_frequency,
        emphasis_factor,
        window_length,
        window_shift,
        &data_length,
        &sample_rate,
        &mfcc_ptr,
        &mfcc_length) != CMFCC_SUCCESS
    ) {
        // failed
        PyErr_SetString(PyExc_ValueError, "Error while calling compute_mfcc_from_file()");
        return NULL;
    }

    // create mfcc object
    mfcc_dimensions[0] = mfcc_length;
    mfcc_dimensions[1] = mfcc_size;
    mfcc = (PyArrayObject *)PyArray_SimpleNewFromData(2, mfcc_dimensions, NPY_DOUBLE, mfcc_ptr);

    // build the tuple to be returned
    tuple = PyTuple_New(3);
    PyTuple_SetItem(tuple, 0, PyArray_Return(mfcc));
    PyTuple_SetItem(tuple, 1, Py_BuildValue("I", data_length));
    PyTuple_SetItem(tuple, 2, Py_BuildValue("I", sample_rate));
    return tuple;
}


static PyMethodDef cmfcc_methods[] = {
    {
        "compute_from_data",
        compute_from_data,
        METH_VARARGS,
        "Given the data from a mono PCM16 WAVE file, compute and return the MFCCs\n"
        ":param object data_raw: numpy 1D array of float values, one per sample\n"
        ":param uint sample_rate: the sample rate of the WAVE file\n"
        ":param uint filter_bank_size: the number of MFCC filters\n"
        ":param uint mfcc_size: the number of MFCCs\n"
        ":param uint fft_order: the order of the FFT\n"
        ":param float lower_frequency: cut below this frequency, in Hz\n"
        ":param float upper_frequency: cut above this frequency, in Hz\n"
        ":param float emphasis_factor: pre-amplify frames by this factor\n"
        ":param float window_length: MFCC window lenght, in s\n"
        ":param float window_shift: MFCC window shift, in s\n"
        ":rtype: tuple (mfccs, data_length, sample_rate)"
    },
    {
        "compute_from_file",
        compute_from_file,
        METH_VARARGS,
        "Given the path of the mono PCM16 WAVE file, compute and return the MFCCs\n"
        ":param string audio_file_path: the path of the audio file\n"
        ":param uint filter_bank_size: the number of MFCC filters\n"
        ":param uint mfcc_size: the number of MFCCs\n"
        ":param uint fft_order: the order of the FFT\n"
        ":param float lower_frequency: cut below this frequency, in Hz\n"
        ":param float upper_frequency: cut above this frequency, in Hz\n"
        ":param float emphasis_factor: pre-amplify frames by this factor\n"
        ":param float window_length: MFCC window lenght, in s\n"
        ":param float window_shift: MFCC window shift, in s\n"
        ":rtype: tuple (mfccs, data_length, sample_rate)"
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
    "cmfcc",        /* m_name */
    "cmfcc",        /* m_doc */
    -1,             /* m_size */
    cmfcc_methods,  /* m_methods */
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
    m = Py_InitModule("cmfcc", cmfcc_methods);
#endif

    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_cmfcc(void) {
    PyObject *ret = moduleinit();
    import_array();
    return ret;
}
#else
PyMODINIT_FUNC initcmfcc(void) {
    moduleinit();
    import_array();
}
#endif



