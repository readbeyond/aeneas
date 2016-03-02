/*

Python C Extension for reading WAVE mono files.

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
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cwave_func.h"

static PyObject *get_audio_info(PyObject *self, PyObject *args) {
    PyObject *tuple;
    char *audio_file_path;
    FILE *audio_file_ptr;
    struct WAVE_INFO audio_info;
    uint32_t sample_rate, total_samples; // a WAVE file cannot have more than 2^32 samples

    // s = string
    if (!PyArg_ParseTuple(args, "s", &audio_file_path)) {
        PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments");
        return NULL;
    }

    audio_file_ptr = wave_open(audio_file_path, &audio_info);
    if (audio_file_ptr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Error while opening the WAVE file");
        return NULL;
    }
    sample_rate = audio_info.leSampleRate;
    total_samples = audio_info.coNumSamples;
    wave_close(audio_file_ptr);

    // build the tuple to be returned
    tuple = PyTuple_New(2);
    PyTuple_SetItem(tuple, 0, Py_BuildValue("I", sample_rate));
    PyTuple_SetItem(tuple, 1, Py_BuildValue("I", total_samples));
    return tuple;
}

static PyObject *read_audio_data(PyObject *self, PyObject *args) {
    PyObject *tuple;
    PyArrayObject *audio_data;
    npy_intp audio_data_dimensions[1];
    char *audio_file_path;
    FILE *audio_file_ptr;
    struct WAVE_INFO audio_info;
    uint32_t from_sample, num_samples, total_samples; // a WAVE file cannot have more than 2^32 samples
    uint32_t sample_rate;                             // sample_rate is a uint32_t in the WAVE header
    double *buffer;                                   // this buffer will store the data read

    // s = string
    // I = unsigned int
    if (!PyArg_ParseTuple(args, "sII", &audio_file_path, &from_sample, &num_samples)) {
        PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments");
        return NULL;
    }

    audio_file_ptr = wave_open(audio_file_path, &audio_info);
    if (audio_file_ptr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Error while opening the WAVE file");
        return NULL;
    }
    sample_rate = audio_info.leSampleRate;
    total_samples = audio_info.coNumSamples;

    if (num_samples == 0) {
        num_samples = total_samples;
    }
    if (from_sample + num_samples > total_samples) {
        PyErr_SetString(PyExc_ValueError, "Error while reading WAVE data: wrong index or length");
        return NULL;
    }
    buffer = (double *)calloc(num_samples, sizeof(double));
    if (wave_read_double(audio_file_ptr, &audio_info, buffer, from_sample, num_samples) != CWAVE_SUCCESS) {
        wave_close(audio_file_ptr);
        PyErr_SetString(PyExc_ValueError, "Error while reading WAVE data: unable to read data");
        return NULL;
    }
    wave_close(audio_file_ptr);

    // build the array to be returned
    // create data object
    audio_data_dimensions[0] = num_samples;
    audio_data = (PyArrayObject *)PyArray_SimpleNewFromData(1, audio_data_dimensions, NPY_DOUBLE, buffer);

    // build the tuple to be returned
    tuple = PyTuple_New(2);
    PyTuple_SetItem(tuple, 0, Py_BuildValue("I", sample_rate));
    PyTuple_SetItem(tuple, 1, PyArray_Return(audio_data));
    return tuple;
}


static PyMethodDef cwave_methods[] = {
    {
        "get_audio_info",
        get_audio_info,
        METH_VARARGS,
        "Get information about a WAVE file\n"
        ":param string audio_file_path: the file path of the audio file\n"
        ":rtype: tuple (sample_rate, num_samples)"
    },
    {
        "read_audio_data",
        read_audio_data,
        METH_VARARGS,
        "Get audio data from a WAVE file\n"
        ":param string audio_file_path: the file path of the audio file\n"
        ":param uint from_sample: read from this sample index\n"
        ":param uint num_samples: read this many samples\n"
        ":rtype: tuple (sample_rate, list) where list is a list of float values, one per sample"
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
    "cwave",        /* m_name */
    "cwave",        /* m_doc */
    -1,             /* m_size */
    cwave_methods,  /* m_methods */
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
    m = Py_InitModule("cwave", cwave_methods);
#endif

    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_cwave(void) {
    PyObject *ret = moduleinit();
    import_array();
    return ret;
}
#else
PyMODINIT_FUNC initcwave(void) {
    moduleinit();
    import_array();
}
#endif



