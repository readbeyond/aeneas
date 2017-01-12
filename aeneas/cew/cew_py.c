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

Python C Extension for synthesizing text with eSpeak

*/

#include <Python.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "speak_lib.h"
#include "cew_func.h"

static PyObject *synthesize_multiple(PyObject *self, PyObject *args) {
    PyObject *tuple;
    PyObject *anchors;
    PyObject *fragments;

    char const *output_file_path;
    float quit_after = 0.0;
    int backwards = 0;
    struct FRAGMENT_INFO *fragments_synt;

    int sample_rate; // int because espeak lib returns it as such
    size_t number_of_fragments, i, synthesized;

    // s = string
    // f = float
    // i = integer (used as a boolean, 0=false, 1=true)
    // O = object
    if (!PyArg_ParseTuple(args, "sfiO", &output_file_path, &quit_after, &backwards, &fragments)) {
        PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments");
        Py_XDECREF(fragments);
        return NULL;
    }
    Py_INCREF(fragments);

    // get number of fragments
    number_of_fragments = PyList_Size(fragments);

    // allocate an array of FRAGMENT_INFO to be passed to _synthesize_multiple()
    fragments_synt = (struct FRAGMENT_INFO *)calloc(number_of_fragments, sizeof(struct FRAGMENT_INFO));

    // loop over all input fragments
    for (i = 0; i < number_of_fragments; ++i) {
        PyObject *fragment;

        // get fragment, which is a tuple (fragment_language, fragment_text)
        fragment = PyList_GetItem(fragments, i);
        Py_INCREF(fragment);
        if (!PyArg_ParseTuple(fragment, "ss", &(fragments_synt[i].voice_code), &(fragments_synt[i].text))) {
            PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments");
            free((void*)fragments_synt);
            fragments_synt = NULL;
            Py_XDECREF(fragments);
            Py_XDECREF(fragment);
            return NULL;
        }
        Py_DECREF(fragment);
        //printf("Added: %s %s\n", fragments_synt[i].voice_code, fragments_synt[i].text);
    }
    Py_DECREF(fragments);

    // synthesize multiple
    if (_synthesize_multiple(
                output_file_path,
                &fragments_synt,
                number_of_fragments,
                quit_after,
                backwards,
                &sample_rate,
                &synthesized
        ) != 0) {
        PyErr_SetString(PyExc_ValueError, "Error while synthesizing multiple fragments");
        free((void*)fragments_synt);
        fragments_synt = NULL;
        return NULL;
    }

    // allocates the list of anchors
    //anchors = PyList_New(n);
    anchors = PyList_New(0);

    // loop over all input fragments
    for (i = 0; i < synthesized; i++) {
        PyObject *anchor;

        anchor = PyTuple_New(2);
        // PyTuple_SetItem steals a reference, so no PyDECREF is needed
        PyTuple_SetItem(anchor, 0, Py_BuildValue("f", fragments_synt[i].begin));
        PyTuple_SetItem(anchor, 1, Py_BuildValue("f", fragments_synt[i].end));
        PyList_Append(anchors, anchor);
        Py_DECREF(anchor);
    }

    // deallocate
    free((void*)fragments_synt);
    fragments_synt = NULL;

    // build the tuple to be returned
    // NOTE: returning sample_rate as an int, as the espeak lib does
    tuple = PyTuple_New(3);
    PyTuple_SetItem(tuple, 0, Py_BuildValue("i", sample_rate));
    PyTuple_SetItem(tuple, 1, Py_BuildValue("I", synthesized));
    PyTuple_SetItem(tuple, 2, anchors);

    return tuple;
}

static PyMethodDef cew_methods[] = {
    {
        "synthesize_multiple",
        synthesize_multiple,
        METH_VARARGS,
        "Synthesize multiple text fragments with eSpeak\n"
        ":param string output_file_path: the path of the WAVE file to be created\n"
        ":param float quit_after: if > 0, stop synthesizing when reaching quit_after seconds\n"
        ":param int backwards: if 1, synthesize backwards, from the last fragment to the first\n"
        ":param list fragments: list of (voice_code, text) tuples of text fragments to be synthesized\n"
        ":rtype: tuple (sample_rate, synthesized, list) where list is a list of (begin, end) time values"
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
    "cew",          /* m_name */
    "cew",          /* m_doc */
    -1,             /* m_size */
    cew_methods,    /* m_methods */
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
    m = Py_InitModule("cew", cew_methods);
#endif

    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_cew(void) {
    return moduleinit();
}
#else
PyMODINIT_FUNC initcew(void) {
    moduleinit();
}
#endif



