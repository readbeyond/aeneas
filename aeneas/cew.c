/*

Python C Extension for synthesizing text with espeak

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
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "speak_lib.h"

static float current_time;
static float last_end_time;
static int synth_flags;
static int sample_rate;

static FILE *wave_file = NULL;

// write 4 bytes to a file, least significant first
static void _write_4_bytes(FILE *f, int value) {
	int ix;
	for (ix = 0; ix < 4; ix++) {
		fputc(value & 0xff, f);
		value = value >> 8;
	}
}

// open wave file and write its header
static int _open_wave_file(char const *path, int rate) {
	/*
    00000000  52 49 46 46 a8 af 00 00  57 41 56 45 66 6d 74 20  |RIFF....WAVEfmt |
    00000010  10 00 00 00 01 00 01 00  22 56 00 00 44 ac 00 00  |........"V..D...|
    00000020  02 00 10 00 64 61 74 61  84 af 00 00 00 00 00 00  |....data........|
    */
    static unsigned char wave_hdr[44] = {
		'R' , 'I', 'F'  , 'F', 0x24 , 0xf0, 0xff , 0x7f, 'W' , 'A' , 'V' , 'E' , 'f' , 'm' , 't', ' ',
		0x10, 0  , 0    , 0  , 1    , 0   , 1    , 0   , 9   , 0x3d, 0   , 0   , 0x12, 0x7a, 0  , 0  ,
		2   , 0  , 0x10 , 0  , 'd'  , 'a' , 't'  , 'a' , 0x00, 0xf0, 0xff, 0x7f
    };

	if (path == NULL) {
        return 2;
    }

	while (isspace(*path)) {
        path++;
    }

	wave_file = NULL;
	if (path[0] != 0) {
        wave_file = fopen(path, "wb");
	}
	
	if (wave_file == NULL) {
        return 1;
	}

	fwrite(wave_hdr, 1, 24, wave_file);
	_write_4_bytes(wave_file, rate);
	_write_4_bytes(wave_file, rate * 2);
	fwrite(&wave_hdr[32], 1, 12, wave_file);
	return 0;
}

// close wave file
static int _close_wave_file(void) {
	unsigned int pos;

	if (wave_file == NULL) {
        return 1;
    }

	fflush(wave_file);
	pos = ftell(wave_file);

	fseek(wave_file, 4, SEEK_SET);
	_write_4_bytes(wave_file, pos - 8);

	fseek(wave_file, 40, SEEK_SET);
	_write_4_bytes(wave_file,pos - 44);

	fclose(wave_file);
	wave_file = NULL;

    return 0;
}

// callback for synth events
static int _synth_callback(short *wav, int numsamples, espeak_EVENT *events) {
	if (wav == NULL) {
        return 1;
	}
	while (events->type != 0) {
        if (events->type == espeakEVENT_SAMPLERATE) {
			sample_rate = events->id.number;
		} else if (events->type == espeakEVENT_END) {
            //printf("end event at time: %.3f\n", 1.0 * events->audio_position / 1000);
            last_end_time = (1.0 * events->audio_position / 1000);
        } else if (events->type == espeakEVENT_WORD) {
            //printf("  word event at time: %.3f\n", 1.0 * events->audio_position / 1000);
        }
		events++;
	}
	if (numsamples > 0) {
		fwrite(wav, numsamples * 2, 1, wave_file);
	}
    return 0;
}

static int _cew_terminate_synthesis(void) {
    espeak_Terminate();
    return _close_wave_file();
}

static int _cew_synthesize(char const *text) {
	int size;
    if (text != NULL) {
        size = strlen(text);
        espeak_Synth(text, size + 1, 0, POS_CHARACTER, 0, synth_flags, NULL, NULL);
	}
	if (espeak_Synchronize() != EE_OK) {
        return 1;
	}
    current_time += last_end_time;
    return 0;
}

static int _cew_get_sample_rate(void) {
    return sample_rate;
}

static float _cew_get_current_time(void) {
    return current_time;
}

static int _cew_set_language(char const *voice_code) {
    if (espeak_SetVoiceByName(voice_code) != EE_OK) {
        return 1;
    }
    return 0;
}

static int _cew_initialize(char const *output_file_path) {
    char *data_path;
    
    // data_path is the path to espeak data for additional voices
    // NULL = use default path for espeak-data
	data_path = NULL;

    // set sentinel sample_rate
    sample_rate = 0;
    
    // synthesizer flags
	synth_flags = espeakCHARS_UTF8 | espeakPHONEMES | espeakENDPAUSE;

    // writing to a file (or no output), we can use synchronous mode
    sample_rate = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_path, 0);
   
	// set any non-default values of parameters. This must be done after espeak_Initialize()
    /*
	int volume = -1;
	int speed = -1;
	int pitch = -1;
	int wordgap = -1;
	int option_capitals = -1;
	int option_punctuation = -1;
	int option_phonemes = 0;
	int option_mbrola_phonemes = 0;
	int option_linelength = 0;
    
    if(speed > 0)
		espeak_SetParameter(espeakRATE,speed,0);
	if(volume >= 0)
		espeak_SetParameter(espeakVOLUME,volume,0);
	if(pitch >= 0)
		espeak_SetParameter(espeakPITCH,pitch,0);
	if(option_capitals >= 0)
		espeak_SetParameter(espeakCAPITALS,option_capitals,0);
	if(option_punctuation >= 0)
		espeak_SetParameter(espeakPUNCTUATION,option_punctuation,0);
	if(wordgap >= 0)
		espeak_SetParameter(espeakWORDGAP,wordgap,0);
	if(option_linelength > 0)
		espeak_SetParameter(espeakLINELENGTH,option_linelength,0);
	if(option_punctuation == 2)
		espeak_SetPunctuationList(option_punctlist);
    espeak_SetPhonemeTrace(option_phonemes | option_mbrola_phonemes, f_phonemes_out);
	*/

    // set synth callback 
    espeak_SetSynthCallback(_synth_callback);
    
	// open wave file
    if (wave_file == NULL) {
        if(_open_wave_file(output_file_path, sample_rate) != 0) {
            return 1;
        }
	}

    // reset time
    current_time = 0.0;
    last_end_time = 0.0;

    return 0; 
}

static PyObject *cew_synthesize_single(PyObject *self, PyObject *args) {
    PyObject *tuple;
    char const *output_file_path;
    char const *voice_code;
    char const *text;
    float begin, end;
    int sr;

    // s = string 
    if (!PyArg_ParseTuple(args, "sss", &output_file_path, &voice_code, &text)) {
        PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments");
        return NULL;
    }

    // set output path
    _cew_initialize(output_file_path);

    // get sample rate
    sr = _cew_get_sample_rate();

    if (_cew_set_language(voice_code) != 0) {
        return NULL;    
    }
    begin = _cew_get_current_time();
    _cew_synthesize(text);
    end = _cew_get_current_time();
    //printf("CT: %.3f %.3f\n", begin, end);
   
    // close output wave file
    _cew_terminate_synthesis();
    
    // build the tuple to be returned
    tuple = PyTuple_New(3);
    PyTuple_SetItem(tuple, 0, Py_BuildValue("i", sr));
    PyTuple_SetItem(tuple, 1, Py_BuildValue("f", begin));
    PyTuple_SetItem(tuple, 2, Py_BuildValue("f", end));
    return tuple;
}

static PyObject *cew_synthesize_multiple(PyObject *self, PyObject *args) {
    PyObject *tuple;
    PyObject *anchors;
    PyObject *fragments;
    PyObject *fragment;
    
    char const *output_file_path;
    char const *fragment_language;
    char const *fragment_text;
    float const quit_after;
    int const backwards;
    float begin, end;
    int i, n, sr, sf;

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

    // set output path
    _cew_initialize(output_file_path);

    // get sample rate
    sr = _cew_get_sample_rate();

    // number of synthesized fragments
    sf = 0;

    // get number of fragments
    n = PyList_Size(fragments);

    // allocates the list of anchors
    //anchors = PyList_New(n);
    anchors = PyList_New(0);

    // loop over all input fragments
    for (i = 0; i < n; i++) {
        int j;
        PyObject *anchor;

        // NOTE even if backwards is set, the WAV data will NOT be prepended!
        // get the actual index
        j = i;
        if (backwards != 0) {
            j = n - 1 - i;
        }

        // get fragment, which is a tuple (fragment_language, fragment_text)
        fragment = PyList_GetItem(fragments, j);
        Py_INCREF(fragment);
        if (!PyArg_ParseTuple(fragment, "ss", &fragment_language, &fragment_text)) {
            PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments");
            Py_XDECREF(fragments);
            Py_XDECREF(fragment);
            return NULL;
        }
        Py_DECREF(fragment);
        
        //printf("%s %s\n", fragment_language, fragment_text);
        if (_cew_set_language(fragment_language) != 0) {
            Py_XDECREF(fragments);
            Py_XDECREF(fragment);
            return NULL;
        }
        begin = _cew_get_current_time();
        _cew_synthesize(fragment_text);
        end = _cew_get_current_time();
        //printf("CT: %.3f %.3f\n", begin, end);
        
        anchor = PyTuple_New(2);
        // PyTuple_SetItem steals a reference, so no PyDECREF is needed
        PyTuple_SetItem(anchor, 0, Py_BuildValue("f", begin));
        PyTuple_SetItem(anchor, 1, Py_BuildValue("f", end));
        
        //PyList_SetItem(anchors, i, anchor);
        PyList_Append(anchors, anchor);
        Py_DECREF(anchor);

        // increment number of fragments synthesized
        sf += 1;

        // check if we generated >= quit_after seconds of audio
        // NOTE quit_after=0 disables this check
        if ((quit_after > 0) && (end >= quit_after)) {
            break;
        }
    }

    // close output wave file
    _cew_terminate_synthesis();

    // decrement reference to local object no longer needed
    Py_DECREF(fragments);

    // build the tuple to be returned
    tuple = PyTuple_New(3);
    PyTuple_SetItem(tuple, 0, Py_BuildValue("i", sr));
    PyTuple_SetItem(tuple, 1, Py_BuildValue("i", sf));
    PyTuple_SetItem(tuple, 2, anchors);
    return tuple;
}

static PyMethodDef cew_methods[] = {
    {
        "cew_synthesize_single",
        cew_synthesize_single,
        METH_VARARGS,
        "Synthesize a single text fragment with espeak"
    },
    {
        "cew_synthesize_multiple",
        cew_synthesize_multiple,
        METH_VARARGS,
        "Synthesize multiple text fragments with espeak"
    },
    {
        NULL,
        NULL,
        0,
        NULL
    }
};

PyMODINIT_FUNC initcew(void)  {
    (void) Py_InitModule("cew", cew_methods);
}



