/*

Python C Extension for synthesizing text with eSpeak

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

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "speak_lib.h"
#include "cew_func.h"

static float current_time;
static float last_end_time;
static int synth_flags;
static int sample_rate;

static FILE *wave_file = NULL;

/*
00000000  52 49 46 46 XX XX XX XX  57 41 56 45 66 6d 74 20  |RIFF....WAVEfmt |
00000010  10 00 00 00 01 00 01 00  22 56 00 00 44 ac 00 00  |........"V..D...|
00000020  02 00 10 00 64 61 74 61  XX XX XX XX              |....data....    |
*/
static const unsigned char wave_hdr[44] = {
    'R' , 'I', 'F'  , 'F', 0x2c , 0   , 0    , 0   , 'W' , 'A' , 'V' , 'E' , 'f' , 'm' , 't', ' ',
    0x10, 0  , 0    , 0  , 1    , 0   , 1    , 0   , 9   , 0x3d, 0   , 0   , 0x12, 0x7a, 0  , 0  ,
    2   , 0  , 0x10 , 0  , 'd'  , 'a' , 't'  , 'a' , 0   , 0   , 0   , 0
};

// write an uint32_t as a little endian int to file
// that is, least significant byte first
void _write_uint32_t(FILE *f, int value) {
	int ix;
	for (ix = 0; ix < 4; ix++) {
		fputc(value & 0xff, f);
		value = value >> 8;
	}
}

// open wave file and write its header
// NOTE: the uint32_t representing the file size and data size
//       will be set by _close_wave_file()
//       once all audio samples are generated
int _open_wave_file(char const *path, int rate) {
	if (path == NULL) {
        return CEW_FAILURE;
    }

	while (isspace(*path)) {
        path++;
    }

	wave_file = NULL;
	if (path[0] != 0) {
        //printf("%s", path);
        wave_file = fopen(path, "wb");
	}
	
	if (wave_file == NULL) {
        return CEW_FAILURE;
	}

	fwrite(wave_hdr, 1, 24, wave_file);
	_write_uint32_t(wave_file, rate);
	_write_uint32_t(wave_file, rate * 2);
	fwrite(&wave_hdr[32], 1, 12, wave_file);
	return CEW_SUCCESS;
}

// close wave file
int _close_wave_file(void) {
	long pos;

	if (wave_file == NULL) {
        return CEW_FAILURE;
    }

    // flush and get the current position,
    // which is the file length
	fflush(wave_file);
	pos = ftell(wave_file);

    // set file size at byte #4
	fseek(wave_file, 4, SEEK_SET);
	_write_uint32_t(wave_file, pos - 8);

    // set data size at byte #40
	fseek(wave_file, 40, SEEK_SET);
	_write_uint32_t(wave_file, pos - 44);

    // close file
	fclose(wave_file);
	wave_file = NULL;

    return CEW_SUCCESS;
}

// callback for synth events
int _synth_callback(short *wav, int numsamples, espeak_EVENT *events) {
	if (wav == NULL) {
        return CEW_FAILURE;
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
    return CEW_SUCCESS;
}

// terminate synthesis and close file
int _terminate_synthesis(void) {
    espeak_Terminate();
    return _close_wave_file();
}

// synthesize the given string
int _synthesize_string(char const *text) {
	int size;
    if (text != NULL) {
        size = strlen(text);
        espeak_Synth(text, size + 1, 0, POS_CHARACTER, 0, synth_flags, NULL, NULL);
	}
	if (espeak_Synchronize() != EE_OK) {
        return CEW_FAILURE;
	}
    current_time += last_end_time;
    return CEW_SUCCESS;
}

// set the current language
// NOTE: using espeak_SetVoiceByProperties()
//       to allow voice variants like 'en-us'
int _set_voice_code(char const *voice_code) {
    espeak_VOICE voice;
    memset(&voice, 0, sizeof(voice));
    voice.languages = voice_code;
    if (espeak_SetVoiceByProperties(&voice) != EE_OK) {
        return CEW_FAILURE;
    }
    return CEW_SUCCESS;
}

// initialize the synthesizer
int _initialize_synthesizer(char const *output_file_path) {
    char *data_path;

    // data_path is the path to espeak data for additional voices
    // NULL = use default path for espeak-data
	data_path = NULL;

    // set sentinel sample_rate
    sample_rate = 0;

    // synthesizer flags
    // TODO let the user control espeakENDPAUSE
	synth_flags = espeakCHARS_UTF8 | espeakENDPAUSE;

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
        if(_open_wave_file(output_file_path, sample_rate) != CEW_SUCCESS) {
            return CEW_FAILURE;
        }
	}

    // reset time
    current_time = 0.0;
    last_end_time = 0.0;

    return CEW_SUCCESS;
}

// synthesize a single text fragment
int _synthesize_single(
        const char *output_file_path,
        int *sample_rate_ret,
        struct FRAGMENT_INFO *fragment_ret
    ) {

    // open output wave file
    if (_initialize_synthesizer(output_file_path) != CEW_SUCCESS) {
        return CEW_FAILURE;
    }

    // set voice code
    if (_set_voice_code((*fragment_ret).voice_code) != CEW_SUCCESS) {
        return CEW_FAILURE;
    }

    // synthesize text
    *sample_rate_ret = sample_rate;
    (*fragment_ret).begin = current_time;
    if (_synthesize_string((*fragment_ret).text) != CEW_SUCCESS) {
        return CEW_FAILURE;
    }
    (*fragment_ret).end = current_time;

    // close output wave file
    _terminate_synthesis();

    return CEW_SUCCESS;
}

// synthesize multiple fragments
int _synthesize_multiple(
        const char *output_file_path,
        struct FRAGMENT_INFO **fragments_ret,
        const size_t number_of_fragments,
        const float quit_after,
        const int backwards,
        int *sample_rate_ret,
        size_t *synthesized_ret
    ) {

    size_t i, synthesized, start;

    start = 0;

    if ((backwards != 0) && (quit_after > 0)) {
        // synthesize a first time to determine how many fragments
        // from the back we need to reach quit_after seconds of audio

        // open output wave file
        if (_initialize_synthesizer(output_file_path) != CEW_SUCCESS) {
            return CEW_FAILURE;
        }

        // synthesize from the back
        for (i = number_of_fragments - 1; ; --i) {
            if (_set_voice_code((*fragments_ret)[i].voice_code) != CEW_SUCCESS) {
                return CEW_FAILURE;
            }
            if (_synthesize_string((*fragments_ret)[i].text) != CEW_SUCCESS) {
                return CEW_FAILURE;
            }
            start = i;
            // check if we generated >= quit_after seconds of audio
            if (current_time >= quit_after) {
                break;
            }
            // end of the loop, checked here because i is size_t i.e. unsigned!
            if (i == 0) {
                break;
            }
        }

        // close output wave file
        _terminate_synthesis();
    }

    // open output wave file
    if (_initialize_synthesizer(output_file_path) != CEW_SUCCESS) {
        return CEW_FAILURE;
    }

    // number of synthesized fragments
    synthesized = 0;

    // loop over all input fragments
    for (i = start; i < number_of_fragments; ++i) {
        if (_set_voice_code((*fragments_ret)[i].voice_code) != CEW_SUCCESS) {
            return CEW_FAILURE;
        }

        // NOTE: if backwards, we move the anchor times to the first fragments,
        //       so that looping from 0 to synthesized will give the correct anchors
        //       despite the fact that they will not be saved with the "correct" text
        //       this trick avoids copying data around
        //       if backwards, the user is not expected to use the time anchors anyway
        (*fragments_ret)[i-start].begin = current_time;
        if (_synthesize_string((*fragments_ret)[i].text) != CEW_SUCCESS) {
            return CEW_FAILURE;
        }
        (*fragments_ret)[i-start].end = current_time;
        synthesized += 1;

        // check if we generated >= quit_after seconds of audio
        if ((quit_after > 0) && (current_time >= quit_after)) {
            break;
        }
    }

    // close output wave file
    _terminate_synthesis();

    // save values to be returned
    *sample_rate_ret = sample_rate;
    *synthesized_ret = synthesized;

    return CEW_SUCCESS;
}



