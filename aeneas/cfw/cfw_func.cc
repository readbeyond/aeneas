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

Python C Extension for synthesizing text with Festival

*/

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "festival/festival.h"
#include "cfw_func.h"

#ifndef CFW_DIRECT_WRITE
#define CFW_DIRECT_WRITE CFW_TRUE
#endif

static int initialized = CFW_FALSE;

#if CFW_DIRECT_WRITE
#include "EST_walloc.h"

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
        return CFW_FAILURE;
    }

	while (isspace(*path)) {
        path++;
    }

	wave_file = NULL;
	if (path[0] != 0) {
        wave_file = fopen(path, "wb");
	}
	
	if (wave_file == NULL) {
        return CFW_FAILURE;
	}

	fwrite(wave_hdr, 1, 24, wave_file);
	_write_uint32_t(wave_file, rate);
	_write_uint32_t(wave_file, rate * 2);
	fwrite(&wave_hdr[32], 1, 12, wave_file);
	return CFW_SUCCESS;
}

// close wave file
int _close_wave_file(void) {
	long pos;

	if (wave_file == NULL) {
        return CFW_FAILURE;
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

    return CFW_SUCCESS;
}
#endif

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

#if CFW_DIRECT_WRITE
    // nothing
#else
    EST_Wave wave;
    //printf("\n\nUsing EST_Wave\n\n\n");
#endif

    float current_time;
    size_t i, synthesized, start;
    int sample_rate;

    start = 0;

    // suppress warning
    sample_rate = 16000;

    // festival initialize can be called only once
    // and it returns void (no status code is returned)
    if (initialized == CFW_FALSE) {
        festival_initialize(CFW_LOAD_INIT_FILES, CFW_HEAP_SIZE);
        initialized = CFW_TRUE;
    }

    if ((backwards != 0) && (quit_after > 0)) {
        // synthesize a first time to determine how many fragments
        // from the back we need to reach quit_after seconds of audio

        // temporary wave
        EST_Wave wave_tmp;
        current_time = 0.0;

        // synthesize from the back
        for (i = number_of_fragments - 1; ; --i) {

            // set voice
            if (festival_eval_command((*fragments_ret)[i].voice_code) != CFW_FESTIVAL_SUCCESS) {
                return CFW_FAILURE;
            }

            // synthesize
            if (festival_text_to_wave((*fragments_ret)[i].text, wave_tmp) != CFW_FESTIVAL_SUCCESS) {
                return CFW_FAILURE;
            }
            // update current time
            current_time += ((float)wave_tmp.num_samples()) / wave_tmp.sample_rate();
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
    }

    // number of synthesized fragments
    synthesized = 0;

    // reset time
    current_time = 0.0;

#if CFW_DIRECT_WRITE
    // open wave file
    if (wave_file == NULL) {
        // temporary wave
        EST_Wave wave_tmp;
        // synthesize dummy fragment to get the sample rate
        if (festival_text_to_wave("dummy text", wave_tmp) != CFW_FESTIVAL_SUCCESS) {
            return CFW_FAILURE;
        }
        sample_rate = wave_tmp.sample_rate();
        // open wave file
        if (_open_wave_file(output_file_path, sample_rate) != CFW_SUCCESS) {
            return CFW_FAILURE;
        }
	}
#endif

    // loop over all input fragments
    for (i = start; i < number_of_fragments; ++i) {

        // temporary wave
        EST_Wave wave_tmp;

        // set voice
        if (festival_eval_command((*fragments_ret)[i].voice_code) != CFW_FESTIVAL_SUCCESS) {
            return CFW_FAILURE;
        }

        // NOTE: if backwards, we move the anchor times to the first fragments,
        //       so that looping from 0 to synthesized will give the correct anchors
        //       despite the fact that they will not be saved with the "correct" text
        //       this trick avoids copying data around
        //       if backwards, the user is not expected to use the time anchors anyway

        // set begin time
        (*fragments_ret)[i-start].begin = current_time;
        // synthesize
        if (festival_text_to_wave((*fragments_ret)[i].text, wave_tmp) != CFW_FESTIVAL_SUCCESS) {
            return CFW_FAILURE;
        }
        // update current time
        current_time += ((float)wave_tmp.num_samples()) / wave_tmp.sample_rate();
        (*fragments_ret)[i-start].end = current_time;

        // append audio data
#if CFW_DIRECT_WRITE
        int numsamples = wave_tmp.num_samples();
        short *buffer = (short *)safe_wcalloc(numsamples * sizeof(short));
        wave_tmp.copy_channel(0, buffer);
        fwrite(buffer, numsamples * 2, 1, wave_file);
        wfree((void *)buffer);
#else
        // NOTE: EST_Wave has a concat operator that allows:
        //       wave_accumulator += wave_tmp;
        //       but unfortunately it does a realloc,
        //       so it is very slow when used on many (> 100) fragments.
        wave += wave_tmp;
#endif

        // increase number of synthesized fragments
        synthesized += 1;

        // check if we generated >= quit_after seconds of audio
        if ((quit_after > 0) && (current_time >= quit_after)) {
            break;
        }
    }

#if CFW_DIRECT_WRITE
    // close wave file
    if (_close_wave_file() != CFW_SUCCESS) {
        return CFW_FAILURE;
    }
#else
    // output wave file
    wave.save(output_file_path, "riff");
    sample_rate = wave.sample_rate();
#endif

    // save values to be returned
    *sample_rate_ret = sample_rate;
    *synthesized_ret = synthesized;

    return CFW_SUCCESS;
}



