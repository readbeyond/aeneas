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

Python C Extension for reading WAVE mono files.

*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "cwave_func.h"

#define DRIVER_SUCCESS 0
#define DRIVER_FAILURE 1

// print usage
void _usage(const char *prog) {
    printf("\n");
    printf("Usage: $ %s AUDIO.wav [FROM_SAMPLE] [NUM_SAMPLES]\n", prog);
    printf("\n");
    printf("Example: %s ../tools/res/audio.wav\n", prog);
    printf("         %s ../tools/res/audio.wav 0 100\n", prog);
    printf("         %s ../tools/res/audio.wav 25 75\n", prog);
    printf("\n");
}

int main(int argc, char **argv) {
    FILE *audio_file_ptr;
    struct WAVE_INFO audio_info;
    char *filename;
    double *buffer;
    double duration;
    uint32_t i, from_sample, num_samples;   // a WAVE file cannot have more 2^32 samples

    // parse arguments
    if (argc < 2) {
        _usage(argv[0]);
        return DRIVER_FAILURE;
    }
    filename = argv[1];
    from_sample = 0;
    num_samples = 0;
    if (argc >= 4) {
        from_sample = atol(argv[2]);
        num_samples = atol(argv[3]);
    }

    audio_file_ptr = wave_open(filename, &audio_info);
    if (audio_file_ptr == NULL) {
        printf("Error: cannot open file %s\n", filename);
        return DRIVER_FAILURE;
    }
    duration = 1.0 * audio_info.coNumSamples / audio_info.leSampleRate;

    if (num_samples > 0) {
        buffer = (double *)calloc(num_samples, sizeof(double));
        if (buffer == NULL) {
            printf("Error: cannot allocate buffer\n");
            return DRIVER_FAILURE;
        }
        if (wave_read_double(audio_file_ptr, &audio_info, buffer, from_sample, num_samples) != CWAVE_SUCCESS) {
            printf("Error: cannot read the specified range: %u %u\n", from_sample, num_samples);
            free((void *)buffer);
            buffer = NULL;
            return DRIVER_FAILURE;
        }
        for (i = 0; i < num_samples; ++i) {
            printf("%.12f\n", buffer[i]);
        }
        free((void *)buffer);
        buffer = NULL;
    } else {
        printf("Format:             %u\n", audio_info.leAudioFormat);
        printf("Channels:           %u\n", audio_info.leNumChannels);
        printf("Bits per sample:    %u\n", audio_info.leBitsPerSample);
        printf("Sample rate:        %u\n", audio_info.leSampleRate);
        printf("Number of samples:  %u\n", audio_info.coNumSamples);
        printf("Duration:           %f\n", duration);
        printf("Data starts at pos: %u\n", audio_info.coSubchunkDataStart);
    }

    wave_close(audio_file_ptr);
    return DRIVER_SUCCESS;
}
