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

#define _POSIX_C_SOURCE 200809L

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cew_func.h"

#define DRIVER_SUCCESS 0
#define DRIVER_FAILURE 1

// print usage
void _usage(const char *prog) {
    printf("\n");
    printf("Usage:   %s VOICE_CODE TEXT AUDIO_FILE.wav QUIT_AFTER BACKWARDS\n", prog);
    printf("\n");
    printf("Example: %s en \"Hello|World|My|Dear|Friend\" /tmp/out.wav 0.0 0\n", prog);
    printf("         %s en \"Hello|World|My|Dear|Friend\" /tmp/out.wav 0.0 1\n", prog);
    printf("         %s en \"Hello|World|My|Dear|Friend\" /tmp/out.wav 2.0 0\n", prog);
    printf("         %s en \"Hello|World|My|Dear|Friend\" /tmp/out.wav 2.0 1\n", prog);
    printf("\n");
}

// split a given string using a delimiter character
// adapted from
// http://stackoverflow.com/questions/9210528/split-string-with-delimiters-in-c
char **_str_split(char* a_str, const char a_delim, int *count) {
    char** result    = 0;
    char* tmp        = a_str;
    char* last_delim = 0;
    char delim[2];
    delim[0] = a_delim;
    delim[1] = 0;

    // count how many elements will be extracted
    while (*tmp) {
        if (a_delim == *tmp) {
            ++(*count);
            last_delim = tmp;
        }
        ++tmp;
    }

    // add space for trailing token
    (*count) += last_delim < (a_str + strlen(a_str) - 1);

    // tokenize
    result = calloc((*count), sizeof(char*));
    if (result) {
        size_t idx  = 0;
        char* token = strtok(a_str, delim);

        while (token) {
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
    }

    return result;
}

int main(int argc, char **argv) {

    const char *voice_code, *text, *output_file_name;
    int sample_rate_ret, backwards;
    struct FRAGMENT_INFO fragment;
    float quit_after;
    struct FRAGMENT_INFO *fragments;
    char **texts;
    int i, n;
    size_t synthesized_ret;

    if (argc < 6) {
        _usage(argv[0]);
        return DRIVER_FAILURE;
    }
    voice_code = argv[1];
    text = argv[2];
    output_file_name = argv[3];
    quit_after = (float)atof(argv[4]);
    backwards = atoi(argv[5]);

    // split text into fragments
    n = 0;
    texts = _str_split((char *)text, '|', &n);

    // create fragments
    fragments = (struct FRAGMENT_INFO *)calloc(sizeof(fragment), n);
    for (i = 0; i < n; ++i) {
        fragments[i].voice_code = voice_code;
        fragments[i].text = texts[i];
    }

    // synthesize
    if(_synthesize_multiple(
        output_file_name,
        &fragments,
        n,
        quit_after,
        backwards,
        &sample_rate_ret,
        &synthesized_ret
        ) != CEW_SUCCESS) {
        printf("Error while calling _synthesize_multiple()\n");
        return DRIVER_FAILURE;
    }
    printf("Sample rate: %d\n", sample_rate_ret);
    printf("Synthesized: %lu\n", synthesized_ret);
    for (i = 0; i < synthesized_ret; ++i) {
        printf("%d %.3f %.3f\n", i, fragments[i].begin, fragments[i].end);
    }

    // deallocate
    for (i = 0; i < n; ++i) {
        free((void *)fragments[i].text);
    }
    free((void *)fragments);
    free((void *)texts);
    fragments = NULL;
    texts = NULL;

    return DRIVER_SUCCESS;
}

