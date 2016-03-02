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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cew_func.h"

#define DRIVER_SUCCESS 0
#define DRIVER_FAILURE 1

// print usage
void _usage(const char *prog) {
    printf("\n");
    printf("Usage:   %s VOICE_CODE TEXT AUDIO_FILE.wav single\n", prog);
    printf("         %s VOICE_CODE TEXT AUDIO_FILE.wav multi QUIT_AFTER BACKWARDS\n", prog);
    printf("\n");
    printf("Example: %s en \"Hello World\" /tmp/out.wav single\n", prog);
    printf("         %s en \"Hello|World|My|Dear|Friend\" /tmp/out.wav multi 0.0 0\n", prog);
    printf("         %s en \"Hello|World|My|Dear|Friend\" /tmp/out.wav multi 0.0 1\n", prog);
    printf("         %s en \"Hello|World|My|Dear|Friend\" /tmp/out.wav multi 2.0 1\n", prog);
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

    const char *voice_code, *text, *output_file_name, *mode;
    int sample_rate_ret, backwards;
    struct FRAGMENT_INFO fragment;
    float quit_after;
    struct FRAGMENT_INFO *fragments;
    char **texts;
    int i, n;
    size_t synthesized_ret;

    if (argc < 5) {
        _usage(argv[0]);
        return DRIVER_FAILURE;
    }
    voice_code = argv[1];
    text = argv[2];
    output_file_name = argv[3];
    mode = argv[4];

    if (strcmp(mode, "multi") == 0) {
        if (argc < 7) {
            _usage(argv[0]);
            return DRIVER_FAILURE;
        }
        quit_after = (float)atof(argv[5]);
        backwards = atoi(argv[6]);

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
            printf("Error while calling _synthesize_single()\n");
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
    } else {
        fragment.voice_code = voice_code;
        fragment.text = text;
        if (_synthesize_single(output_file_name, &sample_rate_ret, &fragment) != CEW_SUCCESS) {
            printf("Error while calling _synthesize_single()\n");
            return DRIVER_FAILURE;
        }
        printf("Sample rate: %d\n", sample_rate_ret);
        printf("Begin:       %.3f\n", fragment.begin);
        printf("End:         %.3f\n", fragment.end);
    }

    return DRIVER_SUCCESS;
}

