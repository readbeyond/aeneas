/*

Python C Extension for computing the MFCC

__author__ = "Alberto Pettarin"
__copyright__ = """
    Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
    Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
    Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
    """
__license__ = "GNU AGPL v3"
__version__ = "1.4.1"
__email__ = "aeneas@readbeyond.it"
__status__ = "Production"

*/

//
// this is a simple driver to test on the command line
//
// you can compile it with:
//
// $ gcc cwave_driver.c cwave_func.c -o cwave_driver
//
// use it as follows:
//
// ./cwave_driver audio.wav       => print info about the WAVE file
// ./cwave_driver audio.wav 0 100 => print the value of the first 100 samples, as (signed) double
// ./cwave_driver audio.wav 25 75 => print the value of the samples with index (starting at 0) 25-99, as (signed) double
//

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "cwave_func.h"

int main(int argc, char **argv) {
    FILE *audio_file_ptr;
    struct WAVE_INFO audio_info;
    char *filename;
    double *buffer;
    double duration;
    unsigned int i, from_sample, num_samples;

    if (argc < 2) {
        printf("\nUsage: $ %s AUDIO.wav [FROM_SAMPLE] [NUM_SAMPLES]\n\n", argv[0]);
        return 1;
    }
    filename = argv[1];
    from_sample = 0;
    num_samples = 0;
    if (argc >= 4) {
        from_sample = atol(argv[2]);
        num_samples = atol(argv[3]);
    }

    memset(&audio_info, 0, sizeof(audio_info));
    if (!(audio_file_ptr = wave_open(filename, &audio_info))) {
        printf("Error: cannot open file %s\n", filename);
        return 1;
    }
    duration = 1.0 * audio_info.coNumSamples / audio_info.leSampleRate;

    if (num_samples > 0) {
        buffer = (double *)calloc(num_samples, sizeof(double));
        if (!wave_read_double(audio_file_ptr, &audio_info, buffer, from_sample, num_samples)) {
            printf("Error: cannot read the specified range: %u %u\n", from_sample, num_samples);
            free((void *)buffer);
            buffer = NULL;
            return 1;
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
    return 0;
}
