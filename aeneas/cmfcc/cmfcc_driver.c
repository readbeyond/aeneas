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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "cmfcc_func.h"

#if USE_SNDFILE
#include <sndfile.h>
#else
#include "cwave_func.h"
#endif

#define DRIVER_SUCCESS 0
#define DRIVER_FAILURE 1

// print usage
void _usage(const char *prog) {
    printf("\n");
    printf("Usage:   %s AUDIO_FILE.wav OUTPUT.bin [data|file] [text|binary]\n", prog);
    printf("\n");
    printf("Example: %s ../tools/res/audio.wav /tmp/out.dt.bin data text\n", prog);
    printf("         %s ../tools/res/audio.wav /tmp/out.db.bin data binary\n", prog);
    printf("         %s ../tools/res/audio.wav /tmp/out.ft.bin file text\n", prog);
    printf("         %s ../tools/res/audio.wav /tmp/out.fb.bin file binary\n", prog);
    printf("\n");
}

int main(int argc, char **argv) {

#if USE_SNDFILE
    SNDFILE *audio_file;
    SF_INFO audio_info;
#else
    FILE *audio_file;
    struct WAVE_INFO audio_info;
#endif

    char *audio_file_name, *output_file_name, *mode, *output_format;
    double *data_ptr, *mfcc_ptr;
    FILE *output_file;
    uint32_t sample_rate;
    uint32_t data_length, mfcc_length;
    uint32_t i, j;

    const uint32_t filter_bank_size = 40;
    const uint32_t mfcc_size = 13;
    const uint32_t fft_order = 512;
    const double lower_frequency = 133.3333;
    const double upper_frequency = 6855.4976;
    const double emphasis_factor = 0.97;
    const double window_length = 0.100;
    const double window_shift = 0.040;

    if (argc < 5) {
        _usage(argv[0]);
        return DRIVER_FAILURE;
    }
    audio_file_name = argv[1];
    output_file_name = argv[2];
    mode = argv[3];
    output_format = argv[4];

#ifdef USE_SNDFILE
    printf("Reading WAVE file with libsndfile\n");
#else
    printf("Reading WAVE file with own code\n");
#endif

#ifdef USE_FFTW
    printf("Computing the RFFT with FFTW\n");
#else
    printf("Computing the RFFT with own code\n");
#endif

    if (strcmp(mode, "data") == 0) {
        // load file in RAM
        // using libsndfile for this
        printf("Reading audio file in RAM...\n");

#if USE_SNDFILE
        memset(&audio_info, 0, sizeof(audio_info));
        if (!(audio_file = sf_open(audio_file_name, SFM_READ, &audio_info))) {
            printf("Error: unable to open input file %s.\n", audio_file_name);
            puts(sf_strerror(NULL));
            return DRIVER_FAILURE;
        }
        data_length = audio_info.frames;
        sample_rate = audio_info.samplerate;
        data_ptr = (double *)calloc(data_length, sizeof(double));
        sf_read_double(audio_file, data_ptr, audio_info.frames);
        sf_close(audio_file);
#else
        if (!(audio_file = wave_open(audio_file_name, &audio_info))) {
            printf("Error: unable to open input file %s.\n", audio_file_name);
            return DRIVER_FAILURE;
        }
        data_length = audio_info.coNumSamples;
        sample_rate = audio_info.leSampleRate;
        data_ptr = (double *)calloc(data_length, sizeof(double));
        wave_read_double(audio_file, &audio_info, data_ptr, 0, data_length);
        wave_close(audio_file);
#endif
        printf("Reading audio file in RAM... done\n");

        printf("Computing MFCC from data...\n");
        compute_mfcc_from_data(
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
            &mfcc_length
        );
        printf("Computing MFCC from data... done\n");

        free((void *)data_ptr);
        data_ptr = NULL;
    } else {
        // compute directly from file
        printf("Computing MFCC from file...\n");
        compute_mfcc_from_file(
            audio_file_name,
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
            &mfcc_length
        );
        printf("Computing MFCC from file... done\n");
    }

    printf("Audio file:        %s\n", audio_file_name);
    printf("Number of samples: %u\n", data_length);
    printf("Sample rate:       %u\n", sample_rate);
    printf("MFCC size:         %u\n", mfcc_size);
    printf("MFCC length:       %u\n", mfcc_length);

    // output result to file
    output_file = fopen(output_file_name, "w");
    if (strcmp(output_format, "text") == 0) {
        for (i = 0; i < mfcc_size; ++i) {
            for (j = 0; j < mfcc_length; ++j) {
                // print transposed as a (mfcc_size, mfcc_length) matrix
                fprintf(output_file, "%.18e", mfcc_ptr[j * mfcc_size + i]);
                if (j < mfcc_length - 1) {
                    fprintf(output_file, " ");
                }
            }
            fprintf(output_file, "\n");
        }
    } else {
        // written as a (mfcc_length, mfcc_size matrix)
        fwrite(mfcc_ptr, sizeof(double), mfcc_size * mfcc_length, output_file);
    }
    fclose(output_file);
    printf("Output file:       %s\n", output_file_name);

    free((void *)mfcc_ptr);
    mfcc_ptr = NULL;

    return DRIVER_SUCCESS;
}

