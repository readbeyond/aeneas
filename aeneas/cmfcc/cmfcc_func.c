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

#define MEL_10 2595.0
#define PI     3.1415926535897932384626433832795
#define PI_2   6.2831853071795864769252867665590
#define CUTOFF 0.00001

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <string.h>

#include "cmfcc_func.h"
#include "cwave_func.h"

#ifdef USE_FFTW
#include <rfftw.h>
#endif

// return the min of the given arguments
uint32_t _min(uint32_t a, uint32_t b) {
    if (a < b) {
        return a;
    }
    return b;
}

// return the max of the given arguments
uint32_t _max(uint32_t a, uint32_t b) {
    if (a > b) {
        return a;
    }
    return b;
}

// round the given number to the nearest integer
// or return zero if the argument is negative
// e.g.: 1.1 => 1; 1.6 => 2
uint32_t _round(double x) {
    if (x < 0.0) {
        return 0; //printf("Error: _round argument is negative!!!\n");
    }
    return (uint32_t)floor(x + 0.5);
}

// precompute the sin table for the FFT/RFFT
double *_precompute_sin_table(uint32_t m) {
    const double arg = PI / m * 2;
    const uint32_t size = m - m / 4 + 1;
    double *table;
    int k;

    table = (double *)calloc(size, sizeof(double));
    if (table == NULL) {
        return NULL;
    }
    table[0] = 0;
    for (k = 1; k < size; ++k) {
        table[k] = sin(arg * k);
    }
    table[m / 2] = 0;
    return table;
}
int fft(double *x, double *y, const uint32_t m, double *sin_table) {
    // code adapted from the fft function of SPTK
    double t1, t2;
    double *cosp, *sinp, *xp, *yp;
    int j, lf, li, lix, lmx, mm1, mv2;

    lf = 1;
    lmx = m;

    for (;;) {
        lix = lmx;
        lmx /= 2;
        if (lmx <= 1) {
            break;
        }
        sinp = sin_table;
        cosp = sin_table + m / 4;
        for (j = 0; j < lmx; j++) {
            xp = &x[j];
            yp = &y[j];
            for (li = lix; li <= m; li += lix) {
                t1 = *(xp) - *(xp + lmx);
                t2 = *(yp) - *(yp + lmx);
                *(xp) += *(xp + lmx);
                *(yp) += *(yp + lmx);
                *(xp + lmx) = *cosp * t1 + *sinp * t2;
                *(yp + lmx) = *cosp * t2 - *sinp * t1;
                xp += lix;
                yp += lix;
            }
            sinp += lf;
            cosp += lf;
        }
        lf += lf;
    }

    xp = x;
    yp = y;
    for (li = m / 2; li--; xp += 2, yp += 2) {
        t1 = *(xp) - *(xp + 1);
        t2 = *(yp) - *(yp + 1);
        *(xp) += *(xp + 1);
        *(yp) += *(yp + 1);
        *(xp + 1) = t1;
        *(yp + 1) = t2;
    }

    j = 0;
    xp = x;
    yp = y;
    mv2 = m / 2;
    mm1 = m - 1;
    for (lmx = 0; lmx < mm1; lmx++) {
        if ((li = lmx - j) < 0) {
            t1 = *(xp);
            t2 = *(yp);
            *(xp) = *(xp + li);
            *(yp) = *(yp + li);
            *(xp + li) = t1;
            *(yp + li) = t2;
        }
        li = mv2;
        while (li <= j) {
            j -= li;
            li /= 2;
        }
        j += li;
        xp = x + j;
        yp = y + j;
    }

    return CMFCC_SUCCESS;
}
int rfft(
        double *x,
        double *y,
        const uint32_t m,
        double *sin_table_full,
        double *sin_table_half
    ) {
    // code adapted from the fftr function of SPTK
    int i, j, mv2;
    double xt, yt;
    double *cosp, *sinp, *xp, *xq, *yp, *yq;

    mv2 = m / 2;
    xq = xp = x;
    yp = y;
    for (i = mv2; --i >= 0;) {
        *xp++ = *xq++;
        *yp++ = *xq++;
    }

    if (fft(x, y, mv2, sin_table_half) == -1) {
        return CMFCC_FAILURE;
    }

    sinp = sin_table_full;
    cosp = sin_table_full + m / 4;
    xp = x;
    yp = y;
    xq = xp + m;
    yq = yp + m;
    *(xp + mv2) = *xp - *yp;
    *xp = *xp + *yp;
    *(yp + mv2) = *yp = 0;
    for (i = mv2, j = mv2 - 2; --i; j -= 2) {
        ++xp;
        ++yp;
        ++sinp;
        ++cosp;
        yt = *yp + *(yp + j);
        xt = *xp - *(xp + j);
        *(--xq) = (*xp + *(xp + j) + *cosp * yt - *sinp * xt) * 0.5;
        *(--yq) = (*(yp + j) - *yp + *sinp * yt + *cosp * xt) * 0.5;
    }

    xp = x + 1;
    yp = y + 1;
    xq = x + m;
    yq = y + m;

    for (i = mv2; --i;) {
        *xp++ = *(--xq);
        *yp++ = -(*(--yq));
    }

    return CMFCC_SUCCESS;
}

// convert Hz frequency to Mel frequency
double _hz2mel(const double f) {
    return MEL_10 * log10(1.0 + (f / 700.0));
}

// convert Mel frequency to Hz frequency
double _mel2hz(const double m) {
    return 700.0 * (pow(10.0, m / MEL_10) - 1.0);
}

// pre emphasis of the given frame
// returns the prior to be used for the next frame
int _apply_emphasis(
        double *frame,
        const uint32_t length,
        const double emphasis_factor,
        double *prior
    ) {
    double prior_orig;
    double *frame_orig;
    uint32_t i;

    prior_orig = frame[length - 1];
    frame_orig = (double *)calloc(length, sizeof(double));
    if (frame_orig == NULL) {
        return CMFCC_FAILURE;
    }
    memcpy(frame_orig, frame, length * sizeof(double));
    frame[0] = frame_orig[0] - emphasis_factor * (*prior);
    for (i = 1; i < length; ++i) {
        frame[i] = frame_orig[i] - frame_orig[i-1] * emphasis_factor;
    }
    free((void *)frame_orig);
    frame_orig = NULL;
    *prior = prior_orig;
    return CMFCC_SUCCESS;
}

// own code
// compute the power of the given frame
int _compute_power(
        double *frame,          // it has length == fft_order
        double *power,          // power has length == (fft_order / 2) + 1
        const uint32_t fft_order,
        double *sin_table_full,
        double *sin_table_half
    ) {
    double *tmp;
    uint32_t k;
    const uint32_t n = fft_order;           // length of the I/O vectors
    const uint32_t m = (fft_order / 2) + 1; // length of power

    tmp = (double *)calloc(n + m, sizeof(double));
    if (tmp == NULL) {
        return CMFCC_FAILURE;
    }
    rfft(frame, tmp, fft_order, sin_table_full, sin_table_half);
    power[0] = frame[0] * frame[0];
    for (k = 1; k < m; ++k) {
        power[k] = frame[k] * frame[k] + tmp[k] * tmp[k];
    }
    free((void *)tmp);
    tmp = NULL;
    return CMFCC_SUCCESS;
}

#ifdef USE_FFTW
// fftw code
// compute the power of the given frame
int _compute_power_fftw(
        double *frame,          // it has length == fft_order
        double *power,          // power has length == (fft_order / 2) + 1
        const uint32_t fft_order,
        rfftw_plan plan
    ) {

    uint32_t k;
    double *out;
    const uint32_t n = fft_order;             // length of the I/O vectors
    //const uint32_t m = (fft_order / 2) + 1; // length of power

    out = (double *)calloc(n, sizeof(double));
    if (out == NULL) {
        return CMFCC_FAILURE;
    }
    rfftw_one(plan, frame, out);
    power[0] = out[0] * out[0];
    for (k = 1; k < (n+1)/2; ++k) {
        power[k] = out[k] * out[k] + out[n-k] * out[n-k];
    }
    if (n % 2 == 0) {
        power[n/2] = out[n/2] * out[n/2];
    }
    free((void *)out);
    out = NULL;
    return CMFCC_SUCCESS;
}
#endif

// transform the frame using the Hamming window
int _apply_hamming(
        double *frame,
        const uint32_t frame_length,
        double *coefficients
    ) {
    uint32_t k;

    for (k = 0; k < frame_length; ++k) {
        frame[k] *= coefficients[k];
    }
    return CMFCC_SUCCESS;
}
double *_precompute_hamming(const uint32_t frame_length) {
    const double arg = PI_2 / (frame_length - 1);
    double *coefficients;
    uint32_t k;

    coefficients = (double *)calloc(frame_length, sizeof(double));
    if (coefficients == NULL) {
        return NULL;
    }
    for (k = 0; k < frame_length; ++k) {
        coefficients[k] = (0.54 - 0.46 * cos(k * arg));
    }
    return coefficients;
}

// create Mel filter bank
// return a pointer to a 2D matrix (filters_n x filter_bank_size)
double *_create_mel_filter_bank(
        uint32_t fft_order,
        uint32_t filter_bank_size,
        uint32_t sample_rate,
        double upper_frequency,
        double lower_frequency
    ) {
    const double step_frequency = (1.0 * sample_rate) / fft_order;
    const double melmax = _hz2mel(upper_frequency);
    const double melmin = _hz2mel(lower_frequency);
    const double melstep = (melmax - melmin) / (filter_bank_size + 1);
    const uint32_t filter_edge_length = filter_bank_size + 2;
    const uint32_t filters_n = (fft_order / 2) + 1;
    double *filter_edges, *filters;
    uint32_t k;

    // filter bank
    filters = (double *)calloc(filters_n * filter_bank_size, sizeof(double));
    if (filters == NULL) {
        return NULL;
    }

    // filter edges
    filter_edges = (double *)calloc(filter_edge_length, sizeof(double));
    if (filter_edges == NULL) {
        return NULL;
    }

    for (k = 0; k < filter_edge_length; ++k) {
        filter_edges[k] = _mel2hz(melmin + melstep * k);
    }
    for (k = 0; k < filter_bank_size; ++k) {
        const uint32_t left_frequency = _round(filter_edges[k] / step_frequency);
        const uint32_t center_frequency = _round(filter_edges[k + 1] / step_frequency);
        const uint32_t right_frequency = _round(filter_edges[k + 2] / step_frequency);
        const double width_frequency = (right_frequency - left_frequency) * step_frequency;
        const double height_frequency = 2.0 / width_frequency;
        double left_slope, right_slope;
        uint32_t current_frequency;

        left_slope = 0.0;
        if (center_frequency != left_frequency) {
            left_slope = height_frequency / (center_frequency - left_frequency);
        }
        current_frequency = left_frequency + 1;
        while (current_frequency < center_frequency) {
            filters[current_frequency * filter_bank_size + k] = (current_frequency - left_frequency) * left_slope;
            ++current_frequency;
        }
        if (current_frequency == center_frequency) {
            filters[current_frequency * filter_bank_size + k] = height_frequency;
            ++current_frequency;
        }
        right_slope = 0.0;
        if (center_frequency != right_frequency) {
            right_slope = height_frequency / (center_frequency - right_frequency);
        }
        while (current_frequency < right_frequency) {
            filters[current_frequency * filter_bank_size + k] = (current_frequency - right_frequency) * right_slope;
            ++current_frequency;
        }
    }
    free((void *)filter_edges);
    filter_edges = NULL;
    return filters;
}

// create the DCT matrix
// return a pointer to a 2D matrix (mfcc_size x filter_bank_size)
double *_create_dct_matrix(uint32_t mfcc_size, uint32_t filter_bank_size) {
    double *s2dct;
    uint32_t i, j;

    s2dct = (double *)calloc(mfcc_size * filter_bank_size, sizeof(double));
    if (s2dct == NULL) {
        return NULL;
    }
    for (i = 0; i < mfcc_size; ++i) {
        const double frequency = PI * i / filter_bank_size;
        for (j = 0; j < filter_bank_size; ++j) {
            if (j == 0) {
                s2dct[i * filter_bank_size + j] = cos(frequency * (0.5 + j)) * 0.5;
            } else {
                s2dct[i * filter_bank_size + j] = cos(frequency * (0.5 + j));
            }
        }
    }
    return s2dct;
}

// compute MFCC from either data loaded in RAM or file on disk
int _compute_mfcc(
        double *data_ptr,
        FILE *audio_file_ptr,
        struct WAVE_INFO header,
        const uint32_t data_length,
        const uint32_t sample_rate,
        const uint32_t filter_bank_size,
        const uint32_t mfcc_size,
        const uint32_t fft_order,
        const double lower_frequency,
        const double upper_frequency,
        const double emphasis_factor,
        const double window_length,
        const double window_shift,
        double **mfcc_ptr,
        uint32_t *mfcc_length
    ) {

    double *filters, *s2dct, *sin_table_full, *sin_table_half, *hamming_coefficients;
    double *frame, *power, *logsp;
    double prior, acc;
    uint32_t filters_n, frame_length, frame_shift, frame_length_padded;
    uint32_t number_of_frames, frame_index, frame_start, frame_end;
    uint32_t i, j;

#if USE_FFTW
    rfftw_plan plan;
#endif

    if (upper_frequency > (sample_rate / 2.0)) {
        // upper frequency exceeds Nyquist
        return CMFCC_FAILURE;
    }

#if USE_FFTW
    // create fftw plan for rfft
    plan = rfftw_create_plan(fft_order, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);
#endif

    // create Mel filter bank (2D matrix, filters_n x filter_bank_size)
    filters_n = (fft_order / 2) + 1;
    filters = _create_mel_filter_bank(
            fft_order,
            filter_bank_size,
            sample_rate,
            upper_frequency,
            lower_frequency);
    if (filters == NULL) {
        return CMFCC_FAILURE;
    }

    // compute DCT matrix
    s2dct = _create_dct_matrix(mfcc_size, filter_bank_size);
    if (s2dct == NULL) {
        return CMFCC_FAILURE;
    }

    // length of a frame, in samples
    frame_length = (uint32_t)floor(window_length * sample_rate);
    frame_length_padded = _max(frame_length, fft_order);

    // shift of a frame, in samples
    frame_shift = (uint32_t)floor(window_shift * sample_rate);

    // value of the last sample in the previous frame
    prior = 0.0;

    // number of frames
    number_of_frames = (uint32_t)floor(1.0 * data_length / frame_shift);
    *mfcc_length = number_of_frames;

    // allocate the mfcc matrix
    *mfcc_ptr = (double *)calloc(number_of_frames * mfcc_size, sizeof(double));
    if ((*mfcc_ptr) == NULL) {
        return CMFCC_FAILURE;
    }

    // precompute sin tables and hamming coefficients
    sin_table_full = _precompute_sin_table(fft_order);
    sin_table_half = _precompute_sin_table(fft_order / 2);
    hamming_coefficients = _precompute_hamming(frame_length);
    if ((sin_table_full == NULL) || (sin_table_half == NULL) || (hamming_coefficients == NULL)) {
        return CMFCC_FAILURE;
    }
    //printf("Frame length:        %d\n", frame_length);
    //printf("Frame shift:         %d\n", frame_shift);
    //printf("Frame length padded: %d\n", frame_length_padded);

    // allocate working buffers
    //
    // NOTE: allocating frame of length frame_length_padded so that
    //       if frame_length < fft_order,
    //       frame will be padded to length fft_order with zeroes
    //       otherwise rfft will access random memory!!!
    //
    frame = (double *)calloc(frame_length_padded, sizeof(double));
    power = (double *)calloc(filters_n, sizeof(double));
    logsp = (double *)calloc(filter_bank_size, sizeof(double));
    if ((frame == NULL) || (power == NULL) || (logsp == NULL)) {
        return CMFCC_FAILURE;
    }

    // process frames
    for (frame_index = 0; frame_index < number_of_frames; ++frame_index) {

        // clear working buffers
        memset(frame, 0, frame_length_padded * sizeof(double));
        memset(power, 0, filters_n * sizeof(double));
        memset(logsp, 0, filter_bank_size * sizeof(double));

        // copy frame values
        //
        // NOTE: using memset => frame is zero-padded, if either
        //       (frame_end - frame_start) < frame_length or
        //       frame_length < fft_order
        //
        frame_start = frame_index * frame_shift;
        frame_end = _min(frame_start + frame_length, data_length);
        if (data_ptr == NULL) {
            if (wave_read_double(audio_file_ptr, &header, frame, frame_start, (frame_end - frame_start)) != CWAVE_SUCCESS) {
                return CMFCC_FAILURE;
            }
        } else {
            memcpy(frame, data_ptr + frame_start, (frame_end - frame_start) * sizeof(double));
        }

        //printf("Frame %d : %d -> %d\n", frame_index, frame_start, frame_end);

        // emphasis + hamming + compute power
        if (_apply_emphasis(frame, frame_length, emphasis_factor, &prior) != CMFCC_SUCCESS) {
            return CMFCC_FAILURE;
        }
        if (_apply_hamming(frame, frame_length, hamming_coefficients) != CMFCC_SUCCESS) {
            return CMFCC_FAILURE;
        }

#ifdef USE_FFTW
        // fftw code
        if (_compute_power_fftw(frame, power, fft_order, plan) != CMFCC_SUCCESS) {
            return CMFCC_FAILURE;
        }
#else
        // own code
        if (_compute_power(frame, power, fft_order, sin_table_full, sin_table_half) != CMFCC_SUCCESS) {
            return CMFCC_FAILURE;
        }
#endif

        // apply Mel filter bank
        for (j = 0; j < filter_bank_size; ++j) {
            acc = 0.0;
            for (i = 0; i < filters_n; ++i) {
                acc += power[i] * filters[i * filter_bank_size + j];
            }
            if (acc < CUTOFF) {
                acc = CUTOFF;
            }
            logsp[j] = log(acc);
        }

        // multiply by DCT matrix
        for (i = 0; i < mfcc_size; ++i) {
            acc = 0.0;
            for (j = 0; j < filter_bank_size; ++j) {
                acc += logsp[j] * s2dct[i * filter_bank_size + j];
            }
            (*mfcc_ptr)[frame_index * mfcc_size + i] = acc / filter_bank_size;
        }
    }

    // free objects
#ifdef USE_FFTW
    rfftw_destroy_plan(plan);
#endif
    free((void *)logsp);
    free((void *)power);
    free((void *)frame);
    free((void *)hamming_coefficients);
    free((void *)sin_table_half);
    free((void *)sin_table_full);
    free((void *)s2dct);
    free((void *)filters);
    logsp = NULL;
    power = NULL;
    frame = NULL;
    hamming_coefficients = NULL;
    sin_table_half = NULL;
    sin_table_full = NULL;
    s2dct = NULL;
    filters = NULL;
    return CMFCC_SUCCESS;
}

// compute MFCC from data loaded in RAM
int compute_mfcc_from_data(
        double *data_ptr,
        const uint32_t data_length,
        const uint32_t sample_rate,
        const uint32_t filter_bank_size,
        const uint32_t mfcc_size,
        const uint32_t fft_order,
        const double lower_frequency,
        const double upper_frequency,
        const double emphasis_factor,
        const double window_length,
        const double window_shift,
        double **mfcc_ptr,
        uint32_t *mfcc_length
    ) {

    // to keep the compile happy, it will never be used
    struct WAVE_INFO header;

    return _compute_mfcc(
        data_ptr,
        NULL,
        header,
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
        mfcc_ptr,
        mfcc_length
    );
}

// compute MFCC from file on disk
int compute_mfcc_from_file(
    char *audio_file_path,
    const uint32_t filter_bank_size,
    const uint32_t mfcc_size,
    const uint32_t fft_order,
    const double lower_frequency,
    const double upper_frequency,
    const double emphasis_factor,
    const double window_length,
    const double window_shift,
    uint32_t *data_length,
    uint32_t *sample_rate,
    double **mfcc_ptr,
    uint32_t *mfcc_length
) {

    FILE *audio_file_ptr;
    struct WAVE_INFO header;
    uint32_t sample_rate_loc;
    uint32_t data_length_loc;
    int ret;

    // open file
    audio_file_ptr = wave_open(audio_file_path, &header);
    if (audio_file_ptr == NULL) {
        //printf("Error: cannot open file\n");
        return CMFCC_FAILURE;
    }
    data_length_loc = header.coNumSamples;
    sample_rate_loc = header.leSampleRate;

    // compute mfcc
    ret = _compute_mfcc(
        NULL,
        audio_file_ptr,
        header,
        data_length_loc,
        sample_rate_loc,
        filter_bank_size,
        mfcc_size,
        fft_order,
        lower_frequency,
        upper_frequency,
        emphasis_factor,
        window_length,
        window_shift,
        mfcc_ptr,
        mfcc_length
    );

    // close file
    wave_close(audio_file_ptr);
    *data_length = data_length_loc;
    *sample_rate = sample_rate_loc;

    return ret;
}



