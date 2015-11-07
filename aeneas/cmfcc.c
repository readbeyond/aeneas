/*

Python C Extension for computing the MFCC 

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
#include <numpy/arrayobject.h>
#include <math.h>
#include <numpy/npy_math.h>

#define MEL_10 2595.0 
#define PI     3.1415926535897932384626433832795
#define PI_2   6.2831853071795864769252867665590
#define CUTOFF 0.00001

// return the min of the given arguments
static int _min(int a, int b) {
    if (a < b) {
        return a;
    }
    return b;
}

// round to the nearest integer
static int _round(double x) {
    if (x < 0) {
        return (int)ceil(x - 0.5); 
    }
    return (int)floor(x + 0.5);
}

// precompute the sin table for the FFT/RFFT
static double *precompute_sin_table(int m) {
    const double arg = PI / m * 2;
    const int size = m - m / 4 + 1;
    double *table;
    int k;
    
    table = (double *)calloc(size, sizeof(double));
    table[0] = 0;
    for (k = 1; k < size; ++k) {
        table[k] = sin(arg * k);
    }
    table[m / 2] = 0;
    return table;
}
static int fft(double *x, double *y, const int m, double *sin_table) {
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

    return 0;
}
static int rfft(
        double *x,
        double *y,
        const int m,
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
        return -1;
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

    return 0;
}

// convert Hz frequency to Mel frequency
static double hz2mel(const double f) {
    return MEL_10 * log10(1.0 + (f / 700.0));
}

// convert Mel frequency to Hz frequency
static double mel2hz(const double m) {
    return 700.0 * (pow(10.0, m / MEL_10) - 1.0);
}

// pre emphasis of the given frame
// returns the prior to be used for the next frame
static void apply_emphasis(
        double *frame,
        const int length,
        const double emphasis_factor,
        double *prior
    ) {
    double prior_orig;
    double *frame_orig;
    int i;
    
    prior_orig = frame[length - 1];
    frame_orig = (double *)calloc(length, sizeof(double));
    memcpy(frame_orig, frame, length * sizeof(double));
    frame[0] = frame_orig[0] - emphasis_factor * (*prior);
    for (i = 1; i < length; ++i) {
        frame[i] = frame_orig[i] - frame_orig[i-1] * emphasis_factor;
    }
    free((void *)frame_orig);
    *prior = prior_orig;
}

// compute the power of the given frame
static void compute_power(
        double *frame,
        double *power,
        const int length,
        double *sin_table_full,
        double *sin_table_half
    ) {
    const int extra = 1 + (length / 2);
    double *tmp;
    int k;

    tmp = (double *)calloc(length + extra, sizeof(double));
    rfft(frame, tmp, length, sin_table_full, sin_table_half);
    power[0] = frame[0] * frame[0];
    for (k = 1; k < extra; ++k) {
        power[k] = frame[k] * frame[k] + tmp[k] * tmp[k];
    }
    free((void *)tmp);
}

// transform the frame using the Hamming window
static void apply_hamming(
        double *frame,
        const int length,
        double *coefficients
    ) {
    int k;
    
    for (k = 0; k < length; ++k) {
        frame[k] *= coefficients[k];
    }
}
static double *precompute_hamming(const int length) {
    const double arg = PI_2 / (length - 1);
    double *coefficients;
    int k;
    
    coefficients = (double *)calloc(length, sizeof(double));
    for (k = 0; k < length; ++k) {
        coefficients[k] = (0.54 - 0.46 * cos(k * arg));
    }
    return coefficients;
}

// create Mel filter bank
// return a pointer to a 2D matrix (filters_n x filter_bank_size)
static double *create_mel_filter_bank(
        int fft_order,
        int filter_bank_size,
        int sample_rate,
        double upper_frequency,
        double lower_frequency
    ) {
    const double step_frequency = 1.0 * sample_rate / fft_order; 
    const double melmax = hz2mel(upper_frequency);
    const double melmin = hz2mel(lower_frequency);
    const double melstep = (melmax - melmin) / (filter_bank_size + 1);
    const int filter_edge_length = filter_bank_size + 2; 
    const int filters_n = fft_order / 2 + 1;
    double *filter_edges, *filters;
    int k;
   
    // filter bank  
    filters = (double *)calloc(filters_n * filter_bank_size, sizeof(double));
    
    // filter edges
    filter_edges = (double *)calloc(filter_edge_length, sizeof(double));
    
    // TODO porting Python code "verbatim",
    //      but some code cleanup should be done here
    for (k = 0; k < filter_edge_length; ++k) {
        filter_edges[k] = mel2hz(melmin + melstep * k);
    }
    for (k = 0; k < filter_bank_size; ++k) {
        const int left_frequency = _round(filter_edges[k] / step_frequency);
        const int center_frequency = _round(filter_edges[k + 1] / step_frequency);
        const int right_frequency = _round(filter_edges[k + 2] / step_frequency);
        const double width_frequency = (right_frequency - left_frequency) * step_frequency;
        const double height_frequency = 2.0 / width_frequency;
        double left_slope, right_slope;
        int current_frequency;
        
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
    return filters;
}

// create the DCT matrix
// return a pointer to a 2D matrix (mfcc_size x filter_bank_size)
static double *create_dct_matrix(int mfcc_size, int filter_bank_size) {
    double *s2dct;
    int i, j;
    
    s2dct = (double *)calloc(mfcc_size * filter_bank_size, sizeof(double));
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

// compute the MFCCs of the given signal 
// take the PyObject containing the following arguments (see below)
// and return the MFCCs as a n x mfcc_size 2D array of double, where
//   - n is the number of frames
//   - mfcc_size is the number of ceptral coefficients (including the 0-th)
static PyObject *cmfcc_compute_mfcc(PyObject *self, PyObject *args) {
    PyObject *signal_raw;   // 1D array of double, holding the signal
    int sample_rate;        // sample rate (default: 16000)
    int frame_rate;         // frame rate (default: 25)
    int filter_bank_size;   // number of filters in the filter bank (default: 40)
    int mfcc_size;          // number of ceptral coefficients (default: 13)
    int fft_order;          // FFT order; must be a power of 2 (default: 512)
    double lower_frequency; // lower frequency (default: 133.3333)
    double upper_frequency; // upper frequency; must be <= sample_rate/2 = Nyquist frequency (default: 6855.4976)
    double emphasis_factor; // pre-emphasis factor (default: 0.97)
    double window_length;   // window length (default: 0.0256)

    PyArrayObject *signal, *mfcc;
    npy_intp mfcc_dimensions[2];
    double samples_per_frame, prior, acc;
    double *signal_ptr, *mfcc_ptr;
    double *filters, *s2dct, *sin_table_full, *sin_table_half, *hamming_coefficients;
    double *frame, *power, *logsp;
    int signal_length, filters_n, frame_length, number_of_frames;
    int i, j, frame_index, frame_start, frame_end;

    // TODO use PyArg_ParseTupleAndKeywords instead, to have default values set automatically
    // O = object (do not convert or check for errors)
    // i = integer
    // d = double
    if (!PyArg_ParseTuple(
                args,
                "Oiiiiidddd",
                &signal_raw,
                &sample_rate,
                &frame_rate,
                &filter_bank_size,
                &mfcc_size,
                &fft_order,
                &lower_frequency,
                &upper_frequency,
                &emphasis_factor,
                &window_length)) {
        PyErr_SetString(PyExc_ValueError, "Error while parsing the arguments");
        return NULL;
    }

    if (upper_frequency > sample_rate / 2.0) {
        // raise exception
        PyErr_SetString(PyExc_ValueError, "Upper frequency exceeds Nyquist");
        return NULL;
    }

    // convert to C contiguous array
    signal = (PyArrayObject *) PyArray_ContiguousFromObject(signal_raw, PyArray_DOUBLE, 1, 1);

    // pointer to signal data
    signal_ptr = (double *)signal->data;
    
    // get the number of samples of signal
    // NOTE: this is not the duration (in seconds), which is (n / sample_rate) !
    signal_length = signal->dimensions[0];

    // create Mel filter bank (2D matrix, filters_n x filter_bank_size)
    filters_n = ((fft_order / 2) + 1);
    filters = create_mel_filter_bank(
            fft_order,
            filter_bank_size,
            sample_rate,
            upper_frequency,
            lower_frequency);

    // compute DCT matrix
    s2dct = create_dct_matrix(mfcc_size, filter_bank_size);

    // samples per frame
    samples_per_frame = 1.0 * sample_rate / frame_rate;

    // length of a frame
    frame_length = (int)floor(window_length * sample_rate);

    // value of the last sample in the previous frame
    prior = 0.0;

    // number of frames
    number_of_frames = floor((signal_length / samples_per_frame) + 1);

    // allocate the mfcc matrix 
    mfcc_ptr = (double *)calloc(number_of_frames * mfcc_size, sizeof(double));

    // precompute sin tables
    sin_table_full = precompute_sin_table(fft_order);
    sin_table_half = precompute_sin_table(fft_order / 2);

    // precompute hamming coefficients
    hamming_coefficients = precompute_hamming(frame_length);

    // TODO porting Python code "verbatim",
    //      but some code cleanup should be done here
    // process frames
    for (frame_index = 0; frame_index < number_of_frames; ++frame_index) {
        
        // allocate working buffers
        frame = (double *)calloc(frame_length, sizeof(double));
        power = (double *)calloc(filters_n, sizeof(double));
        logsp = (double *)calloc(filter_bank_size, sizeof(double));

        // TODO porting Python code "verbatim",
        //      but some code cleanup should be done here
        // copy frame values
        frame_start = _round(frame_index * samples_per_frame);
        frame_end = _min(frame_start + frame_length, signal_length);
        // NOTE: using calloc => last frame is zero-padded, if (frame_end - frame_start) < frame_length
        memcpy(frame, signal_ptr + frame_start, (frame_end - frame_start) * sizeof(double));
      
        // emphasis + hamming + compute power
        apply_emphasis(frame, frame_length, emphasis_factor, &prior);
        apply_hamming(frame, frame_length, hamming_coefficients);
        compute_power(frame, power, fft_order, sin_table_full, sin_table_half);
       
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
            mfcc_ptr[frame_index * mfcc_size + i] = acc / filter_bank_size;
        }

        // free working buffers
        free((void *)logsp);
        free((void *)power);
        free((void *)frame);
    }

    // decrement reference to local object no longer needed
    free((void *)hamming_coefficients);
    free((void *)sin_table_half);
    free((void *)sin_table_full);
    free((void *)s2dct);
    free((void *)filters);
    Py_DECREF(signal);
    
    // create mfcc object
    mfcc_dimensions[0] = number_of_frames;
    mfcc_dimensions[1] = mfcc_size;
    mfcc = (PyArrayObject *) PyArray_SimpleNewFromData(2, mfcc_dimensions, PyArray_DOUBLE, mfcc_ptr);
    
    // return computed mfcc 
    return PyArray_Return(mfcc);
}

static PyMethodDef cmfcc_methods[] = {
    {
        "cmfcc_compute_mfcc",
        cmfcc_compute_mfcc,
        METH_VARARGS,
        "Given the wave data, compute and return the MFCCs"
    },
    {
        NULL,
        NULL,
        0,
        NULL
    }
};

PyMODINIT_FUNC initcmfcc(void)  {
    (void) Py_InitModule("cmfcc", cmfcc_methods);
    import_array();
}



