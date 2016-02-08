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

// NOTE: using unsigned int as it is 32-bit wide on all modern architectures
//       not using uint32_t because the MS C compiler does not have <stdint.h>
//       or, at least, it is not easy to use it

// compute MFCC from data loaded in RAM
int compute_mfcc_from_data(
    double *data_ptr,
    const unsigned int data_length,
    const unsigned int sample_rate,
    const unsigned int filter_bank_size,
    const unsigned int mfcc_size,
    const unsigned int fft_order,
    const double lower_frequency,
    const double upper_frequency,
    const double emphasis_factor,
    const double window_length,
    const double window_shift,
    double **mfcc_ptr,
    unsigned int *mfcc_length
);

// compute MFCC from file on disk
int compute_mfcc_from_file(
    char *audio_file_path,
    const unsigned int filter_bank_size,
    const unsigned int mfcc_size,
    const unsigned int fft_order,
    const double lower_frequency,
    const double upper_frequency,
    const double emphasis_factor,
    const double window_length,
    const double window_shift,
    unsigned int *data_length_ret,
    unsigned int *sample_rate_ret,
    double **mfcc_ptr,
    unsigned int *mfcc_length
);

