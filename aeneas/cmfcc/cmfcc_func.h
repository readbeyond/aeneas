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

#include "cint.h"

#define CMFCC_SUCCESS 0
#define CMFCC_FAILURE 1

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
);

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
);

