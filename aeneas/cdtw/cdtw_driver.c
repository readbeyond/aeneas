/*

Python C Extension for computing the DTW

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
#include "cdtw_func.h"

#define DRIVER_SUCCESS 0
#define DRIVER_FAILURE 1

// print usage
void _usage(const char *prog) {
    printf("\n");
    printf("Usage:   %s MFCC_SIZE DELTA MFCC1_FILE MFCC1_LEN MFCC2_FILE MFCC2_LEN [cm|acm|path]\n", prog);
    printf("\n");
    printf("Example: %s 12 3000 ../tests/res/cdtw/mfcc1_12_1332 1332 ../tests/res/cdtw/mfcc2_12_868 868 cm\n", prog);
    printf("         %s 12 3000 ../tests/res/cdtw/mfcc1_12_1332 1332 ../tests/res/cdtw/mfcc2_12_868 868 acm\n", prog);
    printf("         %s 12 3000 ../tests/res/cdtw/mfcc1_12_1332 1332 ../tests/res/cdtw/mfcc2_12_868 868 path\n", prog);
    printf("\n");
}

// read matrix from file
int _read_matrix(const char *file_name, double *matrix, uint32_t n, uint32_t m) {
    uint32_t i, j;
    FILE *file_ptr;

    file_ptr = fopen(file_name, "r");
    if (file_ptr == NULL) {
        return DRIVER_FAILURE;
    }
    
    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            if (!fscanf(file_ptr, "%lf", matrix + i * m + j)) {
                break;
            }
            //printf("Reading %u %u\n", i, j);
        }
    }
    fclose(file_ptr);
    file_ptr = NULL;
    return DRIVER_SUCCESS;
}

// print matrix to stdout
void _print_matrix(double *matrix, uint32_t n, uint32_t m) {
    uint32_t i, j;

    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            printf("%.6f ", matrix[i * m + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {

    double *mfcc1_ptr, *mfcc2_ptr, *cost_matrix_ptr;
    char *mfcc1_file_name, *mfcc2_file_name, *mode;
    uint32_t *centers_ptr;
    uint32_t mfcc_size, delta, mfcc1_len, mfcc2_len;
    uint32_t best_path_length, k;
    struct PATH_CELL *best_path;

    if (argc < 8) {
        _usage(argv[0]);
        return DRIVER_FAILURE;
    }
    mfcc_size = atoi(argv[1]);
    delta = atoi(argv[2]);
    mfcc1_file_name = argv[3];
    mfcc1_len = atoi(argv[4]);
    mfcc2_file_name = argv[5];
    mfcc2_len = atoi(argv[6]);
    mode = argv[7];

    if (delta > mfcc2_len) {
        delta = mfcc2_len;
    }

    // allocate space for the MFCCs and read the input files
    mfcc1_ptr = (double *)calloc(mfcc1_len * mfcc_size, sizeof(double));
    mfcc2_ptr = (double *)calloc(mfcc2_len * mfcc_size, sizeof(double));
    if ((mfcc1_ptr == NULL) || (mfcc2_ptr == NULL)) {
        printf("Error: unable to allocate space for the input MFCCs.\n");
        return DRIVER_FAILURE;
    }
    if (_read_matrix(mfcc1_file_name, mfcc1_ptr, mfcc1_len, mfcc_size) != DRIVER_SUCCESS) {
        printf("Error: unable to read MFCC1.\n");
        return DRIVER_FAILURE;
    }
    if (_read_matrix(mfcc2_file_name, mfcc2_ptr, mfcc2_len, mfcc_size) != DRIVER_SUCCESS) {
        printf("Error: unable to read MFCC2.\n");
        return DRIVER_FAILURE;
    }

    // allocate space for the cost matrix
    cost_matrix_ptr = (double *)calloc(mfcc1_len * delta, sizeof(double));
    centers_ptr = (uint32_t *)calloc(mfcc1_len, sizeof(uint32_t));
    if ((cost_matrix_ptr == NULL) || (centers_ptr == NULL)) {
        printf("Error: unable to allocate space for the cost matrix and the centers.\n");
        return DRIVER_FAILURE;
    }

    // compute cost matrix
    if (_compute_cost_matrix(
                mfcc1_ptr,
                mfcc2_ptr,
                delta,
                cost_matrix_ptr,
                centers_ptr,
                mfcc1_len,
                mfcc2_len,
                mfcc_size) != CDTW_SUCCESS) {
        printf("Error: unable to compute cost matrix.\n");
        return DRIVER_FAILURE;
    }

    if (strcmp(mode, "cm") == 0) {
        // print cost matrix
        _print_matrix(cost_matrix_ptr, mfcc1_len, delta);
    } else if ((strcmp(mode, "acm") == 0) || (strcmp(mode, "path") == 0)) {
        // compute accumulated cost matrix
        if (_compute_accumulated_cost_matrix_in_place(cost_matrix_ptr, centers_ptr, mfcc1_len, delta) != CDTW_SUCCESS) {
            printf("Error: unable to compute accumulated cost matrix.\n");
            return DRIVER_FAILURE;
        }
        if (strcmp(mode, "acm") == 0) {
            // print accumulated cost matrix
            _print_matrix(cost_matrix_ptr, mfcc1_len, delta);
        } else {
            // print best path
            if (_compute_best_path(cost_matrix_ptr, centers_ptr, mfcc1_len, delta, &best_path, &best_path_length) != CDTW_SUCCESS) {
                printf("Error: unable to compute best path.\n");
                return DRIVER_FAILURE;
            }
            for (k = 0; k < best_path_length; ++k) {
                printf("%u %u\n", best_path[k].i, best_path[k].j);
            }
            free((void *)best_path);
        }
    } 

    // deallocate
    free((void *)centers_ptr);
    free((void *)cost_matrix_ptr);
    free((void *)mfcc2_ptr);
    free((void *)mfcc1_ptr);

    return DRIVER_SUCCESS;
}



