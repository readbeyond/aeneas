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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "cdtw_func.h"

void _read_matrix(const char *file_name, double *matrix, unsigned int n, unsigned int m) {
    unsigned int i, j;
    FILE *file_ptr;

    file_ptr = fopen(file_name, "r");
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
}

void _print_matrix(double *matrix, unsigned int n, unsigned int m) {
    unsigned int i, j;

    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            printf("%.6f ", matrix[i * m + j]);
        }
        printf("\n");
    }
}

//
// this is a simple driver to test on the command line
//
// compile it with:
//
// $ gcc cdtw_driver.c cdtw_func.c -o cdtw_driver -lm
//
// use it as follows:
//
// ./cdtw_driver MFCC_SIZE DELTA MFCC1_FILE MFCC1_LEN MFCC2_FILE MFCC2_LEN cm   => compute and print cost matrix
// ./cdtw_driver MFCC_SIZE DELTA MFCC1_FILE MFCC1_LEN MFCC2_FILE MFCC2_LEN acm  => compute and print accumulated cost matrix
// ./cdtw_driver MFCC_SIZE DELTA MFCC1_FILE MFCC1_LEN MFCC2_FILE MFCC2_LEN path => compute and print best path
//
// example:
// ./cdtw_driver 12 3000 ../tests/res/cdtw/mfcc1_12_1332 1332 ../tests/res/cdtw/mfcc2_12_868 868 path
//
int main(int argc, char **argv) {

    double *mfcc1_ptr, *mfcc2_ptr, *cost_matrix_ptr;
    char *mfcc1_file_name, *mfcc2_file_name, *mode;
    unsigned int *centers_ptr;
    unsigned int mfcc_size, delta, mfcc1_len, mfcc2_len, best_path_length, k;
    struct PATH_CELL *best_path;

    if (argc < 8) {
        printf("\nUsage: %s MFCC_SIZE DELTA MFCC1_FILE MFCC1_LEN MFCC2_FILE MFCC2_LEN [cm|acm|path]\n\n", argv[0]);
        return 1;
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

    mfcc1_ptr = (double *)calloc(mfcc1_len * mfcc_size, sizeof(double));
    _read_matrix(mfcc1_file_name, mfcc1_ptr, mfcc1_len, mfcc_size);
    mfcc2_ptr = (double *)calloc(mfcc2_len * mfcc_size, sizeof(double));
    _read_matrix(mfcc2_file_name, mfcc2_ptr, mfcc2_len, mfcc_size);

    // allocate space
    cost_matrix_ptr = (double *)calloc(mfcc1_len * delta, sizeof(double));
    centers_ptr = (unsigned int *)calloc(mfcc1_len, sizeof(unsigned int));

    // compute cost matrix
    _compute_cost_matrix(mfcc1_ptr, mfcc2_ptr, delta, cost_matrix_ptr, centers_ptr, mfcc1_len, mfcc2_len, mfcc_size);
    if (strcmp(mode, "cm") == 0) {
        // print cost matrix
        _print_matrix(cost_matrix_ptr, mfcc1_len, delta);
    } else if ((strcmp(mode, "acm") == 0) || (strcmp(mode, "path") == 0)) {
        // compute accumulated cost matrix
        _compute_accumulated_cost_matrix_in_place(cost_matrix_ptr, centers_ptr, mfcc1_len, delta);
        if (strcmp(mode, "acm") == 0) {
            // print accumulated cost matrix
            _print_matrix(cost_matrix_ptr, mfcc1_len, delta);
        } else {
            // print best path
            _compute_best_path(cost_matrix_ptr, centers_ptr, mfcc1_len, delta, &best_path, &best_path_length);
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

    return 0;
}



