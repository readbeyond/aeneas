#!/bin/bash

gcc cmfcc_driver.c cmfcc_func.c ../cwave/cwave_func.c ../cint/cint.c -o cmfcc_driver_wo_fo -Wall -pedantic -std=c99 -lm
gcc cmfcc_driver.c cmfcc_func.c ../cwave/cwave_func.c ../cint/cint.c -o cmfcc_driver_wo_ff -Wall -pedantic -std=c99 -lm           -lrfftw -lfftw               -DUSE_FFTW
gcc cmfcc_driver.c cmfcc_func.c ../cwave/cwave_func.c ../cint/cint.c -o cmfcc_driver_ws_fo -Wall -pedantic -std=c99 -lm -lsndfile                -DUSE_SNDFILE
gcc cmfcc_driver.c cmfcc_func.c ../cwave/cwave_func.c ../cint/cint.c -o cmfcc_driver_ws_ff -Wall -pedantic -std=c99 -lm -lsndfile -lrfftw -lfftw -DUSE_SNDFILE -DUSE_FFTW


