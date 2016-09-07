#!/bin/bash

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2016, Alberto Pettarin (www.albertopettarin.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

gcc cmfcc_driver.c cmfcc_func.c ../cwave/cwave_func.c ../cint/cint.c -o cmfcc_driver_wo_fo -Wall -pedantic -std=c99 -lm
gcc cmfcc_driver.c cmfcc_func.c ../cwave/cwave_func.c ../cint/cint.c -o cmfcc_driver_wo_ff -Wall -pedantic -std=c99 -lm           -lrfftw -lfftw               -DUSE_FFTW
gcc cmfcc_driver.c cmfcc_func.c ../cwave/cwave_func.c ../cint/cint.c -o cmfcc_driver_ws_fo -Wall -pedantic -std=c99 -lm -lsndfile                -DUSE_SNDFILE
gcc cmfcc_driver.c cmfcc_func.c ../cwave/cwave_func.c ../cint/cint.c -o cmfcc_driver_ws_ff -Wall -pedantic -std=c99 -lm -lsndfile -lrfftw -lfftw -DUSE_SNDFILE -DUSE_FFTW

