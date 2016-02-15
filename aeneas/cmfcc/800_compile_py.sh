#!/bin/bash

rm -rf build *.so
python cmfcc_setup.py build_ext --inplace

