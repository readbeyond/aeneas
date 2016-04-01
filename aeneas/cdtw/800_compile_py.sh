#!/bin/bash

rm -rf build *.so
python cdtw_setup.py build_ext --inplace

