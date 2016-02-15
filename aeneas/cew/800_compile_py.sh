#!/bin/bash

rm -rf build *.so
python cew_setup.py build_ext --inplace

