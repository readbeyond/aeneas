#!/bin/bash

rm -rf build *.so
python cwave_setup.py build_ext --inplace

