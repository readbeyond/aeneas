#!/bin/bash

# __author__ = "Alberto Pettarin"
# __copyright__ = """
#     Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
#     Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
#     Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
#     """
# __license__ = "GNU AGPL 3"
# __version__ = "1.3.2"
# __email__ = "aeneas@readbeyond.it"
# __status__ = "Production"

cd aeneas

echo "[WARN] This script is deprecated. Instead, please use"
echo "[WARN] python setup.py build_ext --inplace"
echo "[WARN] See the README for details"
echo ""

echo "[INFO] Compiling cdtw..."
python cdtw_setup.py build_ext --inplace
echo "[INFO] Compiling cdtw... done"

echo "[INFO] Compiling cmfcc..."
python cmfcc_setup.py build_ext --inplace
echo "[INFO] Compiling cmfcc... done"

echo "[INFO] Compiling cew..."
python cew_setup.py build_ext --inplace
echo "[INFO] Compiling cew... done"

cd ..

