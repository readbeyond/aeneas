#!/bin/bash

# __author__ = "Alberto Pettarin"
# __copyright__ = """
#     Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
#     Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
#     Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
#     """
# __license__ = "GNU AGPL 3"
# __version__ = "1.1.1"
# __email__ = "aeneas@readbeyond.it"
# __status__ = "Production"

cd aeneas

echo "[INFO] Compiling cdtw..."
python cdtw_setup.py build_ext --inplace
echo "[INFO] Compiling cdtw... done"

echo "[INFO] Compiling cmfcc..."
python cmfcc_setup.py build_ext --inplace
echo "[INFO] Compiling cmfcc... done"

cd ..

