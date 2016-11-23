#!/bin/bash

# __author__ = "Alberto Pettarin"
# __copyright__ = """
#     Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
#     Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
#     Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
#     """
# __license__ = "GNU AGPL 3"
# __version__ = "1.7.0"
# __email__ = "aeneas@readbeyond.it"
# __status__ = "Production"

# check that the sdist is in place
if [ ! -e ../../dist/aeneas*.tar.gz ]
then
    echo "[ERRO] No .tar.gz found in the dist/ directory, aborting!"
    exit 1
fi

# check that we are running in a venv
python -c 'import sys; sys.exit(0) if hasattr(sys, "real_prefix") else sys.exit(1)'
if [ "$?" -eq 1 ]
then
    echo "[ERRO] This script should be run in a venv only!"
    echo "[INFO] Try:"
    echo "[INFO] $ source bin/activate"
    echo ""
    exit 1
fi
echo ""

echo "[INFO] Uninstalling..."
rm -f aeneas*.tar.gz
rm -rf output
mkdir output
pip uninstall aeneas -y
echo "[INFO] Uninstalling... done"
echo ""

if [ "$#" -ge 1 ]
then
    echo "[INFO] Quitting before installing aeneas."
    echo ""
    exit 0
fi

echo "[INFO] Installing..."
cp ../../dist/aeneas*.tar.gz .
pip install numpy
pip install aeneas*.tar.gz
rm aeneas*.tar.gz
echo "[INFO] Installing... done"
echo ""

echo "[INFO] Diagnostics..."
python -m aeneas.diagnostics
echo "[INFO] Diagnostics... done"
echo ""

echo "[INFO] Running some examples..."
echo ""
python -m aeneas.tools.execute_task --example-smil
echo ""
python -m aeneas.tools.execute_task --example-json
echo ""
python -m aeneas.tools.execute_task --example-festival
echo ""
python -m aeneas.tools.execute_task --example-ctw-espeak
echo ""
python -m aeneas.tools.execute_task --example-textgrid
echo ""
echo "[INFO] Running some examples... done"
echo ""

echo "[INFO] Testing CFW..."
python -c 'import aeneas.globalfunctions as gf; print(gf.can_run_c_extension("cfw"))'
echo "[INFO] Testing CFW... done"

