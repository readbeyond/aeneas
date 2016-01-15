: Compile Python C extensions for aeneas on Windows
: -------------------------------------------------

: BAT Script Author: Richard Margetts

: Author: Alberto Pettarin
: Copyright:
:     Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
:     Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
:     Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
:
: Licence: GNU AGPL 3
: Version: 1.4.0
: Email:   aeneas@readbeyond.it
: Status:  Production

: Before running this batch file in Windows, do the following:
: 1. Download and install the Microsoft Visual C++ compiler for Python
:       http://www.microsoft.com/en-us/download/details.aspx?id=44266
: 2. Open the Visual C++ 2008 Command prompt (via Windows Search)
: 3. Change directory to your aeneas folder
: 4. Now run this batch file

@echo off
echo "[WARN] This script is deprecated. Instead, please use"
echo "[WARN] python setup.py build_ext --inplace"
echo "[WARN] See the README for details"
echo ""
echo "[INFO] Setting environment variables..."
SET DISTUTILS_USE_SDK=1
SET MSSdk=1
echo "[INFO] Setting environment variables... done"

cd aeneas

echo "[INFO] Compiling cdtw..."
python cdtw_setup.py build_ext --inplace
echo "[INFO] Compiling cdtw... done"

echo "[INFO] Compiling cmfcc..."
python cmfcc_setup.py build_ext --inplace
echo "[INFO] Compiling cmfcc... done"

cd ..

echo "[INFO] Now please run"
echo "[INFO]   python aeneas_check_setup.py"
echo "[INFO] to check that the C extensions are compiled."
echo ""
echo "[INFO] If the compilation failed, please check that:"
echo "[INFO]   1. you have the Microsoft Visual C++ compiler for Python installed, and"
echo "[INFO]   2. you ran this script in the Visual C++ Command prompt instead of the regular Windows prompt."
echo "[INFO] See the README for details."

