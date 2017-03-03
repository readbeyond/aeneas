#!/bin/bash

# __author__ = "Alberto Pettarin"
# __copyright__ = """
#     Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
#     Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
#     Copyright 2015-2017, Alberto Pettarin (www.albertopettarin.it)
#     """
# __license__ = "GNU AGPL 3"
# __version__ = "1.7.2"
# __email__ = "aeneas@readbeyond.it"
# __status__ = "Production"

usage() {
    echo ""
    echo "Usage: bash $0 [py2.7|py3.5|py3.6|pypy] [all|nonet|noben|fast|bench|tool|long|net|clean]"
    echo ""
}

clean() {
    echo "[INFO] Cleaning $1 ..."
    cd $1
    rm -rf tests
    cd ..
    echo "[INFO] Cleaning $1 ... done"
}

copy_tests() {
    echo "[INFO] Copying tests for $1 ..."
    bash manage_venvs.sh $1 tests
    echo "[INFO] Copying tests for $1 ... done"
}

venv_activate() {
    echo "[INFO] Activating $1 ..."
    cd $1
    source bin/activate
    cd tests
    echo "[INFO] Activating $1 ... done"
}

venv_deactivate() {
    echo "[INFO] Dectivating $1 ..."
    cd ..
    deactivate
    cd ..
    echo "[INFO] Dectivating $1 ... done"
}

run_fast() {
    echo "[INFO] Running fast tests..."
    python run_all_unit_tests.py    -v >> $1 2> /dev/null
    echo "[INFO] Running fast tests... done"
}

run_bench() {
    echo "[INFO] Running bench tests..."
    python run_all_unit_tests.py -b -v >> $1 2> /dev/null
    echo "[INFO] Running bench tests... done"
}

run_tool() {
    echo "[INFO] Running tool tests..."
    python run_all_unit_tests.py -t -v >> $1 2> /dev/null
    echo "[INFO] Running tool tests... done"
}

run_long() {
    echo "[INFO] Running long tests..."
    python run_all_unit_tests.py -l -v >> $1 2> /dev/null
    echo "[INFO] Running long tests... done"
}

run_net() {
    echo "[INFO] Running net tests..."
    python run_all_unit_tests.py -n -v >> $1 2> /dev/null
    echo "[INFO] Running net tests... done"
}

grep_log() {
    echo "[INFO] Created log file $1"
    echo ""
    grep "INFO" $1
    echo ""
}


# check that we have at least two arguments,
# otherwise show usage and exit
if [ "$#" -lt 2 ]
then
    usage
    exit 1
fi

# get arguments
EX=$1
ACTION=$2

# replace e.g. "venv_python2.7/", "venv_python2.7", "py2.7", "2.7" with "python2.7"
for V in "2.7" "3.5" "py"
do
    if [ "$EX" == "venv_python$V/" ] || [ "$EX" == "venv_python$V" ] || [ "$EX" == "py$V" ] || [ "$EX" == "$V" ]
    then
        if [ "$V" == "py" ]
        then
            EX="pypy"
        else
            EX="python$V"
        fi
    fi
done

# venv directory name
D="venv_$EX"
L="/tmp/log_aeneas_$EX"

# check the full path of the executable
FULLEX=`which $EX`
if [ "$FULLEX" == "" ]
then
    echo "[ERRO] Unable to find the full path of executable $EX, aborting."
    exit 1
fi

# check the action argument
if [ "$ACTION" != "all" ] && [ "$ACTION" != "nonet" ] && [ "$ACTION" != "noben" ] && [ "$ACTION" != "fast" ] && [ "$ACTION" != "bench" ] && [ "$ACTION" != "tool" ] && [ "$ACTION" != "long" ] && [ "$ACTION" != "net" ] && [ "$ACTION" != "clean" ]
then
    usage
    exit 1
fi

# check that the venv exists
if [ ! -e "$D" ]
then
    echo "[ERRO] Install venv with 'bash manage_venvs.sh $EX full', aborting."
    exit 1
fi

# remove log file, if already existing
rm -f $L

if [ "$ACTION" == "clean" ]
then
    clean $D
fi

if [ "$ACTION" == "all" ]
then
    copy_tests $EX
    venv_activate $D
    run_fast $L
    run_tool $L
    run_long $L
    run_bench $L
    run_net $L
    venv_deactivate $D
    grep_log $L
fi

if [ "$ACTION" == "nonet" ]
then
    copy_tests $EX
    venv_activate $D
    run_fast $L
    run_tool $L
    run_long $L
    run_bench $L
    venv_deactivate $D
    grep_log $L
fi

if [ "$ACTION" == "noben" ]
then
    copy_tests $EX
    venv_activate $D
    run_fast $L
    run_tool $L
    run_long $L
    venv_deactivate $D
    grep_log $L
fi

if [ "$ACTION" == "fast" ]
then
    copy_tests $EX
    venv_activate $D
    run_fast $L
    venv_deactivate $D
    grep_log $L
fi

if [ "$ACTION" == "bench" ]
then
    copy_tests $EX
    venv_activate $D
    run_bench $L
    venv_deactivate $D
    grep_log $L
fi

if [ "$ACTION" == "tool" ]
then
    copy_tests $EX
    venv_activate $D
    run_tool $L
    venv_deactivate $D
    grep_log $L
fi

if [ "$ACTION" == "long" ]
then
    copy_tests $EX
    venv_activate $D
    run_long $L
    venv_deactivate $D
    grep_log $L
fi

if [ "$ACTION" == "net" ]
then
    copy_tests $EX
    venv_activate $D
    run_net $L
    venv_deactivate $D
    grep_log $L
fi
