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

usage() {
    echo ""
    echo "Usage: bash $0 [python2.7|python3.4|python3.5|pypy] [uninstall|install|deps|sdist|tests|full]"
    echo ""
}

uninstall() {
    echo "[INFO] Removing venv $1 ..."
    rm -rf "$1"
    echo "[INFO] Removing venv $1 ... done"
}

create() {
    if [ -e "$1" ]
    then
        echo "[INFO] venv $1 already exists."
        echo "[INFO] Use 'uninstall' to remove it before reinstalling it."
    else
        echo "[INFO] Creating venv $1 ..."
        # create virtualenv
        virtualenv -p "$2" "$1"
        # create output directory to run examples
        mkdir $1/output
        echo "[INFO] Creating venv $1 ... done"
    fi
}

deps() {
    if [ ! -e "$1" ]
    then
        echo "[INFO] venv $1 does not exist."
        echo "[INFO] Use 'install' or 'full' to create it."
    else
        echo "[INFO] Installing Python dependencies in $1 ..."
        cd $1
        source bin/activate
        pip install -U pip
        pip install -U numpy
        pip install -U lxml BeautifulSoup4
        pip install -U pafy requests tgt youtube-dl
        # NOTE Pillow might cause errors with pypy: try installing it as last
        pip install -U Pillow 
        deactivate
        cd ..
        echo "[INFO] Installing Python dependencies in $1 ... done"
    fi
}

sdist() {
    #
    # # check that we are running in a venv
    # python -c 'import sys; sys.exit(0) if hasattr(sys, "real_prefix") else sys.exit(1)'
    # if [ "$?" -eq 1 ]
    # then
    #     # not in venv
    # else
    #     # in venv
    # fi
    # echo ""
    #
    if [ ! -e "$1" ]
    then
        echo "[INFO] venv $1 does not exist."
        echo "[INFO] Use 'install' or 'full' to create it."
    else
       
        cd $1
        source bin/activate
        
        echo "[INFO] Uninstalling aeneas..."
        rm -f *.tar.gz
        rm -rf output
        mkdir output
        pip uninstall aeneas -y
        echo "[INFO] Uninstalling aeneas... done"
        
        deactivate
        cd ..
       
        if [ "$2" == "--remove" ]
        then
            echo "[INFO] Uninstall only: returning."
            return
        fi

        if [ "$2" != "" ]
        then
            if [ -e $2 ]
            then
                echo "[INFO] Copying file $2"
                cp $2 $1
            else
                echo "[ERRO] File $2 not found, aborting!"
                return
            fi
        else
            if [ -e ../dist/*.tar.gz ]
            then
                echo "[INFO] Copying file from ../dist/"
                cp ../dist/*.tar.gz $1
            else
                echo "[ERRO] No .tar.gz found in the dist/ directory, aborting!"
                return
            fi
        fi

        cd $1
        source bin/activate

        echo "[INFO] Installing aeneas..."
        pip install numpy
        pip install *.tar.gz
        rm -f *.tar.gz
        echo "[INFO] Installing aeneas... done"

        echo "[INFO] Diagnostics..."
        python -m aeneas.diagnostics
        echo "[INFO] Diagnostics... done"

        echo "[INFO] Testing CFW..."
        python -c 'import aeneas.globalfunctions as gf; print(gf.can_run_c_extension("cfw"))'
        echo "[INFO] Testing CFW... done"

        deactivate
        cd ..
    fi
}

copytests() {
    if [ ! -e "$1" ]
    then
        echo "[INFO] venv $1 does not exist."
        echo "[INFO] Use 'install' or 'full' to create it."
    else
        echo "[INFO] Copying tests in $1 ..."
        cd $1
        source bin/activate
       
        # delete old stuff, if any 
        rm -rf tests
        mkdir tests
        mkdir tests/output
        
        # copy current code locally and set it up
        cp -r ../../aeneas ../../setup.py ../../setupmeta.py ../../run_all_unit_tests.py tests/
        cd tests
        pip install numpy
        OS=`uname`
        if [ "$OS" == "Darwin" ]
        then
            # Mac OS X
            # add /usr/local/lib to path, otherwise cew cannot be built
            python setup.py build_ext --inplace --rpath=/usr/local/lib
        else
            # Linux
            # espeak lib should be globally visible
            python setup.py build_ext --inplace
        fi
        cd ..

        deactivate
        cd ..
        echo "[INFO] Copying tests in $1 ... done"
    fi
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

# venv directory name
D="venv_$EX"

# check the full path of the executable
FULLEX=`which $EX`
if [ "$FULLEX" == "" ]
then
    echo "[ERRO] Unable to find the full path of executable $EX, aborting."
    exit 1
fi

# check the action argument
if [ "$ACTION" != "uninstall" ] && [ "$ACTION" != "install" ] && [ "$ACTION" != "deps" ] && [ "$ACTION" != "sdist" ] && [ "$ACTION" != "tests" ] && [ "$ACTION" != "full" ]
then
    usage
    exit 1
fi

if [ "$ACTION" == "uninstall" ]
then
    uninstall $D
fi

if [ "$ACTION" == "install" ]
then
    create $D $FULLEX
fi

if [ "$ACTION" == "deps" ]
then
    deps $D
fi

if [ "$ACTION" == "sdist" ]
then
    ARCHIVE=""
    if [ "$#" -ge 3 ]
    then
        ARCHIVE=$3
    fi
    sdist $D $ARCHIVE $STOP
fi

if [ "$ACTION" == "tests" ]
then
    copytests $D
fi

if [ "$ACTION" == "full" ]
then
    create $D $FULLEX
    deps $D
    copytests $D
fi
