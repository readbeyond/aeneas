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
    echo "Usage: bash $0 [python2.7|python3.4|python3.5|pypy] [uninstall|install|deps|tests|full]"
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
        # create symlink for testing sdist tar.gz
        cd $1
        ln -s "../install_aeneas_from_sdist_tar_gz.sh" "install_aeneas_from_sdist_tar_gz.sh"
        cd ..
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
        
        # copy current code locally and set it up
        cp -r ../../aeneas ../../*.py tests/
        cd tests
        pip install numpy
        python setup.py build_ext --inplace
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
if [ "$ACTION" != "uninstall" ] && [ "$ACTION" != "install" ] && [ "$ACTION" != "deps" ] && [ "$ACTION" != "tests" ] && [ "$ACTION" != "full" ]
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
