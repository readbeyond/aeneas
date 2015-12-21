#!/bin/bash

# __author__ = "Alberto Pettarin"
# __copyright__ = """
#     Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
#     Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
#     Copyright 2015-2016, Alberto Pettarin (www.albertopettarin.it)
#     """
# __license__ = "GNU AGPL 3"
# __version__ = "1.4.0"
# __email__ = "aeneas@readbeyond.it"
# __status__ = "Production"


EXECUTABLE="python"

if [ "$#" -ge "1" ]
then
    if [ "$1" == "-h" ]
    then
        echo ""
        echo "$ bash $0 [2|2.7|3|3.4|3.5]"
        echo ""
        exit 0
    else
        EXECUTABLE="python$1"
    fi
fi

DEST="/tmp/aeneas_all_$EXECUTABLE"

echo "[INFO] Running all unit tests with $EXECUTABLE..."
$EXECUTABLE -m unittest discover 2> $DEST
echo "[INFO] Running all unit tests... done"
echo ""

RESULT=`tail -n1 $DEST`

if [ "$RESULT" == "OK" ]
then
    echo "[INFO] No errors"
else
    echo "[ERRO] Errors found, see file $DEST"
    echo ""
    cat $DEST
fi



