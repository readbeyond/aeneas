#!/bin/bash

# __author__ = "Alberto Pettarin"
# __copyright__ = """
#     Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
#     Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
#     Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
#     """
# __license__ = "GNU AGPL 3"
# __version__ = "1.3.3"
# __email__ = "aeneas@readbeyond.it"
# __status__ = "Production"

echo "[INFO] Running all unit tests..."

python -m unittest discover

echo "[INFO] Running all unit tests... done"
