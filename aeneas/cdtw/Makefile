# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2017, Alberto Pettarin (www.albertopettarin.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

PREFIX=cdtw

FUNC=$(PREFIX)_func
DRIVER=$(PREFIX)_driver
SETUP_PY=$(PREFIX)_setup.py

SRCS=$(DRIVER).c $(FUNC).c ../cint/cint.c
CLIBS=-lm

CC=gcc
CFLAGS=-Wall -pedantic -std=c99
PYTHON=python



all: driver py

$(DRIVER): $(SRCS)
	$(CC) $(SRCS) -o $(DRIVER) $(CLIBS) $(CFLAGS) 

driver: $(DRIVER)

test: $(DRIVER)
	./$(DRIVER) || exit 0
	./$(DRIVER) 12 3000 ../tests/res/cdtw/mfcc1_12_1332 1332 ../tests/res/cdtw/mfcc2_12_868 868 cm > /dev/null
	./$(DRIVER) 12 3000 ../tests/res/cdtw/mfcc1_12_1332 1332 ../tests/res/cdtw/mfcc2_12_868 868 acm > /dev/null
	./$(DRIVER) 12 3000 ../tests/res/cdtw/mfcc1_12_1332 1332 ../tests/res/cdtw/mfcc2_12_868 868 path > /dev/null

clean_driver:
	rm -rf $(DRIVER)

py: clean_py $(SETUP_PY)
	$(PYTHON) $(SETUP_PY) build_ext --inplace

clean_py:
	rm -rf *.pyc *.so build __pycache__

clean: clean_driver clean_py
