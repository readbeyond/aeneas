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

PREFIX=cfw

FUNC=$(PREFIX)_func
DRIVER=$(PREFIX)_driver
SETUP_PY=$(PREFIX)_setup.py

SRCS=$(DRIVER).cc $(FUNC).cc ../cint/cint.c
CXXLIBS=-Ifestival -lFestival -Ispeech_tools -lestools -lestbase -leststring

CXX=g++
CXXFLAGS=-Wall -pedantic
PYTHON=python



all: driver py

$(DRIVER): $(SRCS)
	$(CXX) $(SRCS) -o $(DRIVER) $(CXXLIBS) $(CXXFLAGS) 

driver: $(DRIVER)

test: $(DRIVER)
	./$(DRIVER) || exit 0
	./$(DRIVER) "(language_english)" "Hello|World|My|Dear|Friend" /tmp/out.wav 0.0 0
	./$(DRIVER) "(language_english)" "Hello|World|My|Dear|Friend" /tmp/out.wav 0.0 1
	./$(DRIVER) "(language_english)" "Hello|World|My|Dear|Friend" /tmp/out.wav 2.0 0
	./$(DRIVER) "(language_english)" "Hello|World|My|Dear|Friend" /tmp/out.wav 2.0 1

clean_driver:
	rm -rf $(DRIVER)

py: clean_py $(SETUP_PY)
	$(PYTHON) $(SETUP_PY) build_ext --inplace

clean_py:
	rm -rf build __pycache__ *.pyc *.so

clean: clean_driver clean_py
