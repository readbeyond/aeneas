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

PREFIX=cmfcc

FUNC=$(PREFIX)_func
DRIVER=$(PREFIX)_driver
DRIVER_WO_FO=$(DRIVER)_wo_fo
DRIVER_WO_FF=$(DRIVER)_wo_ff
DRIVER_WS_FO=$(DRIVER)_ws_fo
DRIVER_WS_FF=$(DRIVER)_ws_ff
SETUP_PY=$(PREFIX)_setup.py

SRCS=$(DRIVER).c $(FUNC).c ../cwave/cwave_func.c ../cint/cint.c
CLIBS=-lm

CC=gcc
CFLAGS=-Wall -pedantic -std=c99
PYTHON=python



all: driver py

$(DRIVER_WO_FO): $(SRCS)
	$(CC) $(SRCS) -o $(DRIVER_WO_FO) $(CLIBS) $(CFLAGS) 

$(DRIVER_WO_FF): $(SRCS)
	$(CC) $(SRCS) -o $(DRIVER_WO_FF) $(CLIBS) $(CFLAGS) -lrfftw -lfftw -DUSE_FFTW

$(DRIVER_WS_FO): $(SRCS)
	$(CC) $(SRCS) -o $(DRIVER_WS_FO) $(CLIBS) $(CFLAGS) -lsndfile -DUSE_SNDFILE

$(DRIVER_WS_FF): $(SRCS)
	$(CC) $(SRCS) -o $(DRIVER_WS_FF) $(CLIBS) $(CFLAGS) -lsndfile -lrfftw -lfftw  -DUSE_SNDFILE -DUSE_FFTW

driver: $(DRIVER_WO_FO) $(DRIVER_WO_FF) $(DRIVER_WS_FO) $(DRIVER_WS_FF)

test_wo_fo: $(DRIVER_WO_FO)
	./$(DRIVER_WO_FO) || exit 0
	./$(DRIVER_WO_FO) ../tools/res/audio.wav /tmp/out.dt.bin data text
	./$(DRIVER_WO_FO) ../tools/res/audio.wav /tmp/out.db.bin data binary
	./$(DRIVER_WO_FO) ../tools/res/audio.wav /tmp/out.ft.bin file text
	./$(DRIVER_WO_FO) ../tools/res/audio.wav /tmp/out.fb.bin file binary

test_wo_ff: $(DRIVER_WO_FF)
	./$(DRIVER_WO_FF) || exit 0
	./$(DRIVER_WO_FF) ../tools/res/audio.wav /tmp/out.dt.bin data text
	./$(DRIVER_WO_FF) ../tools/res/audio.wav /tmp/out.db.bin data binary
	./$(DRIVER_WO_FF) ../tools/res/audio.wav /tmp/out.ft.bin file text
	./$(DRIVER_WO_FF) ../tools/res/audio.wav /tmp/out.fb.bin file binary

test_ws_fo: $(DRIVER_WS_FO)
	./$(DRIVER_WS_FO) || exit 0
	./$(DRIVER_WS_FO) ../tools/res/audio.wav /tmp/out.dt.bin data text
	./$(DRIVER_WS_FO) ../tools/res/audio.wav /tmp/out.db.bin data binary
	./$(DRIVER_WS_FO) ../tools/res/audio.wav /tmp/out.ft.bin file text
	./$(DRIVER_WS_FO) ../tools/res/audio.wav /tmp/out.fb.bin file binary

test_ws_ff: $(DRIVER_WS_FF)
	./$(DRIVER_WS_FF) || exit 0
	./$(DRIVER_WS_FF) ../tools/res/audio.wav /tmp/out.dt.bin data text
	./$(DRIVER_WS_FF) ../tools/res/audio.wav /tmp/out.db.bin data binary
	./$(DRIVER_WS_FF) ../tools/res/audio.wav /tmp/out.ft.bin file text
	./$(DRIVER_WS_FF) ../tools/res/audio.wav /tmp/out.fb.bin file binary

test: test_wo_fo test_wo_ff test_ws_fo test_ws_ff

clean_driver:
	rm -rf $(DRIVER_WO_FO) $(DRIVER_WO_FF) $(DRIVER_WS_FO) $(DRIVER_WS_FF)

py: clean_py $(SETUP_PY)
	$(PYTHON) $(SETUP_PY) build_ext --inplace

clean_py:
	rm -rf *.pyc *.so build __pycache__

clean: clean_driver clean_py
