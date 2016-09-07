#!/bin/bash

# aeneas is a Python/C library and a set of tools
# to automagically synchronize audio and text (aka forced alignment)
#
# Copyright (C) 2012-2013, Alberto Pettarin (www.albertopettarin.it)
# Copyright (C) 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
# Copyright (C) 2015-2016, Alberto Pettarin (www.albertopettarin.it)
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

if [ ! -e cwave_driver ]
then
    bash 000_compile_driver.sh
fi

echo "Run 1"
./cwave_driver
echo ""

echo "Run 2"
./cwave_driver ../tools/res/audio.wav
echo ""

echo "Run 3"
./cwave_driver ../tools/res/audio.wav 0 10
echo ""

echo "Run 4"
./cwave_driver ../tools/res/audio.wav 5 5
echo ""

echo "Run 5"
./cwave_driver ../tests/res/audioformats/mono.empty.wav
./cwave_driver ../tests/res/audioformats/mono.invalid.wav
./cwave_driver ../tests/res/audioformats/mono.zero.wav
./cwave_driver ../tests/res/audioformats/mono.16000.wav
./cwave_driver ../tests/res/audioformats/mono.22050.wav
./cwave_driver ../tests/res/audioformats/mono.44100.wav
./cwave_driver ../tests/res/audioformats/mono.48000.wav
echo ""

