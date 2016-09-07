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

if [ ! -e cmfcc_driver_wo_fo ]
then
    bash 000_compile_driver.sh
fi

echo "Run 1"
./cmfcc_driver_wo_fo
echo ""

echo "Run 2"
./cmfcc_driver_wo_fo ../tools/res/audio.wav /tmp/out.dt.bin data text
echo ""

echo "Run 3"
./cmfcc_driver_wo_fo ../tools/res/audio.wav /tmp/out.db.bin data binary
echo ""

echo "Run 4"
./cmfcc_driver_wo_fo ../tools/res/audio.wav /tmp/out.ft.bin file text
echo ""

echo "Run 5"
./cmfcc_driver_wo_fo ../tools/res/audio.wav /tmp/out.fb.bin file binary
echo ""

