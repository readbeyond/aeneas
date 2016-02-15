#!/bin/bash

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



