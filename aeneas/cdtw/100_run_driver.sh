#!/bin/bash

if [ ! -e cdtw_driver ]
then
    bash 000_compile_driver.sh
fi

echo "Run 1"
./cdtw_driver
echo ""

echo "Run 2 (no stdout)"
./cdtw_driver 12 3000 ../tests/res/cdtw/mfcc1_12_1332 1332 ../tests/res/cdtw/mfcc2_12_868 868 cm > /dev/null
echo ""

echo "Run 3 (no stdout)"
./cdtw_driver 12 3000 ../tests/res/cdtw/mfcc1_12_1332 1332 ../tests/res/cdtw/mfcc2_12_868 868 acm > /dev/null
echo ""

echo "Run 4 (no stdout)"
./cdtw_driver 12 3000 ../tests/res/cdtw/mfcc1_12_1332 1332 ../tests/res/cdtw/mfcc2_12_868 868 path > /dev/null
echo ""



