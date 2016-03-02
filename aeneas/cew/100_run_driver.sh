#!/bin/bash

if [ ! -e cew_driver ]
then
    bash 000_compile_driver.sh
fi

echo "Run 1"
./cew_driver
echo ""

echo "Run 2"
./cew_driver en "Hello World" /tmp/out.wav single
echo ""

echo "Run 3"
./cew_driver en "Hello|World|My|Dear|Friend" /tmp/out.wav multi 0.0 0
echo ""

echo "Run 4"
./cew_driver en "Hello|World|My|Dear|Friend" /tmp/out.wav multi 0.0 1
echo ""

echo "Run 4"
./cew_driver en "Hello|World|My|Dear|Friend" /tmp/out.wav multi 2.0 1
echo ""




