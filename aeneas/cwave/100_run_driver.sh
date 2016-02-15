#!/bin/bash

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

