#!/bin/bash

# configure
TARGET_ARCH_ABI=armv7hf # for Raspberry Pi 3B
PADDLE_LITE_DIR=./Paddle-Lite

# build
rm -rf build
mkdir build
cd build
cmake -DPADDLE_LITE_DIR=./Paddle-Lite -DTARGET_ARCH_ABI=armv7hf ..
make
