#!/bin/bash

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./Paddle-Lite/libs/armv7hf ./build/PersonMonitor ./models/mobilenet_v3_small.nb
