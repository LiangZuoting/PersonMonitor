#!/bin/bash

root=/home/pi/PersonMonitor

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$root/Paddle-Lite/libs/armv7hf $root/build/PersonMonitor $root/models/mobilenet_v3_small.nb
