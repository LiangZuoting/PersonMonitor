#!/bin/bash

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./Paddle-Lite/libs/armv7hf ./PersonMonitor ../models/ssd_mobilenet_v1_pascalvoc_for_cpu/model.nb
