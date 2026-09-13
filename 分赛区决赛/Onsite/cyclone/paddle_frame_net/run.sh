#!/bin/bash
insmod cmadrv.ko
export LD_LIBRARY_PATH=/opt/paddle_frame_net:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/paddle_frame_net/opencv_lib:$LD_LIBRARY_PATH
export PYTHONIOENCODING=utf-8
rmmod nnadrv.ko
insmod cmadrv.ko
rmmod cmadrv.ko
insmod cmadrv.ko
python ssd_detection.py
