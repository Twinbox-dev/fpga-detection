#!/bin/sh
insmod cmadrv.ko
export LD_LIBRARY_PATH=./opencv_lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=./paddlelite_lib:$LD_LIBRARY_PATH+
chmod u+x ssd_detection
./ssd_detection_del config.txt dog.jpg
