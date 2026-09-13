#!/bin/sh

insmod cmadrv.ko

export LD_LIBRARY_PATH=./opencv_lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=./paddlelite_lib:$LD_LIBRARY_PATH

chmod u+x ssd_detection_multi_images

set -- config.txt

for file in ./input/*; do
    [ -f "$file" ] || continue
    set -- "$@" "$file"
done

./ssd_detection_multi_images "$@"
