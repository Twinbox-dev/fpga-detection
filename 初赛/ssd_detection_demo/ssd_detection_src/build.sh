#!/bin/bash




rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DLITE_WITE_PROFILE=1 -DPADDLE_LITE_DIR="../Paddlelite" -DDETECTION_TARGET=camera -DCAMERA_TYPE=aiep -DTARGET_ARCH_ABI=armv7hf -DCMAKE_PREFIX_PATH="../Paddlelite/lib" .. 
make
