#!/bin/bash
set -e

# Opencv
if [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache win opencv]" ]] || \
    [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache win]" ]] || \
    [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache]" ]]
then
    echo "CLEARING OPENCV CACHE..."
    echo "Triggered by commit msg: $TRAVIS_COMMIT_MESSAGE"
    rm -rf dependencies/opencv
fi
echo "Checking OpenCV cache..."
if [[ -d dependencies/opencv ]]
then
    echo "Found OpenCV cache. Build configuration:"
    dependencies/opencv/x64/vc15/bin/opencv_version.exe -v
else
    cd dependencies
    echo "OpenCV cache missing. Rebuilding..."
    wget -q -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
    unzip -q opencv.zip
    cd opencv-4.2.0
    mkdir -p build
    cd build
    # MSMF: see https://github.com/skvark/opencv-python/issues/263
    # CUDA/TBB: turned off because not easy to install on Windows and we cannot easily
    # ship this with the wheel.
    cmake ..\
        -G"Visual Studio 15 2017 Win64"\
        -DCMAKE_BUILD_TYPE=Release\
        -DCMAKE_INSTALL_PREFIX=../../opencv\
        -DBUILD_LIST=core,highgui,videoio,imgcodecs,imgproc,video\
        -DBUILD_opencv_world=ON\
        -DBUILD_EXAMPLES=OFF\
        -DBUILD_DOCS=OFF\
        -DBUILD_PERF_TESTS=OFF\
        -DBUILD_TESTS=OFF\
        -DBUILD_opencv_java=OFF\
        -DBUILD_opencv_python=OFF\
        -DWITH_OPENMP=ON\
        -DWITH_IPP=ON\
        -DWITH_CSTRIPES=ON\
        -DWITH_OPENCL=ON\
        -DWITH_CUDA=OFF\
        -DWITH_TBB=OFF\
        -DWITH_MSMF=OFF
    cmake --build . --target INSTALL --config Release --parallel
    cd ../..
    rm -rf opencv.zip
    rm -rf opencv-4.2.0
    cd ..
fi