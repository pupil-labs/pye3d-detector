#!/bin/bash
set -e

# Opencv
echo "Checking OpenCV cache..."
if [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache linux opencv]" ]] || \
    [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache linux]" ]] || \
    [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache]" ]]
then
    echo "CLEARING OPENCV CACHE..."
    echo "Triggered by commit msg: $TRAVIS_COMMIT_MESSAGE"
    rm -rf dependencies/opencv
fi
if [[ -d dependencies/opencv ]]
then
    echo "Found OpenCV cache. Build configuration:"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/dependencies/opencv/lib64
    dependencies/opencv/bin/opencv_version -v
else
    echo "OpenCV cache missing. Rebuilding..."
    chmod +x .travis/setup_opencv_manylinux.sh
    docker run --rm -v `pwd`:/io quay.io/pypa/manylinux2014_x86_64 /io/.travis/setup_opencv_manylinux.sh
fi
