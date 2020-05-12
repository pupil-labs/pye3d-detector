#!/bin/bash
set -e

# Opencv
echo "Checking OpenCV cache..."
if [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache mac opencv]" ]] || \
    [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache mac]" ]] || \
    [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache]" ]]
then
    echo "CLEARING OPENCV CACHE..."
    echo "Triggered by commit msg: $TRAVIS_COMMIT_MESSAGE"
    rm -rf dependencies/opencv
fi
if [[ -d dependencies/opencv ]]
then
    echo "Found OpenCV cache. Build configuration:"
    dependencies/opencv/bin/opencv_version -v
else
    echo "OpenCV cache missing. Rebuilding..."
    cd dependencies
    wget -q -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
    unzip -q opencv.zip
    cd opencv-4.2.0
    mkdir -p build
    cd build
    # opencv 4.x needs cpp11 support
    cmake ..\
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
        -DWITH_TBB=OFF\
        -DWITH_CUDA=OFF
    make -j2 && make install
    cd ../..
    rm -rf opencv.zip
    rm -rf opencv-4.2.0
    cd ..
fi


# Eigen3
echo "Checking eigen3 cache..."
if [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache mac eigen]" ]] || \
    [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache mac]" ]] || \
    [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache]" ]]
then
    echo "CLEARING EIGEN CACHE..."
    echo "Triggered by commit msg: $TRAVIS_COMMIT_MESSAGE"
    rm -rf dependencies/eigen3
fi
if [[ -d dependencies/eigen3 ]]
then
    echo "Found eigen3 cache."
else
    echo "Eigen3 cache missing. Downloading..."
    cd dependencies
    wget -q -O eigen.zip https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip
    unzip -q eigen.zip
    cd eigen-3.3.7
    mkdir -p build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../../eigen3
    make && make install
    cd ../..
    rm -rf eigen-3.3.7
    rm -rf eigen.zip
    cd ..
fi

# Python
export PYENV_ROOT=${PWD}/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
echo "Checking pyenv cache..."
if [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache mac pyenv]" ]] || \
    [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache mac]" ]] || \
    [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache]" ]]
then
    echo "CLEARING PYENV CACHE..."
    echo "Triggered by commit msg: $TRAVIS_COMMIT_MESSAGE"
    rm -rf .pyenv
fi
if [[ -d .pyenv/bin ]]
then
    echo "Found pyenv cache. Installed versions:"
    eval "$(pyenv init -)"
    pyenv versions
else
    echo "pyenv cache missing. Installing..."
    git clone https://github.com/pyenv/pyenv.git .pyenv
    eval "$(pyenv init -)"
    pyenv install 3.6.8
    pyenv install 3.7.6
    pyenv install 3.8.1
fi
