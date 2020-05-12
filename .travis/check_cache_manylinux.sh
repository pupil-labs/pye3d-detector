#!/bin/bash

# NOTE: Checking is done with a separate script from installing the dependencies, since
# installing is done in the manylinux image, which takes time to download and startup.
# Checking can be done from outside, which makes it a lot faster.
set -e

echo "Checking OpenCV cache..."
if [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache linux opencv]" ]] || \
    [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache linux]" ]] || \
    [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache]" ]]
then
    echo "CLEARING OPENCV CACHE..."
    echo "Triggered by commit msg: $TRAVIS_COMMIT_MESSAGE"
    rm -rf dependencies/opencv
fi

echo "Checking eigen3 cache..."
if [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache linux eigen]" ]] || \
    [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache linux]" ]] || \
    [[ $TRAVIS_COMMIT_MESSAGE =~ "[travis: clear-cache]" ]]
then
    echo "CLEARING EIGEN CACHE..."
    echo "Triggered by commit msg: $TRAVIS_COMMIT_MESSAGE"
    rm -rf dependencies/eigen3
fi

if [[ -d dependencies/opencv ]] && \
    [[ -d dependencies/eigen3 ]]
then
    echo "Found all cache entries."
else
    echo "Cache entries missing. Rebuilding..."
    chmod +x .travis/setup_manylinux.sh
    docker run --rm -v `pwd`:/io quay.io/pypa/manylinux2014_x86_64 /io/.travis/setup_manylinux.sh
fi
