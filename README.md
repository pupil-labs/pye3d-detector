# `pye3d`

![pye3d CI](https://github.com/pupil-labs/pye3d-detector/workflows/pye3d%20CI/badge.svg)

This is a python implementation of the 3D eye model. 

## Build

### Build Dependencies

#### Windows

##### Build and install OpenCV

```bash
# Download OpenCV
Invoke-WebRequest "https://github.com/opencv/opencv/archive/4.2.0.zip" -OutFile opencv.zip

# Prepare build
Expand-Archive opencv.zip
mv opencv/opencv-4.*/* opencv/
mkdir opencv/build

# Enter build path
cd opencv/build

# Configure build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="." -DBUILD_LIST="core,highgui,videoio,imgcodecs,imgproc,video" -DBUILD_opencv_world=ON -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DWITH_OPENMP=ON -DWITH_IPP=ON -DWITH_CSTRIPES=ON -DWITH_OPENCL=ON -DWITH_CUDA=OFF -DWITH_TBB=OFF -DWITH_MSMF=OFF

# Compile
cmake --build . --target INSTALL --config Release --parallel

# Exit build path
cd ../..
```

##### Build and install Eigen

```bash
# Download Eigen
Invoke-WebRequest "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip" -OutFile eigen.zip

# Prepare build
Expand-Archive eigen.zip
mv eigen/eigen-3.*/* eigen/
mkdir eigen/build

# Enter build path
cd eigen/build

# Configure build
cmake .. -DCMAKE_INSTALL_PREFIX="."

# Compile
cmake --build . --target INSTALL --config Release --parallel

# Exit build path
cd ../..
```

#### Ubuntu

##### Build and install OpenCV

```bash
# Download OpenCV
wget -q -O opencv.zip "https://github.com/opencv/opencv/archive/4.2.0.zip"

# Prepare build
unzip -q opencv.zip
mv opencv-4.* opencv/
mkdir opencv/build

# Enter build path
cd opencv/build

# Configure build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="." \
    -DBUILD_LIST="core,highgui,videoio,imgcodecs,imgproc,video" \
    -DBUILD_opencv_world=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_opencv_java=OFF \
    -DBUILD_opencv_python=OFF \
    -DWITH_OPENMP=ON \
    -DWITH_IPP=ON \
    -DWITH_CSTRIPES=ON \
    -DWITH_OPENCL=ON \
    -DWITH_CUDA=OFF \
    -DWITH_TBB=OFF \
    -DWITH_MSMF=OFF

# Compile
make
make install

# Exit build path
cd ../..
```

##### Build and install Eigen

```bash
# Download Eigen
wget -q -O eigen.zip "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip"

# Prepare build
unzip -q eigen.zip
mv eigen-3.* eigen/
mkdir eigen/build

# Enter build path
cd eigen/build

# Configure build
cmake .. \
    -DCMAKE_INSTALL_PREFIX="."

# Compile
make
make install

# Exit build path
cd ../..
```

#### macOS

##### Build and install OpenCV

```bash
# Download OpenCV
wget -q -O opencv.zip "https://github.com/opencv/opencv/archive/4.2.0.zip"

# Prepare build
unzip -q opencv.zip
mv opencv-4.* opencv/
mkdir opencv/build

# Enter build path
cd opencv/build

# Configure build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="." \
    -DBUILD_LIST="core,highgui,videoio,imgcodecs,imgproc,video" \
    -DBUILD_opencv_world=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_DOCS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_opencv_java=OFF \
    -DBUILD_opencv_python=OFF \
    -DWITH_OPENMP=ON \
    -DWITH_IPP=ON \
    -DWITH_CSTRIPES=ON \
    -DWITH_OPENCL=ON \
    -DWITH_CUDA=OFF \
    -DWITH_TBB=OFF \
    -DWITH_MSMF=OFF

# Compile
make
make install

# Exit build path
cd ../..
```

##### Build and install Eigen

```bash
# Download Eigen
wget -q -O eigen.zip "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip"

# Prepare build
unzip -q eigen.zip
mv eigen-3.* eigen/
mkdir eigen/build

# Enter build path
cd eigen/build

# Configure build
cmake .. \
    -DCMAKE_INSTALL_PREFIX="."

# Compile
make
make install

# Exit build path
cd ../..
```
