![pye3d CI](https://github.com/pupil-labs/pye3d-detector/workflows/pye3d%20CI/badge.svg)
[![PyPI version](https://badge.fury.io/py/pye3d.svg)](https://badge.fury.io/py/pye3d)

# pye3d

`pye3d` implements a published mathematical 3D eye model capturing ocular kinematics and optics (see [Swirski and Dodgson, 2013](https://www.researchgate.net/publication/264658852_A_fully-automatic_temporal_approach_to_single_camera_glint-free_3D_eye_model_fitting "Swirski and Dodgson, 2013"), as well as [Dierkes, Kassner, and Bulling, 2019](https://www.researchgate.net/publication/333490770_A_fast_approach_to_refraction-aware_eye-model_fitting_and_gaze_prediction "Dierkes, Kassner, and Bulling, 2019")).

## Build and install from source

### Build Dependencies

You can skip this step if you have OpenCV and Eigen3 compiled and installed on your computer.
Scroll to the bottom and continue with installing pye3d.

#### Windows

Building the dependencies on Windows requires running the commands in [PowerShell](https://docs.microsoft.com/en-us/powershell/scripting/overview). Requires `git` and `cmake` to be in your system PATH. Please run all three install steps in the same shell or redefine `OpenCV_DIR` and `Eigen3_DIR` before running the last step (building and installing pye3d).

##### Build and install OpenCV

```powershell
# Download OpenCV
Invoke-WebRequest "https://github.com/opencv/opencv/archive/4.2.0.zip" -OutFile opencv.zip

# Prepare build
Expand-Archive opencv.zip
mv opencv/opencv-4.*/* opencv/
mkdir opencv/build

# Enter build path
cd opencv/build

# Configure build
cmake .. -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="." -DBUILD_LIST="core,highgui,videoio,imgcodecs,imgproc,video" -DBUILD_opencv_world=ON -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DBUILD_opencv_java=OFF -DBUILD_opencv_python=OFF -DWITH_OPENMP=ON -DWITH_IPP=ON -DWITH_CSTRIPES=ON -DWITH_OPENCL=ON -DWITH_CUDA=OFF -DWITH_TBB=OFF -DWITH_MSMF=OFF

# Compile
cmake --build . --target INSTALL --config Release --parallel

# Define OpenCV location for third step
$Env:OpenCV_DIR = (pwd)

# Exit build path
cd ../..
```

##### Build and install Eigen

```powershell
# Download Eigen
Invoke-WebRequest "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip" -OutFile eigen.zip

# Prepare build
Expand-Archive eigen.zip
mv eigen/eigen-3.*/* eigen/
mkdir eigen/build

# Enter build path
cd eigen/build

# Configure build
cmake .. -A x64 -DCMAKE_INSTALL_PREFIX="."

# Compile
cmake --build . --target INSTALL --config Release --parallel

# Define Eigen3 location for third step
$Env:Eigen3_DIR = (pwd)

# Exit build path
cd ../..
```

Scroll to the bottom for the last step.

#### Ubuntu

##### Build and install OpenCV

```bash
# Download OpenCV
wget -O opencv.zip "https://github.com/opencv/opencv/archive/4.2.0.zip"

# Prepare build
unzip opencv.zip
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

# Define OpenCV location for third step
OpenCV_DIR=${pwd}

# Exit build path
cd ../..
```

##### Build and install Eigen

```bash
# Download Eigen
wget -O eigen.zip "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip"

# Prepare build
unzip eigen.zip
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

# Define Eigen3 location for third step
Eigen3_DIR=${pwd}

# Exit build path
cd ../..
```

Scroll to the bottom for the last step.

#### macOS

Downlaoding the dependencies requires `wget`, which can be installed on macOS with [Homebrew](https://brew.sh):
```bash
brew install wget
```

##### Build and install OpenCV

```bash
# Download OpenCV
wget -O opencv.zip "https://github.com/opencv/opencv/archive/4.2.0.zip"

# Prepare build
unzip opencv.zip
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

# Define OpenCV location for third step
OpenCV_DIR=${pwd}

# Exit build path
cd ../..
```

##### Build and install Eigen

```bash
# Download Eigen
wget -O eigen.zip "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip"

# Prepare build
unzip eigen.zip
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

# Define Eigen3 location for third step
Eigen3_DIR=${pwd}

# Exit build path
cd ../..
```


### Build and install `pye3d` (all platforms)

Requires `OpenCV_DIR` and `Eigen3_DIR` environmental variables pointing to the appropriate install locations.

```powershell
# Download pye3d code
git clone https://github.com/pupil-labs/pye3d-detector.git

# Prepare build
cd pye3d-detector
# optional: activate Python virtual environment here

# Make sure pip is up-to-date
python -m pip install -U pip

# Build and install pye3d
python -m pip install .
```
