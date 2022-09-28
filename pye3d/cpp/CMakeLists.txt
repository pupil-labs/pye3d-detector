cmake_minimum_required(VERSION 3.8)

project("pye3d")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(APPLE)
    # We target macOS 10.12, which does not offer c++17, but we can use c++1z instead.
    # See https://clang.llvm.org/cxx_status.html
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
endif()

# find skbuild cmake packages
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

# find external cmake packages
find_package(NumPy REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# include cpp folder for finding headers
include_directories(".")

# include numpy headers
include_directories(${NumPy_INCLUDE_DIRS})

# Note: add_cython_target does not actually add a target, but fills a variable with the
# corresponding compiled source file, e.g. here 'pupil_detection_3d.cxx'. If only the
# name is specified, it will look for a cython file with the same base name:
# 'pupil_detection_3d.pyx' in this case.
add_cython_target(pupil_detection_3d CXX PY3)

# Create a module library from the source file and wrap it with settings for
# creating a python extension.
add_library(pupil_detection_3d MODULE ${pupil_detection_3d})
python_extension_module(pupil_detection_3d)

# link against external libraries
target_link_libraries(pupil_detection_3d Eigen3::Eigen)

# Same for other cython modules, read comments above for explanation
add_cython_target(projections CXX PY3)
add_library(projections MODULE ${projections})
python_extension_module(projections)
target_link_libraries(projections Eigen3::Eigen)

# Same for other cython modules, read comments above for explanation
add_cython_target(refraction_correction CXX PY3)
add_library(refraction_correction MODULE ${refraction_correction})
python_extension_module(refraction_correction)
target_link_libraries(refraction_correction Eigen3::Eigen)

# install here
install(
    TARGETS pupil_detection_3d projections refraction_correction
    LIBRARY DESTINATION ".")
