cmake_minimum_required (VERSION 3.8)

project("pye3d")

# OpenCV 4 requires at least C++11, but we should be fine even selecting C++17 now
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if (APPLE)
    # We target macOS 10.12, which does not offer c++17, but we can use c++1z instead.
    # See https://clang.llvm.org/cxx_status.html
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
endif ()

# find skbuild cmake packages
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

# find external cmake packages
find_package(NumPy REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# Find OpenCV. Needs to be on cmake search path, or specify OpenCV_DIR variable.
find_package(OpenCV REQUIRED)

# Note: add_cython_target does not actually add a target, but fills a variable with the
# corresponding compiled source file, e.g. here 'search_3d.cxx'. If only the name is
# specified, it will look for a cython file with the same base name: 'search_3d.pyx'.
add_cython_target(search_3d CXX PY3)

# Create a module library from the source file and wrap it with settings for
# creating a python extension.
add_library(search_3d MODULE ${search_3d})
python_extension_module(search_3d)

# include folder for search_3d_cpp.h
include_directories(".")
# include numpy headers
include_directories(${NumPy_INCLUDE_DIRS})
# link against external libraries
target_link_libraries(search_3d ${OpenCV_LIBS})
target_link_libraries(search_3d Eigen3::Eigen)

# install here
install(TARGETS search_3d LIBRARY DESTINATION ".")
