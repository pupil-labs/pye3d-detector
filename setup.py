import os
import platform
from pathlib import Path

from setuptools import find_packages

from skbuild import setup

here = Path(__file__).parent

package = "pye3d"
cpp_dir = f"{package}/cpp"


with open(here / "README.md") as f:
    long_description = f.read()

requirements = [
    "scipy>=1.2.1",
    "numpy",
    "joblib",
    "opencv-python",
    "scikit-learn",
]
extras_require = {"dev": ["pytest", "tox"]}

cmake_args = []
if platform.system() == "Windows":
    # The Ninja cmake generator will use mingw (gcc) on windows travis instances, but we
    # need to use msvc for compatibility.
    cmake_args.append("-GVisual Studio 15 2017 Win64")

elif platform.system() == "Darwin":
    # This is for building wheels with opencv included. OpenCV dylibs have their
    # install_name set to @rpath/xxx.dylib and we need to tell the linker for the python
    # module where to find the libs.
    opencv_dir = os.environ.get("OpenCV_DIR", None)
    if opencv_dir:
        cmake_args.append(f'-DCMAKE_MODULE_LINKER_FLAGS="-Wl,-rpath,{opencv_dir}/lib"')

setup(
    author="Pupil Labs",
    author_email="pypa@pupil-labs.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3"
        "Topic :: Scientific/Engineering"
        "Topic :: Software Development",
    ],
    cmake_args=cmake_args,
    cmake_install_dir=cpp_dir,
    cmake_source_dir=cpp_dir,
    description="3D eye model",
    extras_require=extras_require,
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    name=package,
    package_data={package: ["refraction_models/*.save"]},
    packages=find_packages(),
    url="",
    version="0.0.1",
)
