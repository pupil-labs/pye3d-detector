"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import os
from pathlib import Path
import platform

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
    "scikit-learn==0.24.1",
    "sortedcontainers",
]
extras_require = {
    "dev": ["pytest", "tox", "bump2version", "pandas", "scikit-image"],
    "with-opencv": ["opencv-python"],
}

cmake_args = []
if os.environ.get("CI", "false") == "true" and platform.system() == "Windows":
    # The Ninja cmake generator will use mingw (gcc) on windows travis instances, but we
    # need to use msvc for compatibility.
    cmake_args.append("-GVisual Studio 16 2019")

setup(
    author="Pupil Labs",
    author_email="info@pupil-labs.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
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
    project_urls={
        "Changelog": "https://github.com/pupil-labs/pye3d-detector/blob/master/CHANGELOG.md",
        "Pupil Core Documentation": "https://docs.pupil-labs.com/core/",
        "Pupil Labs Homepage": "https://pupil-labs.com/",
    },
    name=package,
    package_data={package: ["refraction_models/*.save"]},
    packages=find_packages(),
    url="https://github.com/pupil-labs/pye3d-detector/",
    version="0.1.1",
)
