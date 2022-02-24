#!/usr/bin/env python

import os
import platform

from setuptools import find_packages
from skbuild import setup

cpp_dir = "pye3d/cpp"
cmake_args = []

if os.environ.get("CI", "false") == "true" and platform.system() == "Windows":
    # The Ninja cmake generator will use mingw (gcc) on windows travis instances, but we
    # need to use msvc for compatibility.
    cmake_args.append("-GVisual Studio 17 2022")


if __name__ == "__main__":
    setup(
        # `packages` cannot be defined in setup.cfg, otherwise the C extensions will
        # not be copied to the correct place
        packages=find_packages(),
        cmake_args=cmake_args,
        cmake_install_dir=cpp_dir,
        cmake_source_dir=cpp_dir,
        package_data={"pye3d": ["refraction_models/*.msgpack"]},
    )
