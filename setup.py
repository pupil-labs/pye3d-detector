#!/usr/bin/env python

import os
import platform

from skbuild import setup

package = "pye3d"
cpp_dir = f"{package}/cpp"


cmake_args = []
if os.environ.get("CI", "false") == "true" and platform.system() == "Windows":
    # The Ninja cmake generator will use mingw (gcc) on windows travis instances, but we
    # need to use msvc for compatibility.
    cmake_args.append("-GVisual Studio 17 2022")


if __name__ == "__main__":
    setup(
        cmake_args=cmake_args,
        cmake_install_dir=cpp_dir,
        cmake_source_dir=cpp_dir,
    )
