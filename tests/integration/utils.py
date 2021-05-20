from pathlib import Path

import numpy as np


def abs_diff(x1, x2):
    return np.abs(x1 - x2)


def input_dir() -> Path:
    path = _this_dir().joinpath("input")
    assert path.is_dir()  # sanity check
    return path


def output_dir() -> Path:
    path = _this_dir().joinpath("output")
    assert path.is_dir()  # sanity check
    return path


def _this_dir() -> Path:
    path = Path(__file__).resolve().parent
    assert path.is_dir()  # sanity check
    return path
